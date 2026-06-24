"""
Phase 13 真正的 Tensor Parallel 引擎（旧接口已被替换）。

旧接口说明：
  原 tp_engine.py 是 PPEngine 的别名（device_map="balanced"），实现的是
  Pipeline Parallel（PP）而非真正的 Tensor Parallel（TP）。
  Phase 13 将此文件重写为真 TP 引擎（Megatron-LM 风格，NCCL all-reduce）。
  如需 PP 功能，请直接使用 mini_infer.parallel.pp_engine.PPEngine。

TPEngine 使用方式：
    engine = TPEngine(
        model_path="/path/to/Qwen2.5-1.5B-Instruct",
        tp_size=2,
        dtype="float16",
    )
    results = engine.generate(["你好，介绍一下自己"], max_new_tokens=64)
    print(results[0])

内部实现：
  - 使用 torch.multiprocessing.spawn 启动 tp_size 个 worker 进程
  - 每个 worker 实例化 TensorParallelModelRunner（含权重切分和 all-reduce hook）
  - Rank 0 将生成结果写入临时 JSON 文件，主进程读取后返回
  - 使用文件锁（file:// rendezvous）初始化 dist.ProcessGroup，避免端口冲突
"""

from __future__ import annotations

import json
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ──────────────────────────────────────────────
# Worker 函数（每个 rank 进程运行）
# ──────────────────────────────────────────────


def _tp_worker(
    rank: int,
    tp_size: int,
    model_path: str,
    dtype: str,
    prompts: list[str],
    max_new_tokens: int,
    result_file: str,
    rendezvous_file: str,
) -> None:
    """
    每个 TP rank 的 worker 函数，由 mp.spawn 调用。

    所有 rank 同时执行 generate()；all-reduce hook 保证各 rank forward 同步。
    只有 rank 0 把结果写入 result_file（JSON）。
    """
    from mini_infer.parallel.tp_model_runner import TensorParallelModelRunner

    init_method = f"file://{rendezvous_file}"
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=tp_size,
    )

    device = f"cuda:{rank}"
    runner = TensorParallelModelRunner(
        model_path=model_path,
        rank=rank,
        tp_size=tp_size,
        device=device,
        dist_group=None,  # 使用默认进程组
        dtype=dtype,
    )

    results = runner.generate(prompts, max_new_tokens=max_new_tokens)

    # 只有 rank 0 负责写结果
    if rank == 0:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    dist.destroy_process_group()


# ──────────────────────────────────────────────
# TPEngine 主类
# ──────────────────────────────────────────────


class TPEngine:
    """
    Tensor Parallel 推理引擎（Phase 13，真 TP，NCCL all-reduce）。

    与 PPEngine 的区别：
      PPEngine：不同层串行运行在不同 GPU（Pipeline Parallel），吞吐不提升
      TPEngine：同一层的 head/neuron 并行切分到多 GPU，前向后 all-reduce 合并

    参数：
      model_path: 模型本地路径（支持 Qwen2.5 系列）
      tp_size: TP 并行度，默认 2（需要对应数量的 GPU）
      dtype: 权重 dtype，默认 "float16"
    """

    def __init__(
        self,
        model_path: str,
        tp_size: int = 2,
        dtype: str = "float16",
    ) -> None:
        if tp_size < 1:
            raise ValueError(f"tp_size 必须 >= 1，当前 {tp_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("TPEngine 需要 CUDA GPU")
        if torch.cuda.device_count() < tp_size:
            raise RuntimeError(
                f"需要 {tp_size} 个 GPU，但只有 {torch.cuda.device_count()} 个"
            )

        self.model_path = model_path
        self.tp_size = tp_size
        self.dtype = dtype

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
    ) -> list[str]:
        """
        对 prompts 列表做贪心生成。

        内部通过 mp.spawn 启动 tp_size 个进程，完成 TP forward + all-reduce，
        由 rank 0 收集结果并通过临时文件返回给主进程。

        注意：每次调用都会重新 spawn 进程（含模型加载）。
        适合 benchmark 测量，不适合高频调用。
        """
        result_file = tempfile.mktemp(suffix=".json")
        rendezvous_file = tempfile.mktemp()

        try:
            mp.spawn(
                _tp_worker,
                args=(
                    self.tp_size,
                    self.model_path,
                    self.dtype,
                    prompts,
                    max_new_tokens,
                    result_file,
                    rendezvous_file,
                ),
                nprocs=self.tp_size,
                join=True,
            )
            with open(result_file, encoding="utf-8") as f:
                return json.load(f)
        finally:
            for path in [result_file, rendezvous_file]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
