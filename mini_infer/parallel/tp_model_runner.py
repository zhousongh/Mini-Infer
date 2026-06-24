"""
Phase 13 Tensor Parallel 模型执行器（单 rank 视角）。

实现 Megatron-LM 风格的列并行（Column Parallel）和行并行（Row Parallel）权重切分，
通过 PyTorch forward hook 在 o_proj 和 down_proj 输出后注入 NCCL all-reduce。

TP 切分方式（以 Qwen2.5 TP=2 为例）：
  Column Parallel（无通信）：q/k/v_proj, gate_proj, up_proj —— 沿 dim=0（输出维度）切分
  Row Parallel（forward 后 all-reduce）：o_proj, down_proj —— 沿 dim=1（输入维度）切分

  1.5B（12Q/2KV heads，TP=2）：
    rank 0: 6 Q heads（768 dim），1 KV head（128 dim）
    rank 1: 6 Q heads（768 dim），1 KV head（128 dim）

  7B（28Q/4KV heads，TP=2）：
    rank 0: 14 Q heads（1792 dim），2 KV heads（256 dim）
    rank 1: 14 Q heads（1792 dim），2 KV heads（256 dim）
    每卡权重约 9.1 GB（原 18.2 GB 的 50%）

不依赖 mini_infer.kernels.attention 的 paged attention patch；
使用标准 HF DynamicCache 做 KV cache（不集成 Paged KV，专注 TP 正确性验证）。
使用 attn_implementation='flash_attention_2' 确保正确输出（eager 在 transformers 4.43.4 有 mask bug）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

# ──────────────────────────────────────────────
# 工具函数：权重切分
# ──────────────────────────────────────────────


def col_shard(weight: torch.Tensor, rank: int, tp_size: int) -> torch.Tensor:
    """沿 dim=0（输出维度）切分权重，用于列并行线性层（Q/K/V/gate/up）。"""
    d_out = weight.shape[0]
    if d_out % tp_size != 0:
        raise ValueError(
            f"col_shard: 输出维度 {d_out} 不能被 tp_size={tp_size} 整除"
        )
    chunk = d_out // tp_size
    return weight[rank * chunk : (rank + 1) * chunk].contiguous()


def row_shard(weight: torch.Tensor, rank: int, tp_size: int) -> torch.Tensor:
    """沿 dim=1（输入维度）切分权重，用于行并行线性层（O/down）。"""
    d_in = weight.shape[1]
    if d_in % tp_size != 0:
        raise ValueError(
            f"row_shard: 输入维度 {d_in} 不能被 tp_size={tp_size} 整除"
        )
    chunk = d_in // tp_size
    return weight[:, rank * chunk : (rank + 1) * chunk].contiguous()


# ──────────────────────────────────────────────
# 核心：Qwen2/Qwen2.5 权重切分
# ──────────────────────────────────────────────


def _shard_qwen2_weights(
    model: nn.Module, rank: int, tp_size: int
) -> None:
    """
    就地切分 Qwen2/Qwen2.5 模型所有 decoder layer 的权重。

    Column Parallel（dim=0 切分）：q_proj, k_proj, v_proj, gate_proj, up_proj
    Row Parallel（dim=1 切分）：o_proj, down_proj
    Bias（若存在）：column 层的 bias 随权重 col_shard；row 层 bias 仅 rank 0 保留。

    同时更新 attn 模块的 num_heads / num_key_value_heads / num_key_value_groups /
    hidden_size，使 Qwen2Attention.forward() 中的 reshape 与切分后形状匹配。

    lm_head / embed_tokens 不切分（各 rank 独立持有完整副本）。
    """
    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        # ── Column parallel projections ──
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            new_w = col_shard(proj.weight.data, rank, tp_size)
            proj.weight = nn.Parameter(new_w, requires_grad=False)
            if proj.bias is not None:
                new_b = col_shard(proj.bias.data.unsqueeze(-1), rank, tp_size).squeeze(-1)
                proj.bias = nn.Parameter(new_b, requires_grad=False)

        for proj in (mlp.gate_proj, mlp.up_proj):
            new_w = col_shard(proj.weight.data, rank, tp_size)
            proj.weight = nn.Parameter(new_w, requires_grad=False)
            # Qwen2 FFN 没有 bias，保留逻辑以备扩展
            if proj.bias is not None:
                new_b = col_shard(proj.bias.data.unsqueeze(-1), rank, tp_size).squeeze(-1)
                proj.bias = nn.Parameter(new_b, requires_grad=False)

        # ── Row parallel projections ──
        for proj in (attn.o_proj, mlp.down_proj):
            new_w = row_shard(proj.weight.data, rank, tp_size)
            proj.weight = nn.Parameter(new_w, requires_grad=False)
            # Row parallel 中 bias 只在 rank 0 加（避免多 rank 重复累加）
            if proj.bias is not None and rank != 0:
                proj.bias = nn.Parameter(
                    torch.zeros_like(proj.bias), requires_grad=False
                )

        # ── 更新 attention 模块属性，使 forward 中的 reshape 与切分后形状匹配 ──
        # Qwen2Attention.forward() 中：
        #   query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        #   attn_output.reshape(bsz, q_len, self.hidden_size)  ← 需要是切分后的 dim
        assert attn.num_heads % tp_size == 0, (
            f"num_heads={attn.num_heads} 不能被 tp_size={tp_size} 整除"
        )
        assert attn.num_key_value_heads % tp_size == 0, (
            f"num_key_value_heads={attn.num_key_value_heads} 不能被 tp_size={tp_size} 整除"
        )
        attn.num_heads = attn.num_heads // tp_size
        attn.num_key_value_heads = attn.num_key_value_heads // tp_size
        # num_key_value_groups 不变（(28//2) / (4//2) = 7 = 28/4）
        attn.num_key_value_groups = attn.num_heads // attn.num_key_value_heads
        # hidden_size 改为每 rank 的注意力输出维度（= num_heads_per_rank * head_dim）
        # 这影响 attn_output.reshape(bsz, q_len, self.hidden_size) 的目标大小
        # o_proj 的输入 dim 也是这个值（row_shard 已切分为此大小）
        attn.hidden_size = attn.num_heads * attn.head_dim


# ──────────────────────────────────────────────
# All-reduce hook 注入
# ──────────────────────────────────────────────


def _register_tp_allreduce_hooks(
    model: nn.Module,
    dist_group: Optional[dist.ProcessGroup] = None,
) -> list:
    """
    在每个 decoder layer 的 self_attn 和 mlp 模块上注册 forward hook，
    在输出后做 NCCL all-reduce（SUM）以完成行并行的归约。

    self_attn 返回 (hidden_states, attn_weights, past_key_value)，
    对 output[0] 做 all-reduce。
    mlp 返回单个 tensor，直接 all-reduce。

    返回 hook handle 列表（可用 handle.remove() 撤销）。
    """
    handles = []

    def make_allreduce_hook() -> callable:
        def hook(module: nn.Module, inputs: tuple, output):
            if isinstance(output, tuple):
                # Qwen2Attention 返回 (hidden_states, attn_weights, past_key_value)
                tensor = output[0].contiguous()
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dist_group)
                return (tensor,) + output[1:]
            else:
                output = output.contiguous()
                dist.all_reduce(output, op=dist.ReduceOp.SUM, group=dist_group)
                return output

        return hook

    for layer in model.model.layers:
        h1 = layer.self_attn.register_forward_hook(make_allreduce_hook())
        h2 = layer.mlp.register_forward_hook(make_allreduce_hook())
        handles.extend([h1, h2])

    return handles


# ──────────────────────────────────────────────
# 主类：TensorParallelModelRunner
# ──────────────────────────────────────────────


class TensorParallelModelRunner:
    """
    Tensor Parallel 单进程模型执行器（由 TPEngine 的每个 rank worker 实例化）。

    使用方式（通常由 TPEngine._tp_worker 调用）：
        runner = TensorParallelModelRunner(
            model_path="...", rank=0, tp_size=2,
            device="cuda:0", dist_group=None,
        )
        results = runner.generate(["prompt"], max_new_tokens=64)

    tp_size=1 时退化为单卡推理（不做权重切分，不注册 all-reduce hook）。
    """

    def __init__(
        self,
        model_path: str,
        rank: int,
        tp_size: int,
        device: str,
        dist_group: Optional[dist.ProcessGroup] = None,
        dtype: str = "float16",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.rank = rank
        self.tp_size = tp_size
        self.device = device
        self.dist_group = dist_group

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        if rank == 0:
            print(f"[TP rank {rank}/{tp_size}] 加载 {model_path} → {device} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # attn_implementation="flash_attention_2" 产生正确输出（eager 在 transformers
        # 4.43.4 下存在 attention mask bug 导致乱码输出）。
        # Qwen2FlashAttention2 与 Qwen2Attention 使用相同的 num_heads / hidden_size
        # reshape 模式，权重切分策略不变。
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

        # 就地切分权重并注册 all-reduce hooks
        if tp_size > 1:
            _shard_qwen2_weights(self.model, rank, tp_size)
            self._hook_handles = _register_tp_allreduce_hooks(
                self.model, dist_group
            )
        else:
            self._hook_handles = []

        if rank == 0:
            if torch.cuda.is_available():
                used_gb = torch.cuda.memory_allocated(device) / 1e9
                print(
                    f"[TP rank {rank}/{tp_size}] 初始化完成，"
                    f"显存 {used_gb:.2f} GB"
                )

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
    ) -> list[str]:
        """
        对 prompts 列表做贪心生成（greedy decode）。

        所有 rank 需同时调用此方法（all-reduce hook 要求所有 rank 同步执行 forward）。
        只有 rank 0 的返回值有意义；其他 rank 返回值与 rank 0 相同（各自独立解码）。

        synced_gpus=True：确保所有 GPU 在每个生成步同步，防止不同 rank 因 EOS 时间
        不同而导致 collective 操作 hang。
        """
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    synced_gpus=(self.tp_size > 1),
                )
            # 只解码新生成的 token（去掉 prompt 部分）
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = output_ids[0, prompt_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(text)
        return results
