"""
Phase 4 双卡 Pipeline Parallel 测量引擎（HF Pipeline Parallel）。

使用 HF Transformers 的 device_map="balanced" 将模型层均匀分配到两块 GPU，
直接调用 model.generate() 推理，不使用自定义 Paged KV Cache。

这是 Pipeline Parallel（PP），不是真正的 Tensor Parallel（TP）：
- PP（本引擎）：每层完整运行在一块 GPU 上，层间通信是激活张量 [batch, seq, hidden]
- 真 TP（Tensor Parallel）：每层按 head 维度切分，层内通信是 all-reduce（Megatron-LM 风格）

目的：与单卡 Phase 3 和双卡 Replica 对比，评估 HF PP 的吞吐和显存特征。
局限：不集成 Paged KV Cache 和 Continuous Batching，仅供性能测量。

dry_run=True 时为桩实现，不加载真实模型。
"""

import torch

from ..core.config import EngineConfig


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


class _StubPPTokenizer:
    """dry_run 模式下的占位 tokenizer。"""

    pad_token_id = 0
    eos_token_id = -1

    def __call__(
        self,
        texts: list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
    ) -> dict:
        return {
            "input_ids": torch.zeros(len(texts), 4, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
        }

    def batch_decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        return [f"[pp-stub-{i}]" for i in range(token_ids.shape[0])]


class PPEngine:
    """
    HF Pipeline Parallel 双卡引擎（仅用于性能测量，不集成自定义 KV cache）。

    加载方式：device_map="balanced"，Qwen2.5-7B 的 28 层均分到 cuda:0/cuda:1。
    推理路径：HF model.generate()（右填充 batch，greedy decode）。

    这是 Pipeline Parallel，不是 Tensor Parallel：
    - PP：不同层运行在不同 GPU，层间传递激活张量
    - TP：同一层按 head 切分到多 GPU，层内 all-reduce 合并

    适用场景：
    - 评估 PP 双卡相对单卡的吞吐和延迟变化
    - 单卡显存不足时的替代方案（每卡显存减半）

    局限：
    - HF KV 不分页，显存随序列增长
    - 无 continuous batching
    - 输出 token 数由 max_new_tokens 统一控制
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

        if config.dry_run:
            self.tokenizer: _StubPPTokenizer | "AutoTokenizer" = _StubPPTokenizer()
            self.model = None
            self._first_device = "cpu"
            return

        if not torch.cuda.is_available():
            raise RuntimeError("PPEngine 需要 CUDA GPU。")
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                f"PPEngine 需要至少 2 块 GPU，当前检测到 {torch.cuda.device_count()} 块。"
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_name = config.tokenizer_name or config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            padding_side="left",  # decoder-only 模型批推理必须左填充，否则注意力位置偏移
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = _resolve_torch_dtype(config.dtype)
        # device_map="balanced"：HF accelerate 将层均匀分配到所有可用 GPU
        # Qwen2.5-7B（28层）在 2×RTX 4090 上：每卡约 14 层，每卡 ~8 GB 权重
        print("PPEngine: 使用 device_map='balanced' 加载模型到 2 块 GPU (Pipeline Parallel) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="balanced",
        )
        self.model.eval()

        # 输入 tensor 需要发送到 embedding 层所在的设备
        self._first_device = str(next(self.model.parameters()).device)
        print(f"PPEngine: embedding 层在 {self._first_device}，hf_device_map 片段：")
        if hasattr(self.model, "hf_device_map"):
            for k, v in list(self.model.hf_device_map.items())[:5]:
                print(f"  {k}: {v}")
            print("  ...")

    def generate(self, prompts: list[str], max_new_tokens: int = 128) -> list[str]:
        """
        批量推理。使用 HF model.generate()，左填充对齐，greedy decode。
        返回仅包含新生成 token 的文本（不含输入 prompt）。
        """
        if self.config.dry_run:
            return [f"[pp-stub:{p[:8]}]" for p in prompts]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self._first_device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 只解码新生成的 token，不含输入 prompt
        new_ids = output_ids[:, prompt_len:]
        return self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
