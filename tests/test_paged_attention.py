"""
Phase 6 PagedAttention 正确性测试。覆盖范围：
  - paged_decode_attention 路径与 HF Transformers 直接 generate 的 token 序列比较
  - 使用 Qwen2.5-7B-Instruct，greedy 采样，短序列（5 decode steps）
  - 在 cuda:0 上运行 mini-infer（paged），在 cuda:1 上运行 HF（baseline），对比输出

需要：2 × RTX 4090，MODEL 环境变量指向模型目录（或使用默认路径）。
无 GPU 环境时自动跳过（pytest.mark.skipif）。
"""

import os

import pytest
import torch

MODEL_PATH = os.environ.get(
    "MODEL",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
    ),
)
MAX_NEW_TOKENS = 5
PROMPT = "The capital of France is"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="需要至少 1 个 GPU",
)
def test_paged_attention_generates_without_error() -> None:
    """
    最小 smoke test：paged attention 路径能完成 generate 而不报错。
    单 GPU（cuda:0）即可运行。
    """
    from mini_infer import EngineConfig, LLMEngine

    config = EngineConfig(
        model_name=MODEL_PATH,
        device="cuda:0",
        dtype="float16",
        num_gpu_blocks=200,
        block_size=256,
    )
    engine = LLMEngine(config)
    results = engine.generate([PROMPT], max_new_tokens=MAX_NEW_TOKENS)

    assert len(results) == 1
    assert len(results[0]) > 0, "输出不应为空"
    print(f"\n[paged] 输出：{results[0]!r}")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="需要 2 个 GPU 才能同时运行 mini-infer + HF baseline 对比",
)
def test_paged_attention_matches_hf_greedy() -> None:
    """
    正确性对比：paged attention 输出 token 序列应与 HF Transformers greedy 一致。

    mini-infer 在 cuda:0，HF baseline 在 cuda:1，避免显存冲突。
    greedy 采样（temperature=0）下 token 序列应完全一致。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mini_infer import EngineConfig, LLMEngine
    from mini_infer.core.request import SamplingParams

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # --- mini-infer paged 路径（cuda:0）---
    config = EngineConfig(
        model_name=MODEL_PATH,
        device="cuda:0",
        dtype="float16",
        num_gpu_blocks=200,
        block_size=256,
    )
    engine = LLMEngine(config)
    paged_results = engine.generate([PROMPT], max_new_tokens=MAX_NEW_TOKENS)
    paged_text = paged_results[0]

    # --- HF baseline（cuda:1）---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda:1",
    )
    hf_model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:1")
    with torch.no_grad():
        hf_output = hf_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy
            temperature=1.0,
            top_p=1.0,
            top_k=50,
        )

    # 只取新生成的 token（去掉 prompt 部分）
    prompt_len = inputs["input_ids"].shape[1]
    hf_new_tokens = hf_output[0, prompt_len:].tolist()
    hf_text = tokenizer.decode(hf_new_tokens, skip_special_tokens=True)

    print(f"\n[paged] 输出：{paged_text!r}")
    print(f"[hf]    输出：{hf_text!r}")

    # greedy 下 token 序列应一致（允许末尾空格差异）
    assert paged_text.strip() == hf_text.strip(), (
        f"Token 序列不一致！\n  paged: {paged_text!r}\n  hf:    {hf_text!r}"
    )

    # 清理显存
    del hf_model
    torch.cuda.empty_cache()
