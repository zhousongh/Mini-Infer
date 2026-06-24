"""benchmarks/benchmark_mla.py

Phase 14：MLA（Multi-head Latent Attention）KV Cache 对比 benchmark。

分两个 section：

  Section 1（无需模型权重，纯 CPU）
    打印 GQA vs MLA-naive vs MLA-latent 的理论 KV cache 大小对比表。
    验证 DeepSeek-V2-Lite 的压缩比（~56.25% vs GQA）。

  Section 2（需要模型权重，GPU）
    加载 DeepSeek-V2-Lite，greedy generate 若干 token，
    记录 torch.cuda.max_memory_allocated 峰值，
    对比两种 cache 策略的实际 GPU 显存占用。

用法：
  # Section 1 only（无需模型）
  conda run -n ai-infra python benchmarks/benchmark_mla.py --section 1

  # Section 2（需要模型权重）
  conda run -n ai-infra python benchmarks/benchmark_mla.py --section 2 \\
    --model-path ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0

  # 全部
  conda run -n ai-infra python benchmarks/benchmark_mla.py --section all \\
    --model-path <path>
"""

import argparse
import os
import sys

# Section 1 不需要 GPU，避免在没有权重时意外触发 CUDA 初始化
_MODEL_PATH_DEFAULT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite"
    "/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0"
)


# ─────────────────────────────────────────────────────────────
# Section 1：理论 KV cache 大小对比（无需模型）
# ─────────────────────────────────────────────────────────────


def section1_theory():
    """打印 GQA vs MLA naive vs MLA latent 的理论 KV cache 大小对比。"""
    from mini_infer.modeling.mla_attention import compute_kv_cache_bytes

    print("=" * 60)
    print("Section 1：理论 KV Cache 大小对比")
    print("=" * 60)

    # ── 配置说明 ──
    print()
    print("对比基准（fp16，seq_len=1024，num_layers=27）：")
    print("  GQA       ：Qwen2.5-7B 风格（4 KV heads, head_dim=128）")
    print("  MLA naive ：DeepSeek-V2-Lite（16 heads, q_head_dim=192, v_head_dim=128）")
    print("  MLA latent：DeepSeek-V2-Lite 压缩缓存（kv_lora_rank=512, rope_dim=64）")

    seq_len = 1024
    num_layers = 27  # DeepSeek-V2-Lite 层数

    result = compute_kv_cache_bytes(
        seq_len=seq_len,
        num_layers=num_layers,
        dtype_bytes=2,
        # GQA baseline（Qwen2.5-7B 风格）
        gqa_num_kv_heads=4,
        gqa_head_dim=128,
        # MLA（DeepSeek-V2-Lite）
        mla_num_heads=16,
        mla_q_head_dim=192,
        mla_v_head_dim=128,
        mla_kv_lora_rank=512,
        mla_qk_rope_head_dim=64,
    )

    # ── 每 token 每层 ──
    print()
    print("─" * 60)
    print(f"{'策略':<20} {'bytes/token/layer':>20} {'相对 GQA':>12}")
    print("─" * 60)

    gqa_b = result["gqa_bytes_per_token_layer"]
    naive_b = result["mla_naive_bytes_per_token_layer"]
    latent_b = result["mla_latent_bytes_per_token_layer"]

    print(f"{'GQA (4KV,128dim)':<20} {gqa_b:>20,}  {'100.00%':>10}")
    print(f"{'MLA naive':<20} {naive_b:>20,}  {naive_b/gqa_b*100:>9.2f}%")
    print(f"{'MLA latent':<20} {latent_b:>20,}  {latent_b/gqa_b*100:>9.2f}%")
    print("─" * 60)

    # ── 全局（seq=1024, 27 layers）──
    print()
    print(f"全局 KV cache（seq_len={seq_len}, {num_layers} layers）：")
    print(f"  GQA        = {result['gqa_total_gb']*1024:.1f} MB")
    print(f"  MLA naive  = {result['mla_naive_total_gb']*1024:.1f} MB")
    print(f"  MLA latent = {result['mla_latent_total_gb']*1024:.1f} MB")

    ratio = result["mla_latent_vs_gqa_ratio"]
    print()
    print(f"MLA latent 压缩比（vs GQA）  = {ratio*100:.2f}%")
    print(f"MLA latent vs MLA naive      = {result['mla_latent_vs_mla_naive_ratio']*100:.2f}%")

    # ── 验收断言 ──
    assert ratio < 0.60, f"压缩比 {ratio:.4f} 超出预期（应 < 0.60）"
    assert latent_b == (512 + 64) * 2, f"latent bytes/token/layer = {latent_b}，期望 1152"
    print()
    print("✅ 验收检查通过：压缩比 < 60%，latent = 1152 bytes/token/layer")

    # ── 扩展性说明 ──
    print()
    print("─" * 60)
    print("扩展性对比（相同 VRAM 下的并发上限估算，假设 32 GB 可用显存，仅供参考）：")
    vram = 32 * 1024 * 1024 * 1024  # 32 GB（假设值，非实测）
    for name, b in [("GQA", gqa_b), ("MLA latent", latent_b)]:
        max_tokens = vram // (b * num_layers)
        print(f"  {name:<12}：最多 {max_tokens:,} token（{max_tokens//seq_len:,} × seq={seq_len}）")
    print("─" * 60)
    print()


# ─────────────────────────────────────────────────────────────
# Section 2：真实模型 GPU 显存测量（需要模型权重）
# ─────────────────────────────────────────────────────────────


def section2_gpu(model_path: str):
    """加载 DeepSeek-V2-Lite，测量 greedy generate 时的 GPU 峰值显存。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("Section 2：真实模型 GPU 显存测量")
    print("=" * 60)

    if not os.path.isdir(model_path):
        print(f"[ERROR] 模型路径不存在：{model_path}")
        print("       请确认 DeepSeek-V2-Lite 已下载完成，再运行 --section 2")
        sys.exit(1)

    # 检查是否有 .incomplete shard
    blob_dir = os.path.join(
        os.path.dirname(os.path.dirname(model_path)), "blobs"
    )
    if os.path.isdir(blob_dir):
        incomplete = [f for f in os.listdir(blob_dir) if f.endswith(".incomplete")]
        if incomplete:
            print(f"[ERROR] 发现 {len(incomplete)} 个未完成 shard：")
            for f in incomplete:
                print(f"  {f}")
            print("       请等待下载完成后再运行 --section 2")
            sys.exit(1)

    print(f"模型路径：{model_path}")
    print("加载中（trust_remote_code=True）…")

    os.environ["HF_HUB_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"模型加载峰值显存：{model_mem_gb:.2f} GB")

    prompt = "介绍一下 DeepSeek-V2 的 MLA 注意力机制："
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    max_new_tokens = 64
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print()
    print(f"生成 {max_new_tokens} tokens 峰值显存：{gen_mem_gb:.2f} GB")
    print(f"增量（KV cache + activations）：{gen_mem_gb - model_mem_gb:.3f} GB")
    print()
    print("生成内容（前 200 字符）：")
    print(generated[:200])
    print()



# ─────────────────────────────────────────────────────────────
# Section 3：三种实现单步 decode 延迟对比（需要模型权重，GPU）
# ─────────────────────────────────────────────────────────────


def section3_latency(model_path: str):
    """
    用真实 DeepSeek-V2-Lite 第 0 层权重，对比三种实现的单步 decode 延迟：
      - MLAAttentionNaive：每步展开完整 K/V
      - MLAAttentionLatentCache：每步对全部历史 compressed_kv 做 kv_b_proj 展开
      - MLAAttentionAbsorbed：矩阵吸收，直接用 compressed_kv 计算 score/output

    测试条件：batch=1，seq_len=[1, 64, 256, 1024]（模拟不同长度的 decode 步骤）
    """
    import time

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from mini_infer.modeling.mla_attention import (
        MLAAttentionAbsorbed,
        MLAAttentionLatentCache,
        MLAAttentionNaive,
        MLAConfig,
        MLAKVCacheLatent,
        MLAKVCacheNaive,
    )

    print("=" * 60)
    print("Section 3：三种实现单步 decode 延迟对比")
    print("=" * 60)

    if not os.path.isdir(model_path):
        print(f"[ERROR] 模型路径不存在：{model_path}")
        sys.exit(1)

    os.environ["HF_HUB_OFFLINE"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{device}")

    # 加载第 0 层权重
    hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg = MLAConfig(
        hidden_size=hf_cfg.hidden_size,
        num_heads=hf_cfg.num_attention_heads,
        q_lora_rank=hf_cfg.q_lora_rank,
        qk_nope_head_dim=hf_cfg.qk_nope_head_dim,
        qk_rope_head_dim=hf_cfg.qk_rope_head_dim,
        kv_lora_rank=hf_cfg.kv_lora_rank,
        v_head_dim=hf_cfg.v_head_dim,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True,
    )
    hf_attn = hf_model.model.layers[0].self_attn

    def make_modules():
        naive = MLAAttentionNaive(cfg).to(torch.float16).to(device).eval()
        latent = MLAAttentionLatentCache(cfg).to(torch.float16).to(device).eval()
        absorbed = MLAAttentionAbsorbed(cfg).to(torch.float16).to(device).eval()
        # 复制权重
        for m in [naive, latent, absorbed]:
            with torch.no_grad():
                m.kv_a_proj_with_mqa.weight.copy_(hf_attn.kv_a_proj_with_mqa.weight)
                m.kv_a_layernorm.weight.copy_(hf_attn.kv_a_layernorm.weight)
                m.kv_b_proj.weight.copy_(hf_attn.kv_b_proj.weight)
                m.o_proj.weight.copy_(hf_attn.o_proj.weight)
                if cfg.q_lora_rank is None:
                    m.q_proj.weight.copy_(hf_attn.q_proj.weight)
        return naive, latent, absorbed

    WARMUP, REPEAT = 10, 50
    seq_lens = [1, 64, 256, 1024]

    print()
    print(f"{'seq_len':>8}  {'naive(ms)':>12}  {'latent(ms)':>12}  {'absorbed(ms)':>14}  {'absorbed/naive':>14}")
    print("─" * 70)

    for seq_len in seq_lens:
        naive, latent, absorbed = make_modules()

        # 构造 prefill cache（模拟已有 seq_len 个历史 token）
        bsz = 1
        x_decode = torch.randn(bsz, 1, cfg.hidden_size, dtype=torch.float16, device=device)

        # naive cache
        cache_naive = MLAKVCacheNaive(
            key_states=torch.randn(bsz, cfg.num_heads, seq_len, cfg.q_head_dim, dtype=torch.float16, device=device),
            value_states=torch.randn(bsz, cfg.num_heads, seq_len, cfg.v_head_dim, dtype=torch.float16, device=device),
        )
        # latent/absorbed cache
        cache_latent = MLAKVCacheLatent(
            compressed_kv=torch.randn(bsz, seq_len, cfg.kv_lora_rank, dtype=torch.float16, device=device),
            k_pe=torch.randn(bsz, seq_len, cfg.qk_rope_head_dim, dtype=torch.float16, device=device),
        )

        def bench(fn, cache):
            # warmup
            for _ in range(WARMUP):
                with torch.no_grad():
                    fn(x_decode, past_cache=cache)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPEAT):
                with torch.no_grad():
                    fn(x_decode, past_cache=cache)
            if device == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / REPEAT * 1000  # ms

        t_naive = bench(naive, cache_naive)
        t_latent = bench(latent, cache_latent)
        t_absorbed = bench(absorbed, cache_latent)

        ratio = t_absorbed / t_naive
        print(f"{seq_len:>8}  {t_naive:>12.3f}  {t_latent:>12.3f}  {t_absorbed:>14.3f}  {ratio:>13.2f}x")

    # 循环结束后释放 hf 模型权重
    del hf_attn, hf_model

    print("─" * 70)
    print()
    print("说明：")
    print("  naive    ：缓存完整展开的 key_states/value_states，每步只展开新 token，cache 大（10,240 bytes/token/layer）")
    print("  latent   ：每步对全部历史 compressed_kv 做 kv_b_proj 展开，cache 小（1,152 bytes/token/layer）")
    print("  absorbed ：矩阵吸收，直接用 compressed_kv_normed 计算 score/output，cache 同 latent，避免显式 k_nope 展开")
    print()


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 14 MLA KV Cache benchmark")
    parser.add_argument(
        "--section",
        choices=["1", "2", "3", "all"],
        default="1",
        help="运行哪个 section（默认 1，不需要模型权重）",
    )
    parser.add_argument(
        "--model-path",
        default=_MODEL_PATH_DEFAULT,
        help="DeepSeek-V2-Lite 模型路径（Section 2/3 需要）",
    )
    args = parser.parse_args()

    if args.section in ("1", "all"):
        section1_theory()

    if args.section in ("2", "all"):
        section2_gpu(args.model_path)

    if args.section in ("3", "all"):
        section3_latency(args.model_path)


if __name__ == "__main__":
    main()
