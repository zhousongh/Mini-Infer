"""
gen_charts.py — 从 benchmarks/results/*.json 读取实验数据并生成对比图表。
输出目录：assets/charts/
用法：python scripts/gen_charts.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(__file__)
_ROOT        = os.path.join(_SCRIPT_DIR, "..")
OUT_DIR      = os.path.join(_ROOT, "assets", "charts")
RESULTS_DIR  = os.path.join(_ROOT, "benchmarks", "results")
os.makedirs(OUT_DIR, exist_ok=True)


def _load(filename: str) -> dict:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 中文字体 ──────────────────────────────────────────────────────────────────
_CJK_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
_cjk_font = None
for _fp in _CJK_FONT_PATHS:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        _prop = fm.FontProperties(fname=_fp)
        _cjk_font = _prop.get_name()
        break

# ── 全局样式 ──────────────────────────────────────────────────────────────────
if _cjk_font:
    plt.rcParams["font.family"] = [_cjk_font, "DejaVu Sans", "sans-serif"]

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
})

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B2"
GRAY   = "#8C8C8C"


# ── 图 1：主线吞吐演进 ────────────────────────────────────────────────────────
def chart_throughput_evolution():
    d = _load("throughput_evolution.json")
    rows   = d["data"]
    phases = [r["label"].split(" (")[0] + "\n(" + r["label"].split("(")[1] if "(" in r["label"] else r["label"]
              for r in rows]
    tps    = [r["throughput_tok_s"] for r in rows]
    pct    = [f'{r["vs_hf_pct"]:.1f}%' if r["vs_hf_pct"] < 100 else "—" for r in rows]
    colors = [ORANGE, BLUE, GRAY]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(phases, tps, color=colors, width=0.45, zorder=3)

    for bar, p, t in zip(bars, pct, tps):
        ax.text(bar.get_x() + bar.get_width() / 2, t + 4,
                f"{t:.0f} tok/s\n({p})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    hf_tps = rows[-1]["throughput_tok_s"]
    ax.set_ylim(0, 480)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"主线吞吐演进 — {d['meta']['model']}, batch={d['meta']['batch_size']}")
    ax.axhline(hf_tps, color=GRAY, linestyle="--", linewidth=1.2, zorder=2, label="HF baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_throughput_evolution.png"))
    plt.close(fig)
    print("✓ 01_throughput_evolution.png")


# ── 图 2：CUDA Graph decode 延迟 ──────────────────────────────────────────────
def chart_cuda_graph():
    d      = _load("cuda_graph.json")
    rows   = d["data"]
    bs         = [r["batch_size"] for r in rows]
    eager_ms   = [r["eager_ms"]   for r in rows]
    graph_ms   = [r["graph_ms"]   for r in rows]
    speedups   = [r["speedup"]    for r in rows]

    x = np.arange(len(bs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))

    b1 = ax.bar(x - w/2, eager_ms, w, color=ORANGE, label="Eager", zorder=3)
    b2 = ax.bar(x + w/2, graph_ms, w, color=BLUE,   label="CUDA Graph", zorder=3)

    for bar, sp in zip(b2, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{sp:.2f}×",
                ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"bs={b}" for b in bs])
    ax.set_ylabel("Decode step latency (ms)")
    ax.set_title(f"CUDA Graph vs Eager — {d['meta']['model']} decode latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_cuda_graph.png"))
    plt.close(fig)
    print("✓ 02_cuda_graph.png")


# ── 图 3：MoE EP 吞吐演进 ─────────────────────────────────────────────────────
def chart_moe_ep():
    d      = _load("moe_ep_evolution.json")
    rows   = d["data"]
    labels = [r["label"] for r in rows]
    tps    = [r["throughput_tok_s"] for r in rows]
    ratios = [f'{r["vs_dense"]:.3f}×' for r in rows]
    colors = [GRAY, ORANGE, GREEN, BLUE]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, tps, color=colors, width=0.5, zorder=3)

    for bar, r, t in zip(bars, ratios, tps):
        ax.text(bar.get_x() + bar.get_width() / 2, t + 400,
                f"{t/1000:.1f}k\n({r})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 65000)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"MoE Expert Parallelism 吞吐演进 — {d['meta']['hardware']}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_moe_ep_evolution.png"))
    plt.close(fig)
    print("✓ 03_moe_ep_evolution.png")


# ── 图 4：Flash Decoding 延迟 vs seq_len ──────────────────────────────────────
def chart_flash_decode():
    d    = _load("flash_decode.json")
    rows = d["data"]
    seq_lens      = [r["seq_len"]        for r in rows]
    flash_attn_ms = [r["flash_attn_ms"]  for r in rows]
    triton_65_ms  = [r["triton_65_ms"]   for r in rows]
    flash_dec_ms  = [r["flash_decode_ms"] for r in rows]
    crossover     = d["summary"]["crossover_seq_len"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(seq_lens, flash_attn_ms, "o-", color=GRAY,   label="flash_attn（基准）", linewidth=2)
    ax.plot(seq_lens, triton_65_ms,  "s-", color=ORANGE, label="Triton baseline", linewidth=2)
    ax.plot(seq_lens, flash_dec_ms,  "^-", color=BLUE,   label="Flash Decoding（split-K）", linewidth=2)

    ax.axvline(crossover, color=GREEN, linestyle=":", linewidth=1.2, label=f"crossover ≈ {crossover}")
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Attention latency (ms)")
    ax.set_title(f"Flash Decoding (Split-K) vs Triton — {d['meta']['model']}, bs=1")
    ax.legend(loc="upper left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels([str(s) for s in seq_lens])
    speedup = d["summary"]["speedup_vs_triton_at_4096"]
    ax.annotate(f"{speedup:.2f}× faster\nvs Triton @ seq=4096",
                xy=(4096, flash_dec_ms[-1]), xytext=(2200, 0.18),
                arrowprops=dict(arrowstyle="->", color=BLUE),
                fontsize=9, color=BLUE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_flash_decode.png"))
    plt.close(fig)
    print("✓ 04_flash_decode.png")


# ── 图 5：Chunked Prefill ITL spike 对比 ──────────────────────────────────────
def chart_chunked_prefill():
    d          = _load("chunked_prefill.json")
    rows       = d["data"]
    categories = [r["label"]              for r in rows]
    itl_spike  = [r["max_itl_spike_ms"]   for r in rows]
    throughput = [r["throughput_tok_s"]   for r in rows]

    x  = np.arange(len(categories))
    w  = 0.35
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - w/2, itl_spike,  w, color=RED,  label="Max ITL spike (ms)", zorder=3)
    b2 = ax2.bar(x + w/2, throughput, w, color=BLUE, label="Throughput (tok/s)", zorder=3)

    for bar, v in zip(b1, itl_spike):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 2,
                 f"{v:.0f} ms", ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold")
    for bar, v in zip(b2, throughput):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 2,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel("Max ITL spike (ms)", color=RED)
    ax2.set_ylabel("Throughput (tok/s)", color=BLUE)
    ax1.set_ylim(0, 200)
    ax2.set_ylim(270, 320)
    ax1.set_title(f"Chunked Prefill — ITL spike vs throughput — {d['meta']['model']}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    reduction = d["summary"]["itl_spike_reduction_chunk128_pct"]
    ax1.annotate(f"−{reduction:.1f}%", xy=(1 - w/2, itl_spike[1]), xytext=(0.5, 120),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=10, color=RED, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_chunked_prefill.png"))
    plt.close(fig)
    print("✓ 05_chunked_prefill.png")


# ── 图 6：W8A8 量化显存与吞吐 ─────────────────────────────────────────────────
def chart_w8a8():
    d = _load("w8a8_quant.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    # 左图：显存对比
    models  = ["FP16", "W8A8"]
    mem_mb  = [d["memory"]["fp16_weight_mb"], d["memory"]["w8a8_weight_mb"]]
    bars = ax1.bar(models, mem_mb, color=[ORANGE, BLUE], width=0.4, zorder=3)
    for bar, v in zip(bars, mem_mb):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 30,
                 f"{v:.0f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 4200)
    ax1.set_ylabel("Weight memory (MB)")
    ax1.set_title("显存：权重占用")
    reduction = d["memory"]["reduction_pct"]
    ax1.annotate(f"−{reduction:.1f}%", xy=(1, mem_mb[1]), xytext=(0.6, 3000),
                 arrowprops=dict(arrowstyle="->", color=BLUE),
                 fontsize=11, color=BLUE, fontweight="bold")

    # 右图：decode tps 对比
    models2 = ["FP16", "W8A8\n(mixed fallback)"]
    tps2    = [d["throughput"]["fp16_decode_tok_s"], d["throughput"]["w8a8_decode_tok_s"]]
    bars2 = ax2.bar(models2, tps2, color=[ORANGE, RED], width=0.4, zorder=3)
    for bar, v in zip(bars2, tps2):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 8,
                 f"{v:.0f} tok/s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 650)
    ax2.set_ylabel("Decode throughput (tok/s)")
    ax2.set_title("推理吞吐（小 M 限制）")
    ax2.text(0.5, 0.12, "decode 退回 FP16 mixed fallback\n（torch._int_mm 小 M 限制）",
             transform=ax2.transAxes, ha="center", fontsize=8.5,
             color="gray", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", alpha=0.8))

    fig.suptitle(f"W8A8 量化（per-channel int8）— {d['meta']['model']}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "06_w8a8_quant.png"))
    plt.close(fig)
    print("✓ 06_w8a8_quant.png")


if __name__ == "__main__":
    chart_throughput_evolution()
    chart_cuda_graph()
    chart_moe_ep()
    chart_flash_decode()
    chart_chunked_prefill()
    chart_w8a8()
    print(f"\n所有图表已保存到 {os.path.abspath(OUT_DIR)}/")
