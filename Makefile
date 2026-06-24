# 默认使用系统 python；在 conda 环境中覆盖：
#   PYTHON="conda run -n ai-infra python" make test
#
# 模型路径默认值指向标准 HuggingFace 缓存；可通过环境变量覆盖：
#   MODEL_1_5B=/your/path make bench-quant
#   MODEL_7B=/your/path   make bench
PYTHON     ?= python
PYTEST     ?= python -m pytest
MODEL_1_5B ?= $(HOME)/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
MODEL_7B   ?= $(HOME)/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct

# ── 测试 ──────────────────────────────────────────────────────────────
.PHONY: test test-fast test-gpu

## 全量测试（需要 GPU，约 50s）
test:
	HF_HUB_OFFLINE=1 $(PYTEST) tests/ -q

## 快速测试：不需要 GPU，约 10s
test-fast:
	$(PYTEST) tests/ -q \
		--ignore=tests/test_paged_attention.py \
		--ignore=tests/test_triton_attn.py \
		--ignore=tests/test_flash_decode.py \
		--ignore=tests/test_moe.py \
		--ignore=tests/test_mla_attention.py

## 仅 GPU 专项测试
test-gpu:
	HF_HUB_OFFLINE=1 $(PYTEST) \
		tests/test_paged_attention.py \
		tests/test_triton_attn.py \
		tests/test_flash_decode.py \
		-v

# ── Demo ─────────────────────────────────────────────────────────────
.PHONY: demo demo-quant demo-cuda-graph demo-prefix-cache

## 全部对比演示（quant + cuda-graph + prefix-cache）
demo:
	HF_HUB_OFFLINE=1 $(PYTHON) demo.py --model $(MODEL_1_5B) --mode all

## FP16 vs W8A8 量化对比
demo-quant:
	HF_HUB_OFFLINE=1 $(PYTHON) demo.py --model $(MODEL_1_5B) --mode quant

## Eager vs CUDA Graph 延迟对比
demo-cuda-graph:
	HF_HUB_OFFLINE=1 $(PYTHON) demo.py --model $(MODEL_1_5B) --mode cuda-graph

## Prefix Cache 冷启动 vs 命中 TTFT 对比
demo-prefix-cache:
	HF_HUB_OFFLINE=1 $(PYTHON) demo.py --model $(MODEL_1_5B) --mode prefix-cache

# ── 服务与聊天 ────────────────────────────────────────────────────────
.PHONY: serve serve-real chat

## 启动 dry-run HTTP 服务（无需模型权重）
serve:
	$(PYTHON) serve.py --dry-run --port 8000

## 启动真实模型 HTTP 服务（需要 Qwen2.5-7B）
serve-real:
	HF_HUB_OFFLINE=1 $(PYTHON) serve.py --model $(MODEL_7B) --port 8000

## 聊天（dry-run 模式）
chat:
	$(PYTHON) quick_chat.py

# ── Benchmark ────────────────────────────────────────────────────────
.PHONY: bench bench-moe bench-quant

## 主线 benchmark：mini-infer vs HF baseline（需要 Qwen2.5-7B）
bench:
	HF_HUB_OFFLINE=1 $(PYTHON) benchmarks/benchmark_flash.py \
		--model $(MODEL_7B) --batch-size 8 --compare

## MoE EP benchmark（需要 2 GPU，无需模型权重）
bench-moe:
	$(PYTHON) benchmarks/benchmark_moe.py \
		--compare --batch-size 4 --seq-len 16 \
		--hidden-size 512 --intermediate-size 1024 \
		--num-experts 8 --top-k 2 --warmup 2 --runs 5 --src-rank 1

## 量化 benchmark：FP16 vs W8A8（需要 Qwen2.5-1.5B）
bench-quant:
	HF_HUB_OFFLINE=1 $(PYTHON) benchmarks/benchmark_quant.py \
		--model $(MODEL_1_5B) --compare --batch-size 4 --max-new-tokens 50

# ── 帮助 ─────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "mini-infer Makefile 命令："
	@echo ""
	@echo "  测试"
	@echo "    make test              全量测试（需要 GPU，约 50s）"
	@echo "    make test-fast         快速测试（不需要 GPU，约 10s）"
	@echo "    make test-gpu          仅 GPU 专项测试"
	@echo ""
	@echo "  对比演示（需要 Qwen2.5-1.5B，约 3 GB VRAM）"
	@echo "    make demo              全部三种对比"
	@echo "    make demo-quant        FP16 vs W8A8 量化"
	@echo "    make demo-cuda-graph   Eager vs CUDA Graph"
	@echo "    make demo-prefix-cache 冷启动 vs 前缀命中"
	@echo ""
	@echo "  服务"
	@echo "    make serve             dry-run HTTP 服务（无需权重）"
	@echo "    make serve-real        真实模型 HTTP 服务（需要 7B）"
	@echo "    make chat              dry-run 聊天"
	@echo ""
	@echo "  Benchmark（需要对应模型权重）"
	@echo "    make bench             主线 vs HF baseline（需要 7B）"
	@echo "    make bench-moe         MoE EP benchmark（需要 2 GPU）"
	@echo "    make bench-quant       量化 benchmark（需要 1.5B）"
	@echo ""

.DEFAULT_GOAL := help
