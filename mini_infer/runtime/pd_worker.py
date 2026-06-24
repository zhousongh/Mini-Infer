"""
Phase 15：PD 解耦 Worker 进程。

PrefillWorker：
  - 接收 (prompt, sampling_params) 请求
  - 执行 prefill forward，提取 KV，发送 KVPayload 给 DecodeWorker
  - 返回 first_token_id（不做 decode loop）

DecodeWorker：
  - 接收 KVPayload，重建 KV 到 block tensor
  - 执行 decode loop 直到 EOS 或 max_new_tokens
  - 返回生成文本

两个 worker 均运行在独立进程中，通过 multiprocessing.Queue 通信。
"""

from __future__ import annotations

import time
from multiprocessing import Queue
from typing import Any

import torch
from transformers import DynamicCache

from ..cache.kv_transfer import KVPayload, KVSender, extract_kv_from_past
from ..core.config import EngineConfig
from ..core.request import SamplingParams

# ──────────────────────────────────────────────────────────────────────────────
# 消息类型
# ──────────────────────────────────────────────────────────────────────────────

class PrefillRequest:
    """Router → PrefillWorker 的请求消息。"""
    def __init__(self, request_id: str, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.0, top_p: float = 1.0) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p


class DecodeResult:
    """DecodeWorker → Router 的结果消息。"""
    def __init__(self, request_id: str, output_text: str,
                 transfer_time: float = 0.0, decode_time: float = 0.0,
                 prefill_time: float = 0.0) -> None:
        self.request_id = request_id
        self.output_text = output_text
        self.transfer_time = transfer_time
        self.decode_time = decode_time
        self.prefill_time = prefill_time  # 来自 KVPayload.prefill_time（PrefillWorker 内计时）


# ──────────────────────────────────────────────────────────────────────────────
# PrefillWorker
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(config: EngineConfig):
    """在 worker 进程内直接加载模型（不经过 ModelRunner，避免 paged decode patch）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=config.device,
    )
    model.eval()
    return model, tokenizer, eos_token_id


def run_prefill_worker(
    config: EngineConfig,
    req_queue: Queue,      # PrefillRequest 入队
    kv_queue: Queue,       # KVPayload 出队（→ DecodeWorker）
    stop_event: Any,       # multiprocessing.Event
) -> None:
    """
    PrefillWorker 主循环（在独立进程中运行）。

    循环从 req_queue 取请求 → prefill → 提取 KV → 发送 KVPayload。
    """
    from ..modeling.model_runner import _sample_token

    if not config.dry_run:
        try:
            model, tokenizer, eos_token_id = _load_model_and_tokenizer(config)
        except Exception as e:
            import traceback
            print(f"[PrefillWorker] 模型加载失败: {e}", flush=True)
            traceback.print_exc()
            return

    sender = KVSender(kv_queue)

    def tokenize(prompt: str) -> list[int]:
        if config.dry_run:
            return [ord(c) % 1000 for c in prompt[:32]]
        return tokenizer.encode(prompt, add_special_tokens=True)

    while not stop_event.is_set():
        try:
            req: PrefillRequest = req_queue.get(timeout=1.0)
        except Exception:
            continue

        try:
            t0 = time.perf_counter()
            prompt_token_ids = tokenize(req.prompt)
            prompt_len = len(prompt_token_ids)

            if config.dry_run:
                # dry_run：不做真实 forward，直接构造空 KVPayload
                payload = KVPayload(
                    request_id=req.request_id,
                    kv_layers=[],
                    first_token_id=1,
                    prompt_len=prompt_len,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    prefill_time=time.perf_counter() - t0,
                )
            else:
                input_ids = torch.tensor(
                    [prompt_token_ids], dtype=torch.long, device=config.device
                )
                with torch.no_grad():
                    out = model(
                        input_ids=input_ids,
                        past_key_values=DynamicCache(),
                        use_cache=True,
                    )

                params = SamplingParams(
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                )
                first_token_id = _sample_token(out.logits[0, -1], params)
                kv_layers = extract_kv_from_past(out.past_key_values, prompt_len)

                payload = KVPayload(
                    request_id=req.request_id,
                    kv_layers=kv_layers,
                    first_token_id=first_token_id,
                    prompt_len=prompt_len,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    prefill_time=time.perf_counter() - t0,
                )

            sender.send(payload)
        except Exception as e:
            import traceback
            print(f"[PrefillWorker] 请求 {req.request_id!r} 处理失败: {e}", flush=True)
            traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# DecodeWorker
# ──────────────────────────────────────────────────────────────────────────────

def run_decode_worker(
    config: EngineConfig,
    kv_queue: Queue,        # KVPayload 入队（← PrefillWorker）
    result_queue: Queue,    # DecodeResult 出队（→ PDEngine）
    stop_event: Any,
) -> None:
    """
    DecodeWorker 主循环（在独立进程中运行）。

    循环从 kv_queue 取 KVPayload → 重建 DynamicCache → decode loop → 发送结果。
    """
    from ..modeling.model_runner import _sample_token

    if not config.dry_run:
        try:
            model, tokenizer, eos_token_id = _load_model_and_tokenizer(config)
        except Exception as e:
            import traceback
            print(f"[DecodeWorker] 模型加载失败: {e}", flush=True)
            traceback.print_exc()
            return
    else:
        eos_token_id = -1

    def decode_token_to_text(token_id: int) -> str:
        if config.dry_run:
            return f" [{token_id}]"
        return tokenizer.decode([token_id], skip_special_tokens=True)

    while not stop_event.is_set():
        try:
            payload: KVPayload = kv_queue.get(timeout=1.0)
        except Exception:
            continue

        try:
            t_recv = time.perf_counter()

            params = SamplingParams(
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
            )

            generated_ids = [payload.first_token_id]
            generated_texts = [decode_token_to_text(payload.first_token_id)]

            if config.dry_run:
                # dry_run：生成固定 token 序列直到 max_new_tokens
                for _ in range(params.max_new_tokens - 1):
                    tok = 2
                    generated_ids.append(tok)
                    generated_texts.append(f" [{tok}]")
                    if tok == eos_token_id:
                        break
            else:
                # 重建 DynamicCache（从 KVPayload 的 kv_layers）
                past_kv = _rebuild_dynamic_cache(payload.kv_layers, config.device)

                cur_token_id = payload.first_token_id
                for _ in range(params.max_new_tokens - 1):
                    if cur_token_id == eos_token_id:
                        break
                    input_ids = torch.tensor(
                        [[cur_token_id]], dtype=torch.long, device=config.device
                    )
                    with torch.no_grad():
                        out = model(
                            input_ids=input_ids,
                            past_key_values=past_kv,
                            use_cache=True,
                        )
                    past_kv = out.past_key_values
                    cur_token_id = _sample_token(out.logits[0, -1], params)
                    generated_ids.append(cur_token_id)
                    generated_texts.append(decode_token_to_text(cur_token_id))
                    if cur_token_id == eos_token_id:
                        break

            decode_time = time.perf_counter() - t_recv
            output_text = "".join(generated_texts)

            result_queue.put(DecodeResult(
                request_id=payload.request_id,
                output_text=output_text,
                transfer_time=0.0,  # 由 PDEngine 层计算
                decode_time=decode_time,
                prefill_time=payload.prefill_time,  # 从 PrefillWorker 透传
            ))
        except Exception as e:
            import traceback
            print(f"[DecodeWorker] 请求 {payload.request_id!r} 处理失败: {e}", flush=True)
            traceback.print_exc()


def _rebuild_dynamic_cache(
    kv_layers: list[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> DynamicCache:
    """
    从 KVPayload.kv_layers 重建 HF DynamicCache。

    kv_layers[i] = (k, v)，shape [seq_len, num_kv_heads, head_dim]
    DynamicCache 期望 shape [batch, num_kv_heads, seq_len, head_dim]
    """
    cache = DynamicCache()
    for i, (k, v) in enumerate(kv_layers):
        # [seq, heads, dim] → [1, heads, seq, dim]
        k_gpu = k.to(device).permute(1, 0, 2).unsqueeze(0)
        v_gpu = v.to(device).permute(1, 0, 2).unsqueeze(0)
        cache.update(k_gpu, v_gpu, i)
    return cache
