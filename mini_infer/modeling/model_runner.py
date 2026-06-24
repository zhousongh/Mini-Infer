"""
Phase 3/5/6/9/10/12/16 模型执行器。

Phase 16 新增（W8A8 量化）：
  - 加载模型后，若 config.quant_mode == "w8a8"，调用 quantize_model() 原地替换线性层。
  - 第一版 contract：activation per-row + weight per-channel。
  - embed_tokens / lm_head / attention qkv/o_proj / 小层保持 fp16，不参与量化。
  - 量化不影响 quant_mode="" 的默认 fp16 路径。

Phase 12 新增（CUDA Graph）：
  - warmup_cuda_graphs(batch_sizes)：引擎启动时调用，为每个 batch_size 预热 + 捕获 CUDA 图
  - decode_batch() 新增 graph replay 路径：
      - 找到最小 padded_bs >= actual_bs
      - copy_() 更新静态 buffer（input_ids / position_ids / block_table / cache_seqlens）
      - graph.replay() 回放捕获好的 CUDA kernel 序列，跳过 Python dispatch overhead
      - ensure_next_slot / advance_seq_lens / 采样 仍在 graph 外部执行（Python 操作）
  - max_kv_len 在 graph 模式下固定为 config.max_model_len，消除 .item() 同步，同时保证
    RoPE cos/sin 形状恒定（cos_cached[:max_model_len] 是常量形状，CUDA Graph 兼容）

Phase 10 新增：
  - prefill_with_prefix(state, cached_len, cached_blocks)：
      从 block tensor 重建前缀 KV（get_prefix_kv），
      仅对后缀 token 做 HF forward（past_key_values = prefix DynamicCache），
      再用 write_prefill_kv_suffix 把后缀 KV 写入新块，采样第一个 token。


Phase 9 新增（相比 Phase 6）：
  - prefill_chunk(state, token_start, token_end, past_cache, is_last_chunk)：
      对 prompt_token_ids[token_start:token_end] 做部分 prefill
      非最后 chunk：返回积累的 DynamicCache（供下一 chunk 使用）
      最后一个 chunk：写入 block tensor，采样第一个 token，设 prefilled=True，返回 None

Phase 6 变化（相比 Phase 5）：
  - decode_batch() 切换到 True PagedAttention 路径：
      - 删除 gather_batch_kv / DynamicCache / write_decode_kv
      - 改用 kv_cache.ensure_next_slot + build_block_tables + PagedDecodeContext
      - flash_attn_with_kvcache 直接从 block tensor 寻址，in-place 写入新 KV
  - __init__() 调用 patch_model_for_paged_decode() 永久 patch Qwen2 attention 层
  - prefill() 路径不变（仍用 DynamicCache，patch 在 prefill 时自动回退原始 HF forward）

Phase 5 变化（相比 Phase 3）：
  - decode_batch() 内三个关键段添加 torch.profiler.record_function 标签

Phase 3 变化（相比 Phase 2）：
  - decode_batch() 改用 DynamicCache 替代 tuple 格式的 past_key_values

Phase 2 已有的设计：
  - prefill() 写完 past_key_values 后，将 KV 写入 KVCacheManager 的 block tensor
  - decode_batch() batch 所有活跃请求做一次 GPU forward

dry_run=True 时保留桩实现，不加载真实模型，供无 GPU 的单元测试使用。
"""

import math

import torch
from transformers import DynamicCache

from ..cache.kv_cache import KVCacheManager
from ..core.config import EngineConfig
from ..core.request import RequestState, SamplingParams


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    """把配置里的 dtype 字符串转换成 torch dtype。"""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def _sample_token(logits: torch.Tensor, params: SamplingParams) -> int:
    """从最后一个位置的 logits 采样下一个 token，支持 greedy（temperature=0）和 top-p。"""
    if params.temperature == 0.0:
        return int(logits.argmax(dim=-1).item())

    logits = logits / params.temperature
    probs = torch.softmax(logits, dim=-1)

    if params.top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative - sorted_probs > params.top_p] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, num_samples=1).item())


class _StubTokenizer:
    """dry_run 模式下的占位 tokenizer，不依赖 transformers，仅供本地测试用。"""

    eos_token_id = -1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(f" [{tid}]" for tid in token_ids)


class ModelRunner:
    """模型执行器。dry_run=True 时使用桩实现（不加载模型），False 时加载真实模型和 tokenizer。"""

    def __init__(self, config: EngineConfig, kv_cache: KVCacheManager) -> None:
        self.config = config
        self.kv_cache = kv_cache

        if config.dry_run:
            self.tokenizer: _StubTokenizer | "AutoTokenizer" = _StubTokenizer()
            self.eos_token_id: int = -1
            self.model = None
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if config.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("配置要求 CUDA 设备，但当前环境未检测到可用 GPU。")

            tokenizer_name = config.tokenizer_name if config.tokenizer_name is not None else config.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 修复 Phase 1 bug：eos_token_id or 0 在 eos_token_id=None 时静默赋 0
            eos_id = self.tokenizer.eos_token_id
            self.eos_token_id = eos_id if eos_id is not None else -1

            dtype = _resolve_torch_dtype(config.dtype)
            # 使用 device_map 直接加载到目标 GPU，避免 CPU 中转导致的峰值显存 ×2
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=config.device,
            )
            self.model.eval()

            # Phase 16：W8A8 量化（quant_mode="w8a8" 时原地替换 MLP 线性层）
            if config.quant_mode == "w8a8":
                import torch.nn as _nn

                import mini_infer.modeling.quantization as _quant_mod
                _n_before = sum(1 for _, m in self.model.named_modules() if isinstance(m, _nn.Linear))
                _quant_mod.quantize_model(self.model)
                _n_after = sum(1 for _, m in self.model.named_modules() if isinstance(m, _quant_mod.QuantLinear))
                print(
                    f"[W8A8] quantize_model: {_n_after}/{_n_before} linear layers quantized "
                    f"(skipped {_n_before - _n_after} attention/lm_head/small layers)"
                )

            # Phase 6：永久 patch attention 层，decode 时走 paged attention 路径
            import mini_infer.kernels.attention as _attn_mod
            self._paged_ctx = _attn_mod.patch_model_for_paged_decode(self.model, self.kv_cache)

        # Phase 12：CUDA Graph pool（batch_size → CUDAGraph）和静态 buffer
        # 在 dry_run 模式下保持为空，不影响现有路径
        self._cuda_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_static: dict[int, dict] = {}  # bs → {input_ids, position_ids, block_table, cache_seqlens, logits}

    def prefill(self, states: list[RequestState]) -> None:
        """
        对每个请求独立跑 prefill forward，从 prefill logits 采样第一个 token，
        并将 KV 写入 KVCacheManager 的 block tensor。
        """
        if self.config.dry_run:
            for state in states:
                # 桩实现：生成 token [1] 作为第一个生成 token
                state.append_generated(1, " [1]")
                state.prefilled = True
                if len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                    state.mark_finished("length")
            return

        for state in states:
            input_ids = torch.tensor(
                [state.prompt_token_ids], dtype=torch.long, device=self.config.device
            )
            with torch.no_grad():
                # 传入空 DynamicCache，使模型返回 DynamicCache 而非 tuple，消除弃用警告
                out = self.model(input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True)

            # 将 prefill KV 写入 block tensor（替代 Phase 1 的 _past_kv dict）
            self.kv_cache.write_prefill_kv(state.request.request_id, out.past_key_values)

            next_token_id = _sample_token(out.logits[0, -1], state.request.sampling_params)
            state.append_generated(next_token_id, "")
            state.prefilled = True

            if next_token_id == self.eos_token_id:
                state.mark_finished("eos")
            elif len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                state.mark_finished("length")

    def prefill_with_prefix(
        self,
        state: RequestState,
        cached_len: int,
        cached_blocks: list[int],
    ) -> None:
        """
        Phase 10：后缀 prefill（有前缀命中时调用）。

        - 从 block tensor 重建前缀 KV（DynamicCache），只在 GPU 路径执行
        - 仅对 prompt_token_ids[cached_len:] 做 HF forward
        - 把后缀 KV 写入新分配的块（write_prefill_kv_suffix）
        - 采样并记录第一个生成 token，设 prefilled=True

        dry_run 路径与普通 prefill 桩实现相同。
        """
        if self.config.dry_run:
            state.append_generated(1, " [1]")
            state.prefilled = True
            if len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                state.mark_finished("length")
            return

        suffix_ids = state.prompt_token_ids[cached_len:]

        # 从 block tensor 重建前缀 KV（DynamicCache 格式）
        prefix_kv = self.kv_cache.get_prefix_kv(cached_blocks, cached_len)

        # 只对后缀 token 做 HF forward（past_key_values = prefix_kv）
        # DynamicCache.get_seq_length() 返回 cached_len，HF 会自动把 position_ids
        # 设为 [cached_len, cached_len+1, ..., prompt_len-1]，RoPE 位置编码正确
        suffix_input_ids = torch.tensor(
            [suffix_ids], dtype=torch.long, device=self.config.device
        )
        with torch.no_grad():
            out = self.model(
                input_ids=suffix_input_ids,
                past_key_values=prefix_kv,
                use_cache=True,
            )

        # 只写后缀 KV 到新块，跳过已缓存的前缀（避免覆盖共享 block）
        self.kv_cache.write_prefill_kv_suffix(
            state.request.request_id, out.past_key_values, cached_len
        )

        next_token_id = _sample_token(out.logits[0, -1], state.request.sampling_params)
        state.append_generated(next_token_id, "")
        state.prefilled = True

        if next_token_id == self.eos_token_id:
            state.mark_finished("eos")
        elif len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
            state.mark_finished("length")

    def prefill_chunk(
        self,
        state: RequestState,
        token_start: int,
        token_end: int,
        past_cache,
        is_last_chunk: bool,
    ):
        """
        Phase 9：对 prompt_token_ids[token_start:token_end] 做部分 prefill。

        past_cache: 上一 chunk 返回的 DynamicCache（第一个 chunk 传 None）。
        is_last_chunk=False：返回积累的 DynamicCache，不采样 token，不设 prefilled=True。
        is_last_chunk=True：写入 block tensor，采样第一个 token，设 prefilled=True，返回 None。

        调用方职责：
          - 非最后 chunk：保存返回值供下一次调用使用
          - 最后 chunk：丢弃返回值（None），并将请求移入 running
        """
        state.prefilled_tokens = token_end

        if self.config.dry_run:
            if is_last_chunk:
                state.append_generated(1, " [1]")
                state.prefilled = True
                if len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                    state.mark_finished("length")
            return None

        chunk_ids = state.prompt_token_ids[token_start:token_end]
        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=self.config.device)

        if past_cache is None:
            past_cache = DynamicCache()

        with torch.no_grad():
            out = self.model(input_ids=input_ids, past_key_values=past_cache, use_cache=True)

        if is_last_chunk:
            self.kv_cache.write_prefill_kv(state.request.request_id, out.past_key_values)
            next_token_id = _sample_token(out.logits[0, -1], state.request.sampling_params)
            state.append_generated(next_token_id, "")
            state.prefilled = True
            if next_token_id == self.eos_token_id:
                state.mark_finished("eos")
            elif len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                state.mark_finished("length")
            return None
        else:
            return out.past_key_values  # 积累的 DynamicCache，供下一 chunk 使用

    def decode_batch(self, states: list[RequestState]) -> None:
        """
        Batch decode（Phase 6 True PagedAttention 路径）：
        flash_attn_with_kvcache 直接从 block tensor 寻址，无 gather / DynamicCache / write_kv。

        算法：
          1. ensure_next_slot：确保下一写入位置已有物理块
          2. build_block_tables：构造 block_table / cache_seqlens 张量
          3. _paged_ctx.set：注入上下文，触发 patched attention 层走 paged 路径
          4. model.forward（无 past_key_values，无 attention_mask）：
             flash_attn_with_kvcache 在每层 in-place 写入新 KV，并完成 attention 计算
          5. advance_seq_lens：递增各请求的 seq_len（KV 已由 flash_attn 写入）
          6. 采样下一个 token，更新请求状态
        """
        active = [s for s in states if not s.finished]
        if not active:
            return

        if self.config.dry_run:
            active_ids = []
            for state in active:
                next_idx = len(state.generated_token_ids) + 1
                state.append_generated(next_idx, f" [{next_idx}]")
                if len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                    state.mark_finished("length")
                else:
                    active_ids.append(state.request.request_id)
            # 更新 dry_run 模式下的块管理元数据（不写 GPU 张量）
            if active_ids:
                self.kv_cache.write_decode_kv(active_ids, None, None)
            return

        request_ids = [s.request.request_id for s in active]
        bs = len(active)

        # Phase 6 True PagedAttention 路径（Phase 12 在此基础上可选 CUDA Graph 回放）：
        # 1. 确保每个请求的下一个写入位置已有物理块（必须在 graph 外执行，修改 Python 状态）
        self.kv_cache.ensure_next_slot(request_ids)

        # 2. 构造 flash_attn 所需张量
        block_table, cache_seqlens = self.kv_cache.build_block_tables(request_ids)

        # 3. input_ids：每个请求最后生成的 token，shape [batch, 1]
        last_tokens = [s.generated_token_ids[-1] for s in active]
        input_ids = torch.tensor(
            [[t] for t in last_tokens], dtype=torch.long, device=self.config.device
        )

        # 4. position_ids：新 token 的真实位置 = 当前 cache 长度
        position_ids = cache_seqlens.long().unsqueeze(1)

        # 5. Phase 12：尝试 CUDA Graph 路径；否则走 eager 路径
        padded_bs = self._find_padded_bs(bs)
        if padded_bs is not None:
            # Graph replay（max_kv_len 固定为 max_model_len，不需要 .item() 同步）
            logits_batch = self._graph_decode_forward(
                bs, padded_bs, input_ids, position_ids, block_table, cache_seqlens
            )
        else:
            # Eager 路径：max_kv_len 在此处做唯一一次 .item() 同步
            max_kv_len = int(cache_seqlens.max().item()) + 1
            logits_batch = self._eager_decode_forward(
                input_ids, position_ids, block_table, cache_seqlens, max_kv_len
            )

        # 6. flash_attn 已 in-place 写入新 KV，只需递增 seq_len（必须在 graph 外执行）
        self.kv_cache.advance_seq_lens(request_ids)

        # 7. 采样下一个 token，更新请求状态
        for b, state in enumerate(active):
            next_token_id = _sample_token(logits_batch[b], state.request.sampling_params)
            state.append_generated(next_token_id, "")

            if next_token_id == self.eos_token_id:
                state.mark_finished("eos")
            elif len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                state.mark_finished("length")

    def free_request(self, state: RequestState) -> None:
        """Phase 2 中 KV 由 KVCacheManager 管理，此处为接口兼容保留，无需操作。"""
        pass

    # ------------------------------------------------------------------
    # Phase 12：CUDA Graph 捕获与回放
    # ------------------------------------------------------------------

    def warmup_cuda_graphs(
        self, batch_sizes: list[int] | None = None, warmup_iters: int = 3
    ) -> None:
        """
        Phase 12：为指定 batch size 列表捕获 CUDA Graph，供 decode_batch 使用。

        调用时机：LLMEngine 初始化完成、模型加载后，首次请求到来前。
        dry_run=True 时直接返回，不做任何操作。

        捕获策略：
          - max_kv_len 固定为 config.max_model_len，使 RoPE cos/sin 形状恒定
          - block_table 静态宽度 = ceil(max_model_len / block_size)（每请求最大块数）
          - 捕获前做 warmup_iters 次 eager forward，让 CUDA 分配好所有 workspace
        """
        if self.config.dry_run:
            return
        if batch_sizes is None:
            batch_sizes = [bs for bs in [1, 2, 4, 8] if bs <= self.config.max_batch_size]

        max_blocks_per_seq = math.ceil(self.config.max_model_len / self.config.block_size)
        device = self.config.device
        fixed_max_kv_len = self.config.max_model_len

        for bs in sorted(batch_sizes):
            # 静态输入 buffer
            s_input_ids = torch.zeros(bs, 1, dtype=torch.long, device=device)
            s_position_ids = torch.zeros(bs, 1, dtype=torch.long, device=device)
            s_block_table = torch.zeros(bs, max_blocks_per_seq, dtype=torch.int32, device=device)
            s_cache_seqlens = torch.zeros(bs, dtype=torch.int32, device=device)

            # Warmup：让 CUDA 分配好 workspace tensor，避免 graph 捕获时触发动态分配
            for _ in range(warmup_iters):
                self._paged_ctx.set(s_block_table, s_cache_seqlens, fixed_max_kv_len)
                with torch.no_grad():
                    _ = self.model(
                        input_ids=s_input_ids,
                        position_ids=s_position_ids,
                        use_cache=False,
                    )
                self._paged_ctx.clear()
            torch.cuda.synchronize()

            # 捕获 CUDA Graph
            g = torch.cuda.CUDAGraph()
            self._paged_ctx.set(s_block_table, s_cache_seqlens, fixed_max_kv_len)
            with torch.cuda.graph(g):
                with torch.no_grad():
                    graph_out = self.model(
                        input_ids=s_input_ids,
                        position_ids=s_position_ids,
                        use_cache=False,
                    )
                # logits[:, 0, :] 的 slice 也在 graph 内完成，保证形状固定
                s_logits = graph_out.logits[:, 0, :]  # [bs, vocab_size]
            self._paged_ctx.clear()

            # bt_staging：预分配 block_table staging buffer，避免 _graph_decode_forward
            # 每次调用都分配临时 GPU tensor（固定形状 [bs, max_blocks_per_seq]）
            s_bt_staging = torch.zeros(bs, max_blocks_per_seq, dtype=torch.int32, device=device)

            self._cuda_graphs[bs] = g
            self._graph_static[bs] = {
                "input_ids": s_input_ids,
                "position_ids": s_position_ids,
                "block_table": s_block_table,
                "cache_seqlens": s_cache_seqlens,
                "logits": s_logits,
                "bt_staging": s_bt_staging,
            }

    def _find_padded_bs(self, actual_bs: int) -> int | None:
        """找到 >= actual_bs 的最小已捕获 batch size；不存在则返回 None（走 eager 路径）。"""
        for padded in sorted(self._cuda_graphs.keys()):
            if padded >= actual_bs:
                return padded
        return None

    def _graph_decode_forward(
        self,
        actual_bs: int,
        padded_bs: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Phase 12：CUDA Graph 回放路径。

        将实际 batch（actual_bs）的数据 copy_() 进静态 buffer，pad 到 padded_bs，
        replay graph，返回 actual_bs 行的 logits（已 clone，脱离静态 buffer）。

        注意：paged_ctx 在此处设置并清除，max_kv_len 固定为 config.max_model_len。
        """
        static = self._graph_static[padded_bs]
        max_blocks = static["block_table"].shape[1]

        # 1. 用预分配的 staging buffer 拼接 padded block_table（避免每步分配临时 tensor）
        bt_cols = block_table.shape[1]
        staging = static["bt_staging"]
        staging.zero_()
        staging[:actual_bs, : min(bt_cols, max_blocks)].copy_(
            block_table[:, : min(bt_cols, max_blocks)]
        )

        # 2. copy_() 更新静态 buffer（in-place，不改变 shape）
        static["input_ids"][:actual_bs].copy_(input_ids)
        if actual_bs < padded_bs:
            static["input_ids"][actual_bs:].zero_()
        static["position_ids"][:actual_bs].copy_(position_ids)
        if actual_bs < padded_bs:
            static["position_ids"][actual_bs:].zero_()
        static["cache_seqlens"][:actual_bs].copy_(cache_seqlens)
        if actual_bs < padded_bs:
            static["cache_seqlens"][actual_bs:].zero_()
        static["block_table"].copy_(staging)

        # 3. 设置 paged context（max_kv_len 固定，与 graph 捕获时一致）
        self._paged_ctx.set(
            static["block_table"],
            static["cache_seqlens"],
            self.config.max_model_len,
        )

        # 4. Replay + try/finally 保证异常时 paged_ctx 被清除
        #    （不清除会导致下一次 prefill 误走 decode 路径）
        try:
            self._cuda_graphs[padded_bs].replay()
        finally:
            self._paged_ctx.clear()

        return static["logits"][:actual_bs].clone()

    def _eager_decode_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_kv_len: int,
    ) -> torch.Tensor:
        """Phase 12：抽取 eager forward 为独立方法，供 graph 回路降级使用。"""
        self._paged_ctx.set(block_table, cache_seqlens, max_kv_len)
        try:
            with torch.profiler.record_function("model_forward"):
                with torch.no_grad():
                    out = self.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        use_cache=False,
                    )
            logits = out.logits[:, 0, :].clone()  # [batch, vocab_size]
            del out
            return logits
        finally:
            self._paged_ctx.clear()
