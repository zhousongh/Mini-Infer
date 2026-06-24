"""
Phase 3 / Phase 6 / Phase 10 Paged KV Cache 管理器。

Phase 10 新增接口（Prefix Caching）：
  - find_prefix_cache(token_ids)：查找前缀命中，返回 (cached_len, phys_block_ids)
  - init_request_with_prefix(state, cached_len, cached_blocks)：复用前缀物理块初始化请求
  - write_prefill_kv_suffix(request_id, past_key_values, cached_len)：只写后缀部分 KV
  - get_prefix_kv(phys_block_ids, cached_len)：从 block tensor 重建 DynamicCache
  - register_prefix_blocks(token_ids, block_table)：注册新计算的 blocks 到前缀缓存
  - register_prefix_blocks_for_request(request_id, token_ids)：按 request_id 注册
  - evict_lru_prefix_block()：LRU 淘汰（ref_count==1 才可淘汰）
  物理块引用计数（_ref_count）统一管理 prefix cache 共享块的生命周期：
    · _allocate_block() 分配时 ref_count = 1（请求持有）
    · register_prefix_blocks() 注册时 ref_count += 1（cache 额外持有）
    · free_request() 减 1；降到 0 才归还 _free_blocks（cached block 不提前释放）
    · evict_lru_prefix_block() 减 1；降到 0 时归还 _free_blocks

Phase 6 新增接口（True PagedAttention）：
  - ensure_next_slot()：decode 前预分配下一 token 的物理块
  - build_block_tables()：构造 flash_attn_with_kvcache 所需的 block_table / cache_seqlens 张量
  - advance_seq_lens()：flash_attn in-place 写完 KV 后递增 _seq_lens

Phase 3 优化：gather_batch_kv() 从嵌套 Python 循环改为向量化 advanced indexing，
减少 Python 解释器开销，所有 block gather 操作合并为单次 CUDA kernel 调用。

核心设计：预分配固定大小的 GPU block tensor 池，每个请求持有一个 BlockTable
（逻辑块号 → 物理块号的映射），通过 FreeBlockPool（deque）管理可用物理块。
相比 Phase 1 的 HF past_key_values dict，优势在于：
  - 显存有上限（num_gpu_blocks 固定）
  - 支持 batch decode（不同长度请求共享同一个 block pool）
  - free_request 引用计数归零后归还物理块，无显存碎片

dry_run=True 时不分配 GPU tensor，仅做块管理逻辑的元数据追踪，供无 GPU 环境的测试使用。
"""

import hashlib
import math
import struct
from collections import OrderedDict, deque
from dataclasses import dataclass

import torch

from ..core.config import EngineConfig
from ..core.request import RequestState


def _resolve_dtype(dtype: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]


@dataclass(slots=True)
class KVAllocation:
    """记录单个请求当前的块分配情况，供外部查询和测试使用。"""

    allocated_blocks: int = 0
    cached_tokens: int = 0


class PrefixCacheManager:
    """
    前缀缓存管理（Phase 10）：维护 block-level hash → 物理块映射和 LRU 淘汰顺序。

    只负责 hash/LRU 查找逻辑，不直接操作 GPU tensor 或物理块生命周期。
    物理块的 ref_count 和 free_blocks 由 KVCacheManager 持有，由调用方更新。

    设计原则：
      - compute_hashes / find / register 只读写 self._cache / self._lru
      - try_evict 接收 ref_count dict，清理后返回物理块 id，调用方负责归还 free_blocks
      这样 PrefixCacheManager 可以被替换（如换成 RadixAttention 的 trie 结构）
      而不影响 KVCacheManager 的块池管理逻辑。
    """

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        # hash → phys_block_id
        self._cache: dict[int, int] = {}
        # LRU 顺序（最近使用在末尾，淘汰时从头取）
        self._lru: OrderedDict[int, None] = OrderedDict()

    def compute_hashes(self, token_ids: list[int]) -> list[int]:
        """
        计算 token_ids 各完整 block 的链式 hash。

        每个 block 的 hash 包含前缀历史（chain），避免相同 token 在不同位置错误命中。
        只计算完整 block（末尾不足 block_size 的部分不参与 hash）。

        限制最大可缓存 block 数：
          - prompt_len % block_size == 0：只缓存前 N-1 块，保留最后一块作为 suffix
          - 否则：缓存所有完整 block（partial tail 自然成为 suffix）
        保证 find 返回的 cached_len < prompt_len，避免 suffix 为空时传入空 input_ids。
        """
        num_full_blocks = len(token_ids) // self.block_size
        max_cacheable = (num_full_blocks - 1) if (
            len(token_ids) > 0 and len(token_ids) % self.block_size == 0
        ) else num_full_blocks

        hashes: list[int] = []
        prev_hash = 0
        for i in range(max_cacheable):
            start = i * self.block_size
            end = start + self.block_size
            # SHA-256 确保确定性（避免 PYTHONHASHSEED 随机化）和低碰撞概率
            buf = struct.pack(
                f">Q{end - start}i",
                prev_hash & 0xFFFFFFFFFFFFFFFF,
                *token_ids[start:end],
            )
            block_hash = int.from_bytes(hashlib.sha256(buf).digest()[:8], "big")
            hashes.append(block_hash)
            prev_hash = block_hash
        return hashes

    def find(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """
        查找最长前缀命中，返回 (cached_len, phys_block_ids)。

        cached_len 按 block_size 对齐，且 < len(token_ids)（保证非空 suffix）。
        命中时更新 LRU 顺序。
        """
        hashes = self.compute_hashes(token_ids)
        cached_blocks: list[int] = []
        for block_hash in hashes:
            if block_hash not in self._cache:
                break
            cached_blocks.append(self._cache[block_hash])
            self._lru.move_to_end(block_hash)
        return len(cached_blocks) * self.block_size, cached_blocks

    def register(self, token_ids: list[int], block_table: list[int]) -> list[int]:
        """
        将 prompt 完整 block 注册到前缀缓存，更新 LRU。

        已缓存的 block 只刷新 LRU 顺序；新 block 加入缓存。
        返回新加入缓存的物理块列表，调用方负责对这些块执行 ref_count += 1。
        """
        hashes = self.compute_hashes(token_ids)
        new_blocks: list[int] = []
        for block_hash, phys_block in zip(hashes, block_table):
            if block_hash in self._cache:
                self._lru.move_to_end(block_hash)
                continue
            self._cache[block_hash] = phys_block
            self._lru[block_hash] = None
            new_blocks.append(phys_block)
        return new_blocks

    def try_evict(self, ref_count: dict[int, int]) -> int | None:
        """
        尝试淘汰 LRU 中 ref_count == 1 的块（仅 cache 持有，无请求引用）。

        成功：从 cache / LRU 中移除并清理 ref_count，返回物理块 id（调用方归还 free_blocks）。
        失败：返回 None（所有缓存块均被请求引用，不可淘汰）。
        """
        for block_hash in list(self._lru.keys()):
            phys_block = self._cache[block_hash]
            if ref_count.get(phys_block, 1) == 1:
                del self._cache[block_hash]
                del self._lru[block_hash]
                ref_count.pop(phys_block, None)
                return phys_block
        return None

    def count_evictable(self, ref_count: dict[int, int]) -> int:
        """返回当前 LRU 中 cache-only、可淘汰的 block 数。"""
        count = 0
        for block_hash in self._lru.keys():
            phys_block = self._cache[block_hash]
            if ref_count.get(phys_block, 1) == 1:
                count += 1
        return count

    def size(self) -> int:
        """返回当前缓存的 block 数。"""
        return len(self._cache)


class KVCacheManager:
    """
    Paged KV cache manager：预分配 GPU block tensor 池，用 BlockTable 管理每个请求。

    存储格式：
        k_cache[layer_idx] shape: [num_gpu_blocks, block_size, num_kv_heads, head_dim]
        v_cache[layer_idx] shape: 同上

    每个 block 存储 block_size 个 token 的 KV，物理块由 _free_blocks 统一管理。
    """

    def __init__(self, config: EngineConfig) -> None:
        self.block_size = config.block_size
        self.num_gpu_blocks = config.num_gpu_blocks
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.device = config.device
        self._dry_run = config.dry_run

        if not config.dry_run:
            dtype = _resolve_dtype(config.dtype)
            # k_cache[l][block_id, slot_id, num_kv_heads, head_dim]
            self.k_cache: list[torch.Tensor] = [
                torch.zeros(
                    self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_dim,
                    device=self.device, dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]
            self.v_cache: list[torch.Tensor] = [
                torch.zeros(
                    self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_dim,
                    device=self.device, dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]

        # 空闲物理块队列
        self._free_blocks: deque[int] = deque(range(self.num_gpu_blocks))
        # 每个请求的逻辑块 → 物理块映射
        self._block_tables: dict[str, list[int]] = {}
        # 每个请求已存入的 KV token 数量
        self._seq_lens: dict[str, int] = {}
        # 每个活跃 GPU 请求预留的最大逻辑 KV block 数（含已分配块）。
        # 预留只影响调度准入，不会提前分配物理块。
        self._reserved_blocks: dict[str, int] = {}

        # Phase 10：前缀缓存管理器（hash/LRU 逻辑）+ 物理块引用计数
        self._pfx = PrefixCacheManager(self.block_size)
        self._ref_count: dict[int, int] = {}

    # ------------------------------------------------------------------
    # 块池管理
    # ------------------------------------------------------------------

    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def _outstanding_reserved_blocks(self) -> int:
        """返回已预留但尚未实际分配的 block 数。"""
        total = 0
        for request_id, reserved in self._reserved_blocks.items():
            allocated = len(self._block_tables.get(request_id, []))
            total += max(0, reserved - allocated)
        return total

    def num_available_blocks_for_reservation(self, include_reclaimable: bool = True) -> int:
        """
        返回还可以承诺给新请求的 block 数。

        free_blocks 是真实空闲物理块；reclaimable_prefix_blocks 是需要时可被
        LRU 淘汰的 cache-only 前缀块；outstanding_reserved_blocks 是已经承诺
        给活跃请求未来 decode 使用、但尚未实际分配的块。
        """
        reclaimable = self.num_reclaimable_prefix_blocks() if include_reclaimable else 0
        return self.num_free_blocks() + reclaimable - self._outstanding_reserved_blocks()

    def can_reserve(self, num_blocks: int, include_reclaimable: bool = True) -> bool:
        """判断当前是否还能为新准入请求预留 num_blocks 个块。"""
        return (
            self.num_available_blocks_for_reservation(
                include_reclaimable=include_reclaimable
            )
            >= num_blocks
        )

    def get_reserved_blocks(self, request_id: str) -> int:
        """返回单个请求的预留 block 数；未记录时退化为当前已分配 block 数。"""
        allocated = len(self._block_tables.get(request_id, []))
        return max(self._reserved_blocks.get(request_id, allocated), allocated)

    def _allocate_block(self) -> int:
        if not self._free_blocks:
            # 尝试淘汰 LRU 前缀缓存块（ref_count==1 才可淘汰）
            if not self.evict_lru_prefix_block():
                raise RuntimeError(
                    "KV cache 已满：没有可用的空闲块，且没有可淘汰的前缀缓存块。"
                    "请减少并发请求数或增大 num_gpu_blocks。"
                )
        block = self._free_blocks.popleft()
        self._ref_count[block] = 1  # 分配时引用计数初始化为 1（请求持有）
        return block

    # ------------------------------------------------------------------
    # 请求生命周期
    # ------------------------------------------------------------------

    def init_request(self, state: RequestState, reserved_blocks: int | None = None) -> None:
        """为新请求分配 prompt 所需的初始物理块，并初始化 seq_len。"""
        request_id = state.request.request_id
        prompt_len = len(state.prompt_token_ids)
        num_blocks = max(1, math.ceil(prompt_len / self.block_size))
        self._block_tables[request_id] = [self._allocate_block() for _ in range(num_blocks)]
        self._reserved_blocks[request_id] = max(
            reserved_blocks if reserved_blocks is not None else num_blocks,
            num_blocks,
        )
        # seq_len 初始化为 prompt_len：prefill 写完后，KV 位置 0..prompt_len-1 会被填充
        self._seq_lens[request_id] = prompt_len

    def free_request(self, state: RequestState) -> None:
        """
        归还请求的所有物理块。

        Phase 10：引用计数控制释放。缓存中的共享块 ref_count > 1，
        free_request 只递减计数，降到 0 时才归还 _free_blocks。
        """
        request_id = state.request.request_id
        blocks = self._block_tables.pop(request_id, [])
        self._reserved_blocks.pop(request_id, None)
        for block in blocks:
            count = self._ref_count.get(block, 1) - 1
            if count <= 0:
                self._free_blocks.append(block)
                self._ref_count.pop(block, None)
            else:
                self._ref_count[block] = count
        self._seq_lens.pop(request_id, None)

    # ------------------------------------------------------------------
    # KV 写入（prefill 阶段）
    # ------------------------------------------------------------------

    def write_prefill_kv(self, request_id: str, past_key_values: tuple) -> None:
        """
        把 HF 模型 prefill 输出的 past_key_values 写入 block tensor。

        past_key_values: tuple of (K, V) per layer
            K shape: [1, num_kv_heads, prompt_len, head_dim]
        """
        if self._dry_run:
            # dry_run 模式：seq_len 已在 init_request 中设置，无需操作
            return

        actual_layers = len(past_key_values)
        if actual_layers != self.num_layers:
            raise ValueError(
                f"模型输出了 {actual_layers} 层 KV，但 config.num_hidden_layers={self.num_layers}。"
                f"请确认 EngineConfig 的 num_hidden_layers 与模型匹配。"
            )

        prompt_len = past_key_values[0][0].shape[2]
        expected_blocks = max(1, math.ceil(prompt_len / self.block_size))
        actual_blocks = len(self._block_tables[request_id])
        if prompt_len != self._seq_lens[request_id]:
            raise ValueError(
                f"请求 {request_id!r}：init_request 时 prompt_len={self._seq_lens[request_id]}，"
                f"但 past_key_values 显示 seq_len={prompt_len}。tokenize 结果与模型输入不一致。"
            )
        if actual_blocks < expected_blocks:
            raise RuntimeError(
                f"请求 {request_id!r}：prompt 需要 {expected_blocks} 块，但只分配了 {actual_blocks} 块。"
            )

        block_table = self._block_tables[request_id]

        for li in range(self.num_layers):
            # k: [1, num_kv_heads, prompt_len, head_dim] → squeeze → permute → [prompt_len, heads, dim]
            k = past_key_values[li][0][0].permute(1, 0, 2)  # [prompt_len, num_kv_heads, head_dim]
            v = past_key_values[li][1][0].permute(1, 0, 2)

            for blk_idx, phys_blk in enumerate(block_table):
                start = blk_idx * self.block_size
                end = min(start + self.block_size, prompt_len)
                n = end - start
                if n <= 0:
                    break
                self.k_cache[li][phys_blk, :n] = k[start:end]
                self.v_cache[li][phys_blk, :n] = v[start:end]

    # ------------------------------------------------------------------
    # KV 读取（decode 阶段：gather 用于 batch forward）
    # ------------------------------------------------------------------

    def gather_batch_kv(
        self,
        request_ids: list[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
        """
        为 batch decode 聚合所有请求的 KV。使用左填充（left-padding）对齐到 max_seq_len。

        Phase 3 优化：用向量化 advanced indexing 替代嵌套 Python 循环，
        将所有块的 gather 合并为单次 tensor 索引操作。

        算法：
          1. 构建 block_table_tensor [batch, max_num_blocks]
          2. 计算每个输出位置对应的 token 位置（含左填充偏移）
          3. 由 token 位置推算物理块号和块内 slot 号（两个 [batch, max_seq_len] 索引张量）
          4. 一次 advanced indexing 完成 gather，乘以 valid_mask 置零填充位置
          5. permute → [batch, num_kv_heads, max_seq_len, head_dim]

        返回：
            k_batch: list[num_layers]，每个 shape [batch, num_kv_heads, max_seq_len, head_dim]
            v_batch: 同上
            seq_lens: 每个请求的实际 seq_len
        """
        seq_lens = [self._seq_lens[rid] for rid in request_ids]
        max_seq_len = max(seq_lens)
        batch_size = len(request_ids)

        # --- 1. 构建 block table tensor [batch, max_num_blocks] ---
        max_num_blocks = max(len(self._block_tables[rid]) for rid in request_ids)
        block_table_tensor = torch.zeros(
            batch_size, max_num_blocks, dtype=torch.long, device=self.device
        )
        for b, rid in enumerate(request_ids):
            blocks = self._block_tables[rid]
            block_table_tensor[b, : len(blocks)] = torch.tensor(
                blocks, dtype=torch.long, device=self.device
            )

        # --- 2. 计算 token 位置（含左填充偏移）---
        # seq_lens_t: [batch, 1], out_positions: [1, max_seq_len]
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=self.device).unsqueeze(1)
        out_positions = torch.arange(max_seq_len, device=self.device).unsqueeze(0)
        # token_positions[b, i] = i - (max_seq_len - seq_len[b])；负值为填充区
        token_positions = out_positions - (max_seq_len - seq_lens_t)  # [batch, max_seq_len]

        # --- 3. 推算物理块号和 slot 号（填充区 clamp 到 0，结果被 mask 置零）---
        valid_mask = token_positions >= 0  # [batch, max_seq_len]
        token_pos_clamped = token_positions.clamp(min=0)

        block_indices = token_pos_clamped // self.block_size   # [batch, max_seq_len]
        slot_indices  = token_pos_clamped % self.block_size    # [batch, max_seq_len]

        # phys_blocks[b, i] = block_table_tensor[b, block_indices[b, i]]
        batch_range = (
            torch.arange(batch_size, device=self.device)
            .unsqueeze(1)
            .expand_as(block_indices)
        )
        phys_blocks = block_table_tensor[batch_range, block_indices]  # [batch, max_seq_len]

        # valid_mask_f 用于置零填充区，shape [batch, max_seq_len, 1, 1]，dtype 与 cache 一致
        cache_dtype = self.k_cache[0].dtype
        valid_mask_f = valid_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=cache_dtype)

        # --- 4 & 5. 各层 gather + permute ---
        k_batch: list[torch.Tensor] = []
        v_batch: list[torch.Tensor] = []

        for li in range(self.num_layers):
            # k_cache[li]: [num_gpu_blocks, block_size, num_kv_heads, head_dim]
            # advanced indexing → [batch, max_seq_len, num_kv_heads, head_dim]
            k_tokens = self.k_cache[li][phys_blocks, slot_indices] * valid_mask_f
            v_tokens = self.v_cache[li][phys_blocks, slot_indices] * valid_mask_f
            # permute → [batch, num_kv_heads, max_seq_len, head_dim]
            k_batch.append(k_tokens.permute(0, 2, 1, 3))
            v_batch.append(v_tokens.permute(0, 2, 1, 3))

        return k_batch, v_batch, seq_lens

    # ------------------------------------------------------------------
    # KV 写回（decode 阶段：写入新 token 的 KV）
    # ------------------------------------------------------------------

    def write_decode_kv(
        self,
        request_ids: list[str],
        k_new_layers: list[torch.Tensor] | None,
        v_new_layers: list[torch.Tensor] | None,
    ) -> None:
        """
        把 decode forward 输出的新 token KV 写回 block tensor，并递增 seq_len。

        k_new_layers[li] shape: [batch, num_kv_heads, head_dim]
        dry_run 模式或 k_new_layers=None 时，仅更新块管理元数据（seq_len + 块分配）。
        """
        for b, rid in enumerate(request_ids):
            token_pos = self._seq_lens[rid]
            block_idx = token_pos // self.block_size

            # 如需新块则分配
            if block_idx >= len(self._block_tables[rid]):
                self._block_tables[rid].append(self._allocate_block())
                self._reserved_blocks[rid] = max(
                    self._reserved_blocks.get(rid, 0),
                    len(self._block_tables[rid]),
                )

            if not self._dry_run and k_new_layers is not None:
                slot_idx = token_pos % self.block_size
                phys_blk = self._block_tables[rid][block_idx]
                for li in range(self.num_layers):
                    # k_new_layers[li][b] shape: [num_kv_heads, head_dim]
                    self.k_cache[li][phys_blk, slot_idx] = k_new_layers[li][b]
                    self.v_cache[li][phys_blk, slot_idx] = v_new_layers[li][b]  # type: ignore[index]

            self._seq_lens[rid] += 1

    # ------------------------------------------------------------------
    # Phase 7：Preemption — GPU ↔ CPU KV 换出 / 换入
    # ------------------------------------------------------------------

    def swap_out(self, state: "RequestState") -> None:
        """
        将请求的 GPU KV 拷贝到 CPU，释放 GPU 物理块。

        执行后：
          - state.swapped_seq_len = 换出时的 seq_len（用于 swap_in 重建块分配）
          - state.cpu_kv = [(k_cpu_l0, v_cpu_l0), ...]（真实模式，dry_run 下为 None）
          - GPU 物理块已归还到 _free_blocks
        """
        request_id = state.request.request_id
        seq_len = self._seq_lens[request_id]
        state.swapped_seq_len = seq_len

        if not self._dry_run:
            block_table = self._block_tables[request_id]
            kv_dtype = self.k_cache[0].dtype  # 保持与 GPU cache 相同的 dtype（fp16/bf16），避免 2× 内存浪费
            cpu_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for li in range(self.num_layers):
                k_cpu = torch.zeros(seq_len, self.num_kv_heads, self.head_dim, dtype=kv_dtype)
                v_cpu = torch.zeros(seq_len, self.num_kv_heads, self.head_dim, dtype=kv_dtype)
                for blk_idx, phys_blk in enumerate(block_table):
                    start = blk_idx * self.block_size
                    end = min(start + self.block_size, seq_len)
                    n = end - start
                    if n <= 0:
                        break
                    k_cpu[start:end] = self.k_cache[li][phys_blk, :n].cpu()
                    v_cpu[start:end] = self.v_cache[li][phys_blk, :n].cpu()
                cpu_kv.append((k_cpu, v_cpu))
            state.cpu_kv = cpu_kv

        # 释放 GPU 块（dry_run 下只做元数据清理）
        self.free_request(state)
        # Phase 10：清除 prefix 状态，确保 swap_in 后 prefix_cached_blocks 不再引用
        # 已归还的物理块（避免后续误用）。swap_in 会重新分配所有块，不走 prefix 路径。
        state.prefix_cached_len = 0
        state.prefix_cached_blocks = []

    def swap_in(self, state: "RequestState", reserved_blocks: int | None = None) -> None:
        """
        将 CPU KV 拷贝回 GPU，重新分配物理块，恢复请求的 KV cache 状态。

        执行后：
          - GPU 物理块已重新分配，block_table 和 seq_len 已恢复
          - state.cpu_kv = None（CPU 副本已释放）
        """
        request_id = state.request.request_id
        seq_len = state.swapped_seq_len
        num_blocks = max(1, math.ceil(seq_len / self.block_size))

        # 重新分配 GPU 块
        self._block_tables[request_id] = [self._allocate_block() for _ in range(num_blocks)]
        self._reserved_blocks[request_id] = max(
            reserved_blocks if reserved_blocks is not None else num_blocks,
            num_blocks,
        )
        self._seq_lens[request_id] = seq_len

        if not self._dry_run and state.cpu_kv is not None:
            block_table = self._block_tables[request_id]
            for li in range(self.num_layers):
                k_cpu, v_cpu = state.cpu_kv[li]
                for blk_idx, phys_blk in enumerate(block_table):
                    start = blk_idx * self.block_size
                    end = min(start + self.block_size, seq_len)
                    n = end - start
                    if n <= 0:
                        break
                    self.k_cache[li][phys_blk, :n] = k_cpu[start:end].to(self.device)
                    self.v_cache[li][phys_blk, :n] = v_cpu[start:end].to(self.device)

        state.cpu_kv = None

    # ------------------------------------------------------------------
    # 查询接口（向后兼容）
    # ------------------------------------------------------------------

    def get_allocation(self, request_id: str) -> KVAllocation | None:
        """返回请求的块分配摘要，供测试和监控使用。"""
        if request_id not in self._block_tables:
            return None
        return KVAllocation(
            allocated_blocks=len(self._block_tables[request_id]),
            cached_tokens=self._seq_lens.get(request_id, 0),
        )

    def total_allocated_blocks(self) -> int:
        return sum(len(b) for b in self._block_tables.values())

    def stats(self) -> dict[str, object]:
        """
        返回 KV cache 当前状态的只读快照，供调度策略、日志和 benchmark 使用。

        used_blocks 表示物理 block 池中已占用的块数；request_allocated_blocks
        表示所有活跃请求 block_table 长度之和，prefix cache 共享块可能让这个值
        大于物理 used_blocks。
        """
        free_blocks = self.num_free_blocks()
        used_blocks = self.num_gpu_blocks - free_blocks
        request_blocks = {
            request_id: len(blocks)
            for request_id, blocks in self._block_tables.items()
        }
        request_reserved_blocks = {
            request_id: max(self._reserved_blocks.get(request_id, len(blocks)), len(blocks))
            for request_id, blocks in self._block_tables.items()
        }
        request_seq_lens = {
            request_id: self._seq_lens.get(request_id, 0)
            for request_id in self._block_tables
        }
        return {
            "total_blocks": self.num_gpu_blocks,
            "free_blocks": free_blocks,
            "used_blocks": used_blocks,
            "utilization": used_blocks / self.num_gpu_blocks,
            "active_requests": len(self._block_tables),
            "request_allocated_blocks": sum(request_blocks.values()),
            "reserved_blocks": sum(request_reserved_blocks.values()),
            "outstanding_reserved_blocks": self._outstanding_reserved_blocks(),
            "reclaimable_prefix_blocks": self.num_reclaimable_prefix_blocks(),
            "strict_available_blocks_for_reservation": self.num_available_blocks_for_reservation(
                include_reclaimable=False
            ),
            "available_blocks_for_reservation": self.num_available_blocks_for_reservation(),
            "request_blocks": request_blocks,
            "request_reserved_blocks": request_reserved_blocks,
            "request_seq_lens": request_seq_lens,
            "prefix_cache_blocks": self.prefix_cache_size(),
            "ref_counted_blocks": len(self._ref_count),
        }

    # ------------------------------------------------------------------
    # Phase 6：True PagedAttention（flash_attn block_tables）接口
    # ------------------------------------------------------------------

    def ensure_next_slot(self, request_ids: list[str]) -> None:
        """
        确保每个请求下一个 decode 位置已有物理块。
        必须在 build_block_tables() 之前调用，否则 block_table 会缺失新 token 对应的块。
        """
        for rid in request_ids:
            token_pos = self._seq_lens[rid]
            block_idx = token_pos // self.block_size
            if block_idx >= len(self._block_tables[rid]):
                self._block_tables[rid].append(self._allocate_block())
                self._reserved_blocks[rid] = max(
                    self._reserved_blocks.get(rid, 0),
                    len(self._block_tables[rid]),
                )

    def build_block_tables(
        self,
        request_ids: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        为 flash_attn_with_kvcache 构造 block_table 和 cache_seqlens 张量。

        返回：
            block_table:   [batch, max_blocks_per_seq] int32，padding 填 0
            cache_seqlens: [batch] int32，新 token 写入前各请求的 cache 长度

        注意：调用前必须先调用 ensure_next_slot()，确保下一写入位置的块已分配。
        """
        batch_size = len(request_ids)
        max_num_blocks = max(len(self._block_tables[rid]) for rid in request_ids)

        block_table = torch.zeros(
            batch_size, max_num_blocks, dtype=torch.int32, device=self.device
        )
        for b, rid in enumerate(request_ids):
            blocks = self._block_tables[rid]
            block_table[b, : len(blocks)] = torch.tensor(
                blocks, dtype=torch.int32, device=self.device
            )

        cache_seqlens = torch.tensor(
            [self._seq_lens[rid] for rid in request_ids],
            dtype=torch.int32,
            device=self.device,
        )
        return block_table, cache_seqlens

    def advance_seq_lens(self, request_ids: list[str]) -> None:
        """
        flash_attn_with_kvcache 已 in-place 写入新 token KV 后，递增各请求的 seq_len。
        替代 write_decode_kv 的 seq_len 更新部分（GPU KV 写入由 flash_attn 完成）。
        """
        for rid in request_ids:
            self._seq_lens[rid] += 1

    # ------------------------------------------------------------------
    # Phase 10：Prefix Caching（查找/注册/淘汰委托给 PrefixCacheManager）
    # ------------------------------------------------------------------

    def find_prefix_cache(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """
        查找 token_ids 的最长前缀命中，返回 (cached_len, phys_block_ids)。

        cached_len 按 block_size 对齐，且 < len(token_ids)（保证非空 suffix）。
        """
        return self._pfx.find(token_ids)

    def init_request_with_prefix(
        self,
        state: "RequestState",
        cached_len: int,
        cached_blocks: list[int],
        reserved_blocks: int | None = None,
    ) -> None:
        """
        为有前缀命中的请求初始化 KV cache：
          - 复用 cached_blocks（引用计数 +1）
          - 为后缀 token 分配新物理块
          - seq_len 设为 prompt_len（前缀 KV 已在缓存中）

        调用前提：cached_len 是 block_size 的整数倍且 < prompt_len（保证非空 suffix）。
        """
        request_id = state.request.request_id
        prompt_len = len(state.prompt_token_ids)
        suffix_len = prompt_len - cached_len

        # 增加缓存块的引用计数（新请求复用）
        for block in cached_blocks:
            self._ref_count[block] = self._ref_count.get(block, 1) + 1

        # 为后缀分配新块
        num_new_blocks = max(1, math.ceil(suffix_len / self.block_size)) if suffix_len > 0 else 0
        new_blocks = [self._allocate_block() for _ in range(num_new_blocks)]

        self._block_tables[request_id] = list(cached_blocks) + new_blocks
        self._reserved_blocks[request_id] = max(
            reserved_blocks if reserved_blocks is not None else len(self._block_tables[request_id]),
            len(self._block_tables[request_id]),
        )
        self._seq_lens[request_id] = prompt_len

    def register_prefix_blocks(self, token_ids: list[int], block_table: list[int]) -> None:
        """
        prefill 完成后，将 prompt 的完整 block 注册到前缀缓存。

        新加入缓存的块 ref_count += 1（cache 额外持有一个引用）。
        """
        for block in self._pfx.register(token_ids, block_table):
            self._ref_count[block] = self._ref_count.get(block, 1) + 1

    def register_prefix_blocks_for_request(self, request_id: str, token_ids: list[int]) -> None:
        """按 request_id 查找 block_table 后调用 register_prefix_blocks。"""
        block_table = self._block_tables.get(request_id, [])
        self.register_prefix_blocks(token_ids, block_table)

    def evict_lru_prefix_block(self) -> bool:
        """
        淘汰 LRU 中 ref_count == 1 的块（仅 cache 持有），归还 free_blocks。
        返回 True 表示成功淘汰，False 表示没有可淘汰的 block。
        """
        block = self._pfx.try_evict(self._ref_count)
        if block is not None:
            self._free_blocks.append(block)
            return True
        return False

    def num_reclaimable_prefix_blocks(self) -> int:
        """返回当前 prefix cache 中可淘汰、可回收的 block 数。"""
        return self._pfx.count_evictable(self._ref_count)

    def write_prefill_kv_suffix(
        self,
        request_id: str,
        past_key_values: tuple,
        cached_len: int,
    ) -> None:
        """
        把 prefill 输出的 past_key_values 中后缀部分（cached_len 之后）写入 block tensor。

        前 cached_len 个 token 的 KV 已在共享 block 中，跳过不写，避免覆盖共享数据。
        past_key_values[li] 形状：(k[1, kv_heads, prompt_len, head_dim],
                                   v[1, kv_heads, prompt_len, head_dim])
        """
        if self._dry_run:
            return

        prompt_len = past_key_values[0][0].shape[2]
        block_table = self._block_tables[request_id]
        # 新块从这个索引开始（cached_len 是 block_size 整数倍）
        first_new_block_idx = cached_len // self.block_size

        for li in range(self.num_layers):
            k = past_key_values[li][0][0].permute(1, 0, 2)  # [prompt_len, kv_heads, head_dim]
            v = past_key_values[li][1][0].permute(1, 0, 2)
            for blk_idx in range(first_new_block_idx, len(block_table)):
                phys_blk = block_table[blk_idx]
                start = blk_idx * self.block_size
                end = min(start + self.block_size, prompt_len)
                n = end - start
                if n <= 0:
                    break
                self.k_cache[li][phys_blk, :n] = k[start:end]
                self.v_cache[li][phys_blk, :n] = v[start:end]

    def get_prefix_kv(self, phys_block_ids: list[int], cached_len: int):
        """
        从 block tensor 重建前缀 KV，返回 DynamicCache（供 HF model forward 作为 past_key_values）。

        返回的 DynamicCache 包含 cached_len 个 token 的 KV，形状：
            key_cache[li]: [1, num_kv_heads, cached_len, head_dim]
        只在 dry_run=False 路径使用（dry_run 下无 GPU tensor）。
        """
        if self._dry_run:
            raise RuntimeError("get_prefix_kv 不支持 dry_run 模式（无 GPU tensor）")

        from transformers import DynamicCache
        cache = DynamicCache()
        for li in range(self.num_layers):
            k_tensor = torch.zeros(
                1, self.num_kv_heads, cached_len, self.head_dim,
                device=self.device, dtype=self.k_cache[li].dtype,
            )
            v_tensor = torch.zeros_like(k_tensor)
            for blk_idx, phys_blk in enumerate(phys_block_ids):
                start = blk_idx * self.block_size
                end = min(start + self.block_size, cached_len)
                n = end - start
                if n <= 0:
                    break
                # k_cache[l][phys_blk, :n]: [n, kv_heads, head_dim] → [1, kv_heads, n, head_dim]
                k_tensor[0, :, start:end, :] = self.k_cache[li][phys_blk, :n].permute(1, 0, 2)
                v_tensor[0, :, start:end, :] = self.v_cache[li][phys_blk, :n].permute(1, 0, 2)
            cache.update(k_tensor, v_tensor, li)
        return cache

    def prefix_cache_size(self) -> int:
        """返回当前前缀缓存中的 block 数，供测试和监控使用。"""
        return self._pfx.size()

    # ------------------------------------------------------------------
    # Phase 11：Speculative Decoding — KV 回滚
    # ------------------------------------------------------------------

    def rollback_to(self, request_id: str, seq_len: int) -> None:
        """
        Phase 11：Speculative Decoding 回滚接口。

        将请求的 KV cache 截断到 seq_len 长度，释放多余的物理块。
        用于 rejection sampling 后丢弃未被接受的 draft token 对应的 KV。

        调用前提：seq_len <= 当前 _seq_lens[request_id]，且 request_id 已存在。
        对 prefix cache 共享块（ref_count > 1）的释放仅递减引用计数，不归还 _free_blocks。
        """
        if request_id not in self._block_tables:
            return
        current_seq_len = self._seq_lens.get(request_id, 0)
        if seq_len >= current_seq_len:
            return  # 无需截断

        # 目标需要的块数（至少保留 1 块）
        target_num_blocks = max(1, math.ceil(seq_len / self.block_size))
        current_blocks = self._block_tables[request_id]

        if len(current_blocks) > target_num_blocks:
            excess = current_blocks[target_num_blocks:]
            self._block_tables[request_id] = current_blocks[:target_num_blocks]
            for block in excess:
                count = self._ref_count.get(block, 1) - 1
                if count <= 0:
                    self._free_blocks.append(block)
                    self._ref_count.pop(block, None)
                else:
                    self._ref_count[block] = count

        self._seq_lens[request_id] = seq_len

    # ------------------------------------------------------------------
    # 废弃接口（保留以减少测试迁移成本，不再有实际功能）
    # ------------------------------------------------------------------

    def append_token(self, state: RequestState, token_id: int) -> None:
        """已废弃：Phase 2 中 KV 写回通过 write_decode_kv 完成。"""
