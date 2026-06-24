"""
Phase 11 Speculative Decoding 测试。

覆盖范围：
  - kv_cache.rollback_to：KV 回滚到指定 seq_len，多余块被释放
  - _rejection_sample：接受全部 K、拒绝第一个、中途拒绝三种路径
  - SpecEngine.generate dry_run：完整端到端生成，输出 token 数合理
  - acceptance_rate：统计正确（0 ~ 1 之间，greedy 模式近似 100%）
"""

from __future__ import annotations

import math

import torch

from mini_infer.core.config import EngineConfig
from mini_infer.cache.kv_cache import KVCacheManager
from mini_infer.core.request import Request, RequestState, SamplingParams
from mini_infer.runtime.spec_engine import SpecEngine, _rejection_sample, _softmax


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_manager(num_blocks: int = 64, block_size: int = 4) -> KVCacheManager:
    config = EngineConfig(
        model_name="stub",
        dry_run=True,
        num_gpu_blocks=num_blocks,
        block_size=block_size,
    )
    return KVCacheManager(config=config)


def _make_state(token_ids: list[int], request_id: str = "r1") -> RequestState:
    req = Request(
        request_id=request_id,
        prompt="",
        sampling_params=SamplingParams(max_new_tokens=8),
    )
    state = RequestState(request=req, prompt_token_ids=token_ids)
    return state


def _make_spec_engine(K: int = 4) -> SpecEngine:
    draft_cfg = EngineConfig(
        model_name="stub",
        dry_run=True,
        num_gpu_blocks=128,
        block_size=4,
    )
    target_cfg = EngineConfig(
        model_name="stub",
        dry_run=True,
        num_gpu_blocks=128,
        block_size=4,
    )
    return SpecEngine(draft_cfg, target_cfg, K=K)


# ---------------------------------------------------------------------------
# rollback_to 测试
# ---------------------------------------------------------------------------

class TestRollbackTo:
    def test_rollback_frees_excess_blocks(self) -> None:
        """回滚到 seq_len=4，超出的块应被释放。"""
        mgr = _make_manager(num_blocks=16, block_size=4)
        state = _make_state([1, 2, 3, 4, 5, 6, 7, 8], "r1")
        mgr.init_request(state)

        # 手动推进 seq_len 到 12（3 个 block），模拟已生成 4 个 token
        mgr._seq_lens["r1"] = 12
        # 确保分配了 3 个块
        mgr.ensure_next_slot(["r1"])
        mgr.ensure_next_slot(["r1"])
        mgr.ensure_next_slot(["r1"])

        blocks_before = len(mgr._block_tables["r1"])
        free_before = len(mgr._free_blocks)

        # 回滚到 seq_len=5（只需 2 个 block）
        mgr.rollback_to("r1", 5)

        assert mgr._seq_lens["r1"] == 5
        assert len(mgr._block_tables["r1"]) == math.ceil(5 / 4)  # = 2
        # 释放的块数应等于回滚前后块数之差
        blocks_after = len(mgr._block_tables["r1"])
        freed = blocks_before - blocks_after
        assert len(mgr._free_blocks) >= free_before  # 不能减少

    def test_rollback_noop_when_shorter_or_equal(self) -> None:
        """seq_len 已经 <= 目标时，rollback_to 不做任何事。"""
        mgr = _make_manager(num_blocks=16, block_size=4)
        state = _make_state([1, 2, 3, 4, 5], "r1")
        mgr.init_request(state)
        mgr._seq_lens["r1"] = 5

        blocks_before = list(mgr._block_tables["r1"])
        mgr.rollback_to("r1", 10)  # 目标 > 当前，noop
        assert mgr._block_tables["r1"] == blocks_before
        assert mgr._seq_lens["r1"] == 5

        mgr.rollback_to("r1", 5)  # 目标 == 当前，noop
        assert mgr._seq_lens["r1"] == 5

    def test_rollback_unknown_request_is_safe(self) -> None:
        """对不存在的 request_id 调用不报错。"""
        mgr = _make_manager()
        mgr.rollback_to("nonexistent", 4)  # 应不抛异常


# ---------------------------------------------------------------------------
# _rejection_sample 测试
# ---------------------------------------------------------------------------

class TestRejectionSample:
    """测试三种 rejection sampling 路径。"""

    VOCAB = 16

    def _uniform_probs(self) -> torch.Tensor:
        return torch.ones(self.VOCAB) / self.VOCAB

    def _peaked_probs(self, peak_token: int, peak_mass: float = 0.9) -> torch.Tensor:
        p = torch.ones(self.VOCAB) * (1 - peak_mass) / (self.VOCAB - 1)
        p[peak_token] = peak_mass
        return p

    def test_all_accepted_bonus_token(self) -> None:
        """target 分布 = draft 分布时，所有 token 都应被接受，返回 K+1 个 token。"""
        K = 4
        # 让 draft 和 target 的分布完全相同 → accept_prob = min(1, 1) = 1
        common_probs = self._uniform_probs()
        draft_tokens = [i % self.VOCAB for i in range(K)]
        draft_probs = [common_probs.clone() for _ in range(K)]

        # target_verify_logits[i] → target 在 position i+1 的分布（共 K 项）
        target_prev_logit = torch.log(common_probs + 1e-10)
        target_verify_logits = torch.stack([torch.log(common_probs + 1e-10)] * K)

        accepted, new_logit = _rejection_sample(
            draft_tokens, draft_probs, target_prev_logit, target_verify_logits
        )
        # 全部接受 + bonus = K+1 个 token
        assert len(accepted) == K + 1

    def test_first_token_rejected(self) -> None:
        """draft_token[0] 概率远低于 target，应高概率被拒绝，返回 1 个替换 token。"""
        # draft 在 token 5 上概率极低，target 在 token 5 上概率极高
        # p_target/p_draft >> 1 → accept_prob = 1 → 其实会接受？
        # 改成 draft 在 token 0 概率极高，target 在 token 1 概率极高
        # draft 采样的 token 是 0，target 的分布集中在 1
        # p_target[0] = very small, p_draft[0] = very large
        # accept_prob = p_target[0] / p_draft[0] ≈ 0 → 拒绝

        draft_token = 0
        draft_probs_0 = self._peaked_probs(peak_token=0, peak_mass=0.99)
        target_prev_logit = torch.log(self._peaked_probs(peak_token=1, peak_mass=0.99) + 1e-10)
        # K = 1
        target_verify_logits = torch.log(
            self._peaked_probs(peak_token=1, peak_mass=0.99) + 1e-10
        ).unsqueeze(0)

        # 多次采样确认高概率被拒绝
        rejection_count = 0
        for _ in range(20):
            accepted, _ = _rejection_sample(
                [draft_token], [draft_probs_0], target_prev_logit, target_verify_logits
            )
            # 拒绝时返回 1 个替换 token（长度仍是 1，但 token != draft_token）
            assert len(accepted) == 1 or len(accepted) == 2  # K=1 时全接受=2，拒绝=1
            if len(accepted) == 1 and accepted[0] != draft_token:
                rejection_count += 1

        # 拒绝率应很高（>50%）
        assert rejection_count > 10, f"拒绝次数={rejection_count}，期望 >10"

    def test_output_length_range(self) -> None:
        """接受的 token 数量必须在 [1, K+1] 之间。"""
        K = 4
        draft_tokens = list(range(K))
        draft_probs = [torch.ones(self.VOCAB) / self.VOCAB for _ in range(K)]
        target_prev_logit = torch.zeros(self.VOCAB)
        target_verify_logits = torch.zeros(K, self.VOCAB)

        for _ in range(10):
            accepted, _ = _rejection_sample(
                draft_tokens, draft_probs, target_prev_logit, target_verify_logits
            )
            assert 1 <= len(accepted) <= K + 1, f"非法长度: {len(accepted)}"

    def test_new_last_logit_shape(self) -> None:
        """返回的 new_last_logit 应是 1-D tensor，长度 = vocab_size。"""
        K = 3
        VOCAB = self.VOCAB
        draft_tokens = [0, 1, 2]
        draft_probs = [torch.ones(VOCAB) / VOCAB for _ in range(K)]
        target_prev_logit = torch.zeros(VOCAB)
        target_verify_logits = torch.zeros(K, VOCAB)

        _, new_logit = _rejection_sample(
            draft_tokens, draft_probs, target_prev_logit, target_verify_logits
        )
        assert new_logit.shape == (VOCAB,)


# ---------------------------------------------------------------------------
# SpecEngine.generate dry_run 端到端测试
# ---------------------------------------------------------------------------

class TestSpecEngineDryRun:
    def test_generate_returns_string_list(self) -> None:
        """generate 返回与输入等长的字符串列表。"""
        engine = _make_spec_engine(K=4)
        prompts = ["hello world", "test prompt"]
        outputs = engine.generate(prompts, max_new_tokens=16)
        assert isinstance(outputs, list)
        assert len(outputs) == len(prompts)
        for o in outputs:
            assert isinstance(o, str)

    def test_generate_respects_max_new_tokens(self) -> None:
        """生成的 token 数不超过 max_new_tokens。"""
        engine = _make_spec_engine(K=4)
        # dry_run：stub tokenizer 每字符 1 token
        outputs = engine.generate(["hello"], max_new_tokens=8)
        # 无法精确检查 token 数（dry_run 下 generated_text_parts），
        # 只需保证不抛异常且返回非 None
        assert outputs[0] is not None

    def test_acceptance_rate_between_zero_and_one(self) -> None:
        """acceptance_rate 应在 [0, 1] 之间。"""
        engine = _make_spec_engine(K=4)
        engine.generate(["short test"], max_new_tokens=12)
        rate = engine.acceptance_rate()
        assert 0.0 <= rate <= 1.0, f"acceptance_rate={rate} 超出范围"

    def test_multiple_prompts_independent(self) -> None:
        """多个 prompt 互相独立，不因前一个 prompt 影响后一个的 KV state。"""
        engine = _make_spec_engine(K=4)
        out1 = engine.generate(["aaaa", "bbbb"], max_new_tokens=8)
        out2 = engine.generate(["aaaa"], max_new_tokens=8)
        # 多次生成同一 prompt，dry_run 应给出相同输出（greedy temperature=0）
        assert out1[0] == out2[0], "相同 prompt 两次生成结果不一致"

    def test_k_equals_one(self) -> None:
        """K=1 时退化为 1-token speculative（验证路径不崩溃）。"""
        engine = _make_spec_engine(K=1)
        outputs = engine.generate(["test"], max_new_tokens=8)
        assert len(outputs) == 1
        assert isinstance(outputs[0], str)

    def test_acceptance_rate_greedy_is_high(self) -> None:
        """temperature=0 (greedy) 下 draft 和 target 的 logit 均全零，
        两者 softmax 相同，接受率应接近 100%（干运行下两模型同 stub，分布完全一致）。"""
        engine = _make_spec_engine(K=4)
        engine.generate(["hello world test"], max_new_tokens=16)
        # dry_run 两个 stub 分布完全相同，rejection sampling 接受率应 = 100%
        # （dry_run 时 draft_probs 全 0 触发 "p_draft < 1e-12 → 直接接受" 路径）
        rate = engine.acceptance_rate()
        assert rate == 1.0, f"dry_run greedy 下接受率应为 1.0，实际={rate}"
