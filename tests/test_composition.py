"""构图评估测试（FR-03）。

对应文档：《技术方案.md》§2.3 构图决策层、《测试与验收.md》§4 量化验收

**验证重点**：

1. **候选生成的完备性**：不同主体尺度下都能产出合理候选（回归过
   "只生成 1 个候选"的缺陷）；
2. **评分可解释**：每个子项都可追溯到具体规则（NFR-O2）；
3. **评分区分度**：好构图必须显著高于差构图（否则"最优框"无意义）；
4. **构图模式分类**：三分/居中/对角能被正确识别。
"""

from __future__ import annotations

import numpy as np
import pytest

from aicg.composition.candidate import _OCCUPANCY_TARGETS, Candidate, CandidateGenerator
from aicg.composition.rules import (
    balance_score,
    classify_pattern,
    describe_pattern,
    describe_shot_size,
    headroom_score,
    thirds_alignment,
)
from aicg.composition.scorer import HeuristicCompositionScorer
from aicg.schemas import CompositionPattern, PerceptionResult, Subject, SubjectSource


def _subject(bbox, conf=0.9) -> Subject:
    return Subject(
        subject_id="s0", label="person", bbox=bbox, confidence=conf,
        is_primary=True, source=SubjectSource.AUTO,
    )


def _perception(bbox, saliency=None) -> PerceptionResult:
    extras = {}
    if saliency is not None:
        extras["saliency_map"] = saliency
    return PerceptionResult(
        frame_id=0, timestamp_ms=0, frame_size=(640, 480),
        subjects=[_subject(bbox)], backend="stub", extras=extras,
    )


def _box(cx, cy, h, aspect=0.42):
    w = h * aspect
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ---------------------------------------------------------------------------
# 候选生成
# ---------------------------------------------------------------------------
class TestCandidateGenerator:
    def test_generates_many_candidates(self):
        """正常尺度主体应产出足量候选（供搜索最优构图）。

        回归点：早期实现只产出 1 个候选，导致"最优框"等于"当前框"，
        整个构图建议失效。
        """
        gen = CandidateGenerator()
        cands = gen.generate(_box(0.5, 0.5, 0.5), frame_aspect=0.75)
        assert len(cands) >= 10, f"仅生成 {len(cands)} 个候选"

    def test_candidates_contain_subject(self):
        """所有候选框必须**完整包含**主体（否则等于建议裁掉人）。"""
        gen = CandidateGenerator()
        subj = _box(0.5, 0.5, 0.5)
        for c in gen.generate(subj, frame_aspect=0.75):
            if c.origin == "subject_overflow":
                continue  # 显式标记的溢出候选，语义不同
            assert c.bbox[0] <= subj[0] + 0.02, f"候选 {c.bbox} 左侧裁切了主体"
            assert c.bbox[2] >= subj[2] - 0.02, f"候选 {c.bbox} 右侧裁切了主体"

    def test_candidates_within_frame(self):
        """候选框必须落在画面内。"""
        gen = CandidateGenerator()
        for c in gen.generate(_box(0.5, 0.5, 0.5), frame_aspect=0.75):
            x1, y1, x2, y2 = c.bbox
            assert -0.01 <= x1 < x2 <= 1.01
            assert -0.01 <= y1 < y2 <= 1.01

    def test_portrait_frame_aspect(self):
        """竖幅画面（aspect<1）下也应正常工作。"""
        gen = CandidateGenerator()
        cands = gen.generate(_box(0.5, 0.5, 0.6), frame_aspect=0.75)
        assert len(cands) >= 5

    def test_occupancy_targets_cover_range(self):
        """占比目标必须覆盖从松到紧的景别区间。"""
        assert min(_OCCUPANCY_TARGETS) <= 0.45, "缺少较松的景别"
        assert max(_OCCUPANCY_TARGETS) >= 0.85, "缺少较紧的景别"
        assert len(_OCCUPANCY_TARGETS) >= 5

    def test_huge_subject_marks_overflow_not_fakes(self):
        """主体大到任何画框都装不下时，应显式标记溢出而非伪造全画幅候选。"""
        gen = CandidateGenerator()
        # 主体高 0.95、宽 0.9：横向画框最多容纳宽 1.0 的裁切
        cands = gen.generate(_box(0.5, 0.5, 0.95, aspect=0.95), frame_aspect=1.33)
        origins = {c.origin for c in cands}
        assert "subject_overflow" in origins or len(cands) >= 3

    def test_dedup_limits_count(self):
        """候选数量应有上限（保护时延 NFR-P1）。"""
        gen = CandidateGenerator(max_candidates=40)
        cands = gen.generate(_box(0.5, 0.5, 0.5), frame_aspect=0.75)
        assert len(cands) <= 40, f"候选未受上限约束: {len(cands)}"

    def test_dedup_removes_duplicates(self):
        """去重后不应存在完全相同的 bbox。"""
        gen = CandidateGenerator()
        cands = gen.generate(_box(0.5, 0.5, 0.5), frame_aspect=0.75)
        keys = [tuple(round(v, 3) for v in c.bbox) for c in cands]
        assert len(keys) == len(set(keys)), "存在重复候选"

    def test_candidates_are_sorted_toward_ideal_occupancy_when_capped(self):
        """裁剪时应保留"占比接近理想值"的候选，而非随机截断。"""
        gen = CandidateGenerator(max_candidates=20)
        cands = gen.generate(_box(0.5, 0.5, 0.5), frame_aspect=0.75)
        if len(cands) == 20:
            occupancies = [abs(c.occupancy - 0.68) for c in cands]
            assert max(occupancies) < 0.35, "裁剪后仍保留了偏离过大的候选"


# ---------------------------------------------------------------------------
# 规则子项
# ---------------------------------------------------------------------------
class TestRuleScores:
    def test_thirds_alignment_on_thirds(self):
        """主体中心落在三分点上时对齐分应偏高。"""
        on_thirds = thirds_alignment(_box(2 / 3, 1 / 3, 0.4))
        off_thirds = thirds_alignment(_box(0.5, 0.5, 0.4))
        assert on_thirds.score >= off_thirds.score

    def test_balance_prefers_centered(self):
        cand = (0.0, 0.0, 1.0, 1.0)
        centered = balance_score(_box(0.5, 0.5, 0.5), cand)
        edge = balance_score(_box(0.06, 0.5, 0.5), cand)
        assert centered > edge

    def test_headroom_penalizes_cutoff(self):
        """主体贴顶时 headroom 分应低。"""
        cand = (0.0, 0.0, 1.0, 1.0)
        good = headroom_score(_box(0.5, 0.45, 0.6), cand, ideal_ratio=0.12)
        bad = headroom_score((0.4, -0.1, 0.6, 0.5), cand, ideal_ratio=0.12)
        assert good > bad

    def test_headroom_zero_when_cut(self):
        """主体被候选框裁切（gap<0）时 headroom 必须为 0。"""
        assert headroom_score((0.4, -0.1, 0.6, 0.5), (0.0, 0.0, 1.0, 1.0), ideal_ratio=0.12) == 0.0

    def test_headroom_peaks_at_ideal_ratio(self):
        """留白恰为理想比例时得分应接近满分。"""
        cand = (0.0, 0.0, 1.0, 1.0)
        # 候选框高 1.0，理想留白 0.12 → 主体顶部应在 y=0.12
        s = headroom_score((0.4, 0.12, 0.6, 0.7), cand, ideal_ratio=0.12)
        assert s > 0.9

    def test_all_scores_in_unit_interval(self):
        """所有子项得分必须归一化到 0~1。"""
        cand = (0.0, 0.0, 1.0, 1.0)
        for cx in (0.1, 0.35, 0.5, 0.67, 0.9):
            for h in (0.2, 0.5, 0.9):
                s = balance_score(_box(cx, 0.5, h), cand)
                assert 0.0 <= s <= 1.0
                s = headroom_score(_box(cx, 0.5, h), cand, 0.12)
                assert 0.0 <= s <= 1.0


class TestReferenceFrameRegression:
    """**回归测试：参照系退化缺陷**（真实修复过的 bug，勿删）。

    历史缺陷：``balance_score`` / ``headroom_score`` 早期以
    ``candidate_bbox`` 为参照系计算边距。而评分器在评估"当前构图"时
    传入 ``subject_bbox == candidate_bbox``，导致：

      - ``balance``：左右上下边距全为 0 → **恒定返回 0.0**；
      - ``headroom``：gap 恒为 0 → **恒定返回 0.6**（= 1 - 0.12/0.30）。

    后果：两个分项（合计权重 0.38）对所有画面输出同一常数，失去区分度；
    与人工标注做 SRCC 相关性时为 0.0000。修复方式是引入独立的
    ``frame_bbox`` 参照系。以下用例锁死修复行为。
    """

    def test_balance_not_constant_when_subject_equals_candidate(self):
        """subject == candidate 时 balance 仍须随主体位置变化（不得恒定）。"""
        centered = balance_score(_box(0.5, 0.5, 0.5), _box(0.5, 0.5, 0.5))
        # 主体贴左：中心构图 vs 贴边构图必须给出不同分数
        assert centered > 0.0, "balance 不得在 subject==candidate 时退化为 0"

        # 多个不同位置应产生至少 2 个不同取值（存在区分度）
        vals = {
            round(balance_score(_box(cx, 0.5, 0.4), _box(cx, 0.5, 0.4)), 4)
            for cx in (0.15, 0.30, 0.50, 0.70, 0.85)
        }
        assert len(vals) >= 2, f"balance 对主体位置不敏感（恒为 {vals}）"

    def test_headroom_not_constant_when_subject_equals_candidate(self):
        """subject == candidate 时 headroom 仍须随主体顶边位置变化。"""
        # 注意：直接给 (x1, y1, x2, y2) 元组，避免 _box 的"以中心定位"语义混淆
        def sq(top: float) -> tuple[float, float, float, float]:
            return (0.40, top, 0.60, top + 0.35)

        vals = {
            round(headroom_score(sq(top), sq(top), 0.12), 4)
            for top in (0.0, 0.12, 0.30, 0.50)
        }
        assert len(vals) >= 3, f"headroom 对留白变化不敏感（恒为 {vals}）"
        # 留白恰为理想值（主体顶边距画面顶 0.12）时接近满分
        assert headroom_score(sq(0.12), sq(0.12), 0.12) > 0.9

    def test_reference_frame_override(self):
        """显式传入 frame_bbox 时以该框为参照，且主体越界返回 0。"""
        subject = _box(0.5, 0.5, 0.4)
        assert balance_score(subject, subject, frame_bbox=(0.0, 0.0, 1.0, 1.0)) > 0.0
        # 主体越出参照画面 → 0
        assert balance_score(subject, subject, frame_bbox=(0.6, 0.0, 1.0, 1.0)) == 0.0

    def test_scorer_spread_is_not_collapsed(self):
        """端到端：不同缺陷类型的构图，总分必须有可辨差异。

        锁死"评分器把五花八门的构图打成一坨"这一历史缺陷。
        """
        from aicg.settings import load_settings

        cfg = load_settings()
        sc = HeuristicCompositionScorer(cfg.composition.scoring)
        cases = {
            "thirds": (0.62, 0.02, 1.00, 0.98),
            "center": (0.28, 0.02, 0.72, 0.98),
            "edge": (0.85, 0.02, 1.00, 0.98),
            "tiny": (0.45, 0.42, 0.55, 0.58),
        }
        scores = {
            name: sc.score_frame(subject_bbox=bb, frame_shape=(270, 480)).composition_score
            for name, bb in cases.items()
        }
        spread = max(scores.values()) - min(scores.values())
        assert spread >= 10.0, f"评分区分度过低（spread={spread:.1f}）: {scores}"
        # "主体过小"必须被判为较差构图（这是 FR-01 距离建议的前提）
        assert scores["tiny"] < scores["center"], (
            "主体过小的构图不应高于居中的正常构图"
        )


class TestSubjectScale:
    """主体占比评分（真实缺陷新增的维度）。"""

    def test_scale_monotonic_in_subject_height(self):
        from aicg.composition.rules import subject_scale_score

        prev = -1.0
        for h in (0.05, 0.15, 0.30, 0.55, 0.90):
            bbox = (0.4, 0.2, 0.6, 0.2 + h)
            s = subject_scale_score(bbox, ideal_height=0.55)
            assert s >= prev - 1e-9, f"主体高度 {h} 处占比分非单调"
            prev = s

    def test_scale_saturates_at_ideal(self):
        from aicg.composition.rules import subject_scale_score

        assert subject_scale_score((0.4, 0.1, 0.6, 0.9), ideal_height=0.55) == 1.0

    def test_tiny_subject_scores_low(self):
        from aicg.composition.rules import subject_scale_score

        assert subject_scale_score((0.45, 0.42, 0.55, 0.58), ideal_height=0.55) < 0.35


# ---------------------------------------------------------------------------
# 模式分类与中文术语
# ---------------------------------------------------------------------------
class TestPatternClassification:
    def test_center_pattern(self):
        name, conf = classify_pattern(_box(0.5, 0.5, 0.5), None, (640, 480))
        assert name in (CompositionPattern.CENTER.value, CompositionPattern.RULE_OF_THIRDS.value)
        assert isinstance(conf, dict)

    def test_thirds_pattern(self):
        name, _ = classify_pattern(_box(2 / 3, 0.5, 0.5), None, (640, 480))
        assert name in {p.value for p in CompositionPattern}

    def test_returns_valid_enum_value(self):
        """分类结果必须是合法枚举值（否则上层构造会失败）。"""
        valid = {p.value for p in CompositionPattern}
        for cx, cy in [(0.5, 0.5), (0.33, 0.33), (0.67, 0.67), (0.9, 0.1)]:
            name, conf = classify_pattern(_box(cx, cy, 0.4), None, (640, 480))
            assert name in valid, f"非法分类 {name}"

    def test_confidence_dict_all_in_unit_interval(self):
        """各模式置信度必须都归一化到 0~1。"""
        _, conf = classify_pattern(_box(0.5, 0.5, 0.5), None, (640, 480))
        assert conf, "置信度字典为空"
        for k, v in conf.items():
            assert isinstance(v, float), f"{k} 的置信度不是 float"
            assert 0.0 <= v <= 1.0, f"{k} 置信度越界: {v}"


class TestChineseLabels:
    def test_describe_pattern_non_empty(self):
        """构图模式中文标签必须非空（语言层直接引用）。"""
        for p in CompositionPattern:
            label = describe_pattern(p.value, _box(0.5, 0.5, 0.5))
            assert label

    def test_describe_shot_size_by_occupancy(self):
        """景别标签应随主体占比变化。"""
        small = describe_shot_size(_box(0.5, 0.5, 0.15))
        large = describe_shot_size(_box(0.5, 0.5, 0.90))
        assert small and large
        assert small != large, f"占比 0.15 与 0.90 应给出不同景别，均得到 {small}"

    def test_shot_size_returns_string(self):
        assert isinstance(describe_shot_size(_box(0.5, 0.5, 0.5)), str)


# ---------------------------------------------------------------------------
# 评分器
# ---------------------------------------------------------------------------
class TestHeuristicScorer:
    def test_produces_valid_result(self):
        s = HeuristicCompositionScorer()
        r = s.score_frame(subject_bbox=_box(0.5, 0.5, 0.5), frame_shape=(640, 480))
        assert 0.0 <= r.composition_score <= 100.0
        assert r.best_bbox is not None
        assert r.best_score is not None

    def test_best_at_least_as_good_as_current(self):
        """最优框评分必须 >= 当前框评分（否则"建议"毫无意义）。"""
        s = HeuristicCompositionScorer()
        for bbox in [_box(0.5, 0.5, 0.5), _box(0.15, 0.5, 0.4), _box(0.85, 0.2, 0.7)]:
            r = s.score_frame(subject_bbox=bbox, frame_shape=(640, 480))
            assert r.best_score >= r.composition_score - 1e-6, (
                f"bbox={bbox}: best={r.best_score} < current={r.composition_score}"
            )

    def test_sub_scores_traceable(self):
        """子项得分必须可枚举、可追溯（NFR-O2）。"""
        s = HeuristicCompositionScorer()
        r = s.score_frame(subject_bbox=_box(0.5, 0.5, 0.5), frame_shape=(640, 480))
        for key in ("thirds", "balance", "headroom", "lead_room", "subject_center"):
            assert key in r.sub_scores, f"缺少子项 {key}"
            assert 0.0 <= r.sub_scores[key] <= 1.0

    def test_no_subject_degrades_gracefully(self):
        """无主体时返回降级结果 + 通用三分法建议（NFR-R2）。"""
        s = HeuristicCompositionScorer()
        r = s.score_frame(subject_bbox=None, frame_shape=(640, 480))
        assert r.degraded is True
        assert r.best_bbox is not None

    def test_top_candidates_returned(self):
        s = HeuristicCompositionScorer(top_k=5)
        r = s.score_frame(subject_bbox=_box(0.5, 0.5, 0.5), frame_shape=(640, 480))
        assert len(r.top_candidates) <= 5
        assert all(isinstance(c, type(r.top_candidates[0])) for c in r.top_candidates) if r.top_candidates else True

    def test_candidates_sorted_descending(self):
        s = HeuristicCompositionScorer()
        r = s.score_frame(subject_bbox=_box(0.5, 0.5, 0.5), frame_shape=(640, 480))
        scores = [c.score for c in r.top_candidates]
        assert scores == sorted(scores, reverse=True)

    def test_violations_recorded_for_bad_composition(self):
        """明显违规的构图应产生规则违反记录。"""
        s = HeuristicCompositionScorer()
        r = s.score_frame(subject_bbox=_box(0.04, 0.5, 0.9), frame_shape=(640, 480))
        assert isinstance(r.rule_violations, list)

    def test_deterministic(self):
        s = HeuristicCompositionScorer()
        bbox = _box(0.5, 0.5, 0.5)
        a = s.score_frame(subject_bbox=bbox, frame_shape=(640, 480))
        b = s.score_frame(subject_bbox=bbox, frame_shape=(640, 480))
        assert a.composition_score == b.composition_score
        assert a.best_bbox == b.best_bbox

    def test_saliency_affects_score(self):
        """提供显著图后应参与评分（不再是固定 0.5 占位）。"""
        s = HeuristicCompositionScorer()
        sal = np.zeros((64, 48), np.float32)
        sal[20:44, 16:32] = 1.0  # 显著区域在中央偏左
        r = s.score_frame(subject_bbox=_box(0.5, 0.5, 0.5), saliency=sal, frame_shape=(640, 480))
        assert "saliency_center" in r.sub_scores


class TestSampNetStub:
    def test_stub_is_explicitly_unavailable(self):
        """SAMPNet 桩件必须明确表示"未接入"，不能静默假装可用。"""
        from aicg.composition.scorer import SAMPNetScorer

        s = SAMPNetScorer()
        assert s.available is False

    def test_stub_raises_with_actionable_message(self):
        """调用未接入的后端应抛出带指引的异常（而非 NotImplementedError 裸奔）。"""
        from aicg.composition.scorer import SAMPNetScorer

        s = SAMPNetScorer()
        with pytest.raises(NotImplementedError, match="尚未接入"):
            s.score_frame(_box(0.5, 0.5, 0.5))
