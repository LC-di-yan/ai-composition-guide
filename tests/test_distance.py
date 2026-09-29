"""距离估算测试（FR-01）。

对应文档：《测试与验收.md》§4 量化验收

**本文件的重点是量纲正确性**——这是踩过的最严重的一个坑：
早期实现把"等效焦距（26mm，相对 35mm 全画幅）"与"物理传感器高度
（6~8mm）"混用，得到 0.00m（错约 1000 倍）。因此这里用一个已知
锚点（调研文档给出的"人体占画面 70% 高度约在 2m"）做硬校验。
"""

from __future__ import annotations

import pytest

from aicg.composition.distance import DistanceEstimator
from aicg.schemas import EstimationMethod
from aicg.settings import DistanceConfig


@pytest.fixture
def est() -> DistanceEstimator:
    return DistanceEstimator()


def _box_of_height(h: float, cx: float = 0.5, cy: float = 0.5):
    w = h * 0.42
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class TestMagnitude:
    def test_no_1000x_error(self, est):
        """核心回归：占高 0.90 必须得到米级结果，不是 0.00m。"""
        r = est.estimate(_box_of_height(0.90))
        assert r.current_distance_m is not None
        assert 1.0 < r.current_distance_m < 3.5, f"实际 {r.current_distance_m}m"

    def test_anchor_matches_research_doc(self, est):
        """调研锚点：占画面 70% 高度 ≈ 2m 左右（允许 ±40% 工程容差）。"""
        r = est.estimate(_box_of_height(0.70))
        assert 1.2 < r.current_distance_m < 2.8, f"实际 {r.current_distance_m}m"

    def test_monotonic(self, est):
        """占比越大 → 距离越近（严格单调）。"""
        ds = [est.estimate(_box_of_height(h)).current_distance_m for h in (0.3, 0.5, 0.7, 0.9)]
        assert all(d is not None for d in ds)
        assert ds == sorted(ds, reverse=True), f"非单调: {ds}"

    def test_reasonable_range(self, est):
        """可信任区间内应落在物理可信范围。

        注意：占高 < 0.15 的极端值会**主动降级**（返回 None），
        这是设计行为而非缺陷——见 :class:`TestDegradation`。
        """
        for h in (0.2, 0.35, 0.5, 0.75, 0.95, 0.99):
            r = est.estimate(_box_of_height(h))
            assert r.current_distance_m is not None, f"h={h} 不应降级"
            assert 1.5 < r.current_distance_m < 15.0, f"h={h} → {r.current_distance_m}m"

    def test_floor_is_documented_and_enforced(self, est):
        """公式硬下界必须被显式暴露，且与配置的 min_distance_m 自洽。

        回归点：早期 ``min_distance_m = 1.2`` 小于公式硬下界 1.79m，
        导致"距离太近"分支成为死代码（永远不可达）。
        """
        floor = est.cfg.formula_floor_m()
        assert 1.5 < floor < 2.2, f"硬下界 {floor:.2f}m 超出预期"
        assert est.cfg.min_distance_m > floor, (
            f"min_distance_m({est.cfg.min_distance_m}) 必须大于公式硬下界({floor:.2f})"
        )


class TestDegradation:
    def test_no_subject_degrades(self, est):
        r = est.estimate(None)
        assert r.degraded is True
        assert r.current_distance_m is None
        assert r.advice_text

    def test_tiny_box_degrades(self, est):
        """主体过小（框高 < 2%）→ 降级，仅给方向不给距离。"""
        r = est.estimate(_box_of_height(0.01))
        assert r.degraded is True
        assert r.current_distance_m is None

    def test_zero_height_degrades(self, est):
        r = est.estimate((0.4, 0.5, 0.6, 0.5))
        assert r.degraded is True

    def test_too_small_box_degrades_not_fabricates(self, est):
        """框高占比过小时**主动降级**，而不是外推出荒谬的远距离。

        回归点：早期实现会输出 38m（占比 0.05），这是假精度。
        """
        for h in (0.03, 0.08, 0.14):
            r = est.estimate(_box_of_height(h))
            assert r.degraded is True, f"h={h} 应降级"
            assert r.current_distance_m is None, f"h={h} 不应给出距离"


class TestAdvice:
    def test_in_range_flag(self, est):
        """落在建议区间内时 is_in_range=True。"""
        cfg = est.cfg
        h = 0.70  # 约 2m，落在 1.2~4.5 内
        r = est.estimate(_box_of_height(h))
        assert r.is_in_range is True
        assert "距离合适" in r.advice_text

    def test_too_close_advice(self, est):
        """占高接近 1.0（约 1.79~1.85m）应触发"太近"提示。

        由于公式硬下界的存在，"太近"只在极端占比下出现——这本身就是
        一条重要的产品事实：本实现几乎不会说"你太近了"。
        """
        r = est.estimate(_box_of_height(0.995))
        assert r.current_distance_m is not None
        assert r.is_in_range is False
        assert r.current_distance_m < est.cfg.min_distance_m
        assert "太近" in r.advice_text or "后退" in r.advice_text

    def test_too_far_advice(self, est):
        r = est.estimate(_box_of_height(0.18))
        assert r.current_distance_m is not None
        assert r.is_in_range is False
        assert "偏远" in r.advice_text or "靠近" in r.advice_text

    def test_position_ratio_in_unit_interval(self, est):
        for h in (0.15, 0.4, 0.7, 0.95):
            r = est.estimate(_box_of_height(h))
            assert 0.0 <= r.position_ratio <= 1.0


class TestContract:
    def test_error_margin_reported(self, est):
        """必须如实上报误差范围（这是"估算非测量"的产品约束）。"""
        r = est.estimate(_box_of_height(0.7))
        assert r.error_margin is not None and r.error_margin > 0

    def test_estimation_method_labeled(self, est):
        r = est.estimate(_box_of_height(0.7))
        assert r.estimation_method is EstimationMethod.BBOX_HEIGHT_RATIO

    def test_min_lt_max(self, est):
        r = est.estimate(_box_of_height(0.7))
        assert r.max_distance_m > r.min_distance_m

    def test_config_validation_rejects_inverted_range(self):
        """配置层应拒绝 min >= max 的非法区间。"""
        with pytest.raises(Exception):
            DistanceConfig(min_distance_m=3.0, max_distance_m=1.0)

    def test_config_validation_rejects_ideal_outside_range(self):
        """理想值必须落在区间内，否则 position_ratio 语义失效。"""
        with pytest.raises(Exception):
            DistanceConfig(min_distance_m=2.0, max_distance_m=5.0, ideal_distance_m=8.0)
        with pytest.raises(Exception):
            DistanceConfig(min_distance_m=2.0, max_distance_m=5.0, ideal_distance_m=1.0)

    def test_custom_height_affects_distance(self):
        """主体真实身高假设越大 → 同框高下估距越远（线性关系）。"""
        e = DistanceEstimator()
        box = _box_of_height(0.7)
        d_short = e.estimate(box, assumed_height_m=1.2).current_distance_m
        d_tall = e.estimate(box, assumed_height_m=1.9).current_distance_m
        assert d_tall > d_short
