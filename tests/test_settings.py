"""配置层测试（含两个真实 bug 的回归锁定）。

对应需求：NFR-M2（配置外置）
对应文档：《技术方案.md》§3、《测试与验收.md》§4

本文件重点覆盖**配置覆盖机制**——这是本项目踩过最隐蔽的一个坑：
扁平点号键被静默丢弃，导致 ``--backend rule`` 看着生效实际无效。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aicg.settings import (
    DEFAULT_CONFIG_PATH,
    DistanceConfig,
    Settings,
    load_settings,
)


class TestLoadSettingsOverride:
    """``load_settings(overrides=...)`` 的覆盖语义。"""

    def test_nested_dict_override(self):
        """嵌套字典形式应生效。"""
        cfg = load_settings(overrides={"perception": {"backend": "rule"}})
        assert cfg.perception.backend == "rule"

    def test_flat_dotted_key_override(self):
        """扁平点号键形式应生效（回归锁定）。

        早期实现直接把 ``"perception.backend"`` 当字面顶层键交给
        ``_deep_merge``，pydantic 默认忽略未知字段 → 覆盖被静默丢弃。
        症状：``--backend rule`` 无效但无任何报错。
        """
        cfg = load_settings(overrides={"perception.backend": "rule"})
        assert cfg.perception.backend == "rule"

    def test_flat_and_nested_equivalent(self):
        """两种写法必须等价。"""
        flat = load_settings(
            overrides={"perception.backend": "rule", "pipeline.target_fps": 5.0}
        )
        nested = load_settings(
            overrides={"perception": {"backend": "rule"}, "pipeline": {"target_fps": 5.0}}
        )
        assert flat.perception.backend == nested.perception.backend == "rule"
        assert flat.pipeline.target_fps == nested.pipeline.target_fps == 5.0

    def test_flat_key_does_not_leak_toplevel(self):
        """点号键不得在顶层留下垃圾键（否则是静默丢弃的征兆）。"""
        cfg = load_settings(overrides={"pipeline.target_fps": 7.5})
        dumped = cfg.model_dump()
        assert "pipeline.target_fps" not in dumped
        assert cfg.pipeline.target_fps == 7.5

    def test_deep_nested_dotted_key(self):
        """三层以上的点号键也要能正确展开。"""
        cfg = load_settings(
            overrides={"composition.distance.min_distance_m": 2.5}
        )
        assert cfg.composition.distance.min_distance_m == 2.5

    def test_override_does_not_mutate_yaml_source(self):
        """覆盖不得污染磁盘上的 yaml 文件。"""
        before = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        load_settings(overrides={"perception.backend": "rule"})
        after = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        assert before == after

    def test_default_backend_is_auto(self):
        """无覆盖时，配置里的值应原样生效。"""
        cfg = load_settings()
        assert cfg.perception.backend == "auto"


class TestEffectiveBackend:
    """后端解析逻辑。"""

    def test_explicit_rule_ignores_weights(self):
        """显式 rule 时即使权重存在也应返回 rule（覆盖优先）。"""
        cfg = load_settings(overrides={"perception.backend": "rule"})
        assert cfg.perception.effective_backend() == "rule"

    def test_explicit_yolo(self):
        cfg = load_settings(overrides={"perception.backend": "yolo"})
        assert cfg.perception.effective_backend() == "yolo"

    def test_auto_resolves_by_weight_existence(self):
        """auto 按权重文件是否存在解析。"""
        cfg = load_settings(overrides={"perception.backend": "auto"})
        weights = cfg.perception.detector.resolved_weights()
        expected = "yolo" if weights.exists() else "rule"
        assert cfg.perception.effective_backend() == expected


class TestDistanceConfigValidation:
    """距离配置的跨字段校验（含一处设计缺陷的回归锁定）。"""

    def test_rejects_inverted_range(self):
        """min >= max 必须在加载时失败，而不是运行时静默失效。"""
        with pytest.raises(ValidationError):
            DistanceConfig(min_distance_m=5.0, max_distance_m=2.0)

    def test_rejects_equal_range(self):
        with pytest.raises(ValidationError):
            DistanceConfig(min_distance_m=3.0, max_distance_m=3.0)

    def test_rejects_ideal_outside_range(self):
        with pytest.raises(ValidationError):
            DistanceConfig(
                min_distance_m=2.0, max_distance_m=5.0, ideal_distance_m=8.0
            )

    def test_accepts_valid_range(self):
        c = DistanceConfig(min_distance_m=2.0, max_distance_m=5.0, ideal_distance_m=2.6)
        assert c.ideal_distance_m == 2.6

    def test_formula_floor_matches_documented_value(self):
        """公式硬下界应与文档标注的 ~1.79m 一致。

        这个下界决定了 ``min_distance_m`` 的下限——早期设 1.2m
        使"距离太近"分支永不可达（静默死代码）。
        """
        floor = DistanceConfig().formula_floor_m()
        assert 1.7 < floor < 1.9

    def test_default_min_above_formula_floor(self):
        """默认配置必须满足 min > 物理下界，否则该校验没意义。"""
        c = DistanceConfig()
        assert c.min_distance_m > c.formula_floor_m()


class TestWithOverrides:
    """``Settings.with_overrides`` 与 ``load_settings`` 的语义一致性。"""

    def test_with_overrides_dotted_path(self):
        cfg = load_settings()
        new = cfg.with_overrides(**{"perception.backend": "rule"})
        assert new.perception.backend == "rule"
        # 原对象不应被修改
        assert cfg.perception.backend == "auto"

    def test_with_overrides_returns_new_instance(self):
        cfg = load_settings()
        new = cfg.with_overrides(**{"pipeline.target_fps": 9.0})
        assert new is not cfg
        assert new.pipeline.target_fps == 9.0
        assert cfg.pipeline.target_fps != 9.0


class TestSummary:
    """配置摘要用于复现与排障，字段必须真实反映当前配置。"""

    def test_summary_reflects_backend(self):
        cfg = load_settings(overrides={"perception.backend": "rule"})
        assert "perception=rule" in cfg.summary()

    def test_summary_reflects_debounce(self):
        cfg = load_settings()
        s = cfg.summary()
        assert "debounce=True" in s
        assert "3frames" in s


class TestSettingsModel:
    def test_project_root_injected(self):
        cfg = Settings()
        assert cfg.project_root.exists()

    def test_pipeline_fps_bounds(self):
        with pytest.raises(ValidationError):
            Settings(**{"pipeline": {"target_fps": 0}})
        with pytest.raises(ValidationError):
            Settings(**{"pipeline": {"target_fps": 100}})
