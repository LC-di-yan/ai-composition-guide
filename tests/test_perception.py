"""感知层测试（FR-02）。

对应文档：《技术方案.md》§2.2 感知层、《测试与验收.md》§5 降级与鲁棒性

**测试策略**：感知层涉及重型模型（YOLO）与 GPU，直接测会拖慢 CI 且
不可复现。因此这里：

1. **规则后端（RulePerception）做真实推理**——纯 CPU、确定性、无权重依赖；
2. **YOLO 后端只测契约**——构造、惰性加载、降级、字段完整性；
3. **降级路径重点覆盖**——权重缺失时必须退回规则后端而不是崩。

这样既保证 CI 速度，又覆盖了关键行为。
"""

from __future__ import annotations

import numpy as np
import pytest

from aicg.perception import RulePerception, build_perception
from aicg.perception.base import BasePerception, SubjectSelector
from aicg.schemas import PerceptionResult, Subject, SubjectSource


# ---------------------------------------------------------------------------
# 规则后端（真实推理）
# ---------------------------------------------------------------------------
class TestRulePerception:
    def test_detects_synthetic_person(self, synthetic_portrait):
        """规则后端应能在合成人像上找到肤色主体。"""
        p = RulePerception()
        r = p.infer(synthetic_portrait, frame_id=0, timestamp_ms=0)

        assert isinstance(r, PerceptionResult)
        assert len(r.subjects) >= 1, "未检测到肤色区域"
        subj = r.primary_subject
        assert subj is not None
        assert 0.2 < subj.center[0] < 0.8

    def test_result_contract_fields(self, synthetic_portrait):
        p = RulePerception()
        r = p.infer(synthetic_portrait, frame_id=3, timestamp_ms=999)
        assert r.frame_id == 3
        assert r.timestamp_ms == 999
        assert r.frame_size == (synthetic_portrait.shape[1], synthetic_portrait.shape[0])
        assert r.backend == p.name
        assert r.perception_ms >= 0

    def test_bboxes_normalized(self, synthetic_portrait):
        """所有输出的 bbox 必须落在 0~1（契约硬要求）。"""
        p = RulePerception()
        r = p.infer(synthetic_portrait, frame_id=0, timestamp_ms=0)
        for s in r.subjects:
            x1, y1, x2, y2 = s.bbox
            assert 0.0 <= x1 < x2 <= 1.0, f"非法 bbox {s.bbox}"
            assert 0.0 <= y1 < y2 <= 1.0, f"非法 bbox {s.bbox}"

    def test_blank_image_returns_empty_not_crash(self):
        """纯色图应返回结果对象而非崩溃；若给出主体则必须是合法框。"""
        blank = np.full((240, 320, 3), 128, np.uint8)
        p = RulePerception()
        r = p.infer(blank, frame_id=0, timestamp_ms=0)
        assert isinstance(r, PerceptionResult)
        for s in r.subjects:
            assert 0.0 <= s.bbox[0] < s.bbox[2] <= 1.0

    def test_deterministic(self, synthetic_portrait):
        """同输入必须同输出（保证指标可复现，NFR-O3）。"""
        p = RulePerception()
        a = p.infer(synthetic_portrait, frame_id=0, timestamp_ms=0)
        b = p.infer(synthetic_portrait, frame_id=0, timestamp_ms=0)
        assert [s.bbox for s in a.subjects] == [s.bbox for s in b.subjects]

    def test_tiny_image_handled(self):
        """极小图像不应导致异常。"""
        tiny = np.full((8, 8, 3), 150, np.uint8)
        p = RulePerception()
        r = p.infer(tiny, frame_id=0, timestamp_ms=0)
        assert isinstance(r, PerceptionResult)

    def test_limitations_disclosed(self):
        """规则后端必须如实披露能力边界（诚实性要求）。

        局限清单会写入 ``PerceptionResult.extras['rule_backend_limitations']``，
        使下游（与使用者）能明确知道"这个结果来自规则启发式，而非学习模型"。
        """
        p = RulePerception()
        r = p.infer(np.full((120, 160, 3), 140, np.uint8), frame_id=0, timestamp_ms=0)
        limits = r.extras.get("rule_backend_limitations")
        assert limits, "规则后端未在结果中披露局限"
        assert len(limits) >= 3

    def test_model_versions_reported(self):
        """必须上报模型版本，保证结果可复现（NFR-O3）。"""
        p = RulePerception()
        mv = p.model_versions()
        assert mv
        assert all(isinstance(v, str) for v in mv.values())


# ---------------------------------------------------------------------------
# 主体选择器（返回索引，非 Subject）
# ---------------------------------------------------------------------------
class TestSubjectSelector:
    def test_prefers_person(self):
        """人优先：即使非人候选面积更大，也应选 person。

        这是产品决策而非模型能力——人像摄影的主体语义就是"人"。
        """
        sel = SubjectSelector(prefer_person=True)
        cands = [
            ("dog", (0.1, 0.1, 0.9, 0.9), 0.9),     # 面积大但是狗
            ("person", (0.3, 0.2, 0.5, 0.6), 0.6),  # 面积小但是人
        ]
        assert sel.select(cands) == 1

    def test_prefers_larger_area_among_persons(self):
        """同为 person 时选面积更大的。"""
        sel = SubjectSelector()
        cands = [
            ("person", (0.1, 0.1, 0.2, 0.2), 0.8),
            ("person", (0.1, 0.1, 0.8, 0.8), 0.8),
        ]
        assert sel.select(cands) == 1

    def test_empty_returns_minus_one(self):
        """空候选返回 -1（而非抛异常）。"""
        assert SubjectSelector().select([]) == -1

    def test_no_person_preference_disabled(self):
        """prefer_person=False 时按面积决策。"""
        sel = SubjectSelector(prefer_person=False)
        cands = [
            ("person", (0.4, 0.4, 0.6, 0.6), 0.9),
            ("dog", (0.1, 0.1, 0.9, 0.9), 0.5),
        ]
        assert sel.select(cands) == 1

    def test_confidence_breaks_ties(self):
        """面积相同时用置信度决胜。"""
        sel = SubjectSelector()
        cands = [
            ("person", (0.2, 0.2, 0.6, 0.6), 0.5),
            ("person", (0.2, 0.2, 0.6, 0.6), 0.9),
        ]
        assert sel.select(cands) == 1


# ---------------------------------------------------------------------------
# 工厂与降级
# ---------------------------------------------------------------------------
def _ultralytics_available() -> bool:
    """ultralytics 是否可导入。

    背景：CI（.github/workflows/ci.yml）刻意不装 torch/ultralytics，
    以验证"无模型权重时链路依然可用"的设计承诺。以下两个测试断言的
    是 **yolo 真实加载行为**，在无 ultralytics 的环境下必然失败
    （用 `sitecustomize` 屏蔽重依赖模拟 CI 实测抓出），
    因此必须与"权重存在"一样作为跳过条件。
    """
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


_requires_yolo = pytest.mark.skipif(
    not _ultralytics_available(),
    reason="ultralytics 未安装（CI 裸环境，yolo 真实加载行为无法验证）",
)


class TestBuildPerception:
    def test_explicit_rule_backend(self):
        p = build_perception("rule")
        assert p.name == "rule"
        assert isinstance(p, BasePerception)

    def test_explicit_yolo_backend_constructs(self, project_root):
        """显式请求 yolo 时应返回 YoloPerception（即使权重缺失也不抛异常）。"""
        p = build_perception("yolo", weights="models/yolov8n-seg.pt", device="cpu")
        assert p.name == "yolo"

    def test_auto_returns_usable_object(self):
        """auto 必须**保证返回可用对象**（NFR-R1）。"""
        p = build_perception("auto")
        assert isinstance(p, BasePerception)
        assert p.name in ("yolo", "rule")

    def test_auto_falls_back_when_weights_bad(self, tmp_path):
        """权重路径无效时 auto 必须退回 rule（降级而非崩溃）。"""
        p = build_perception("auto", weights=str(tmp_path / "definitely_missing.pt"))
        assert p.name == "rule"

    @_requires_yolo
    def test_auto_uses_yolo_when_weights_present(self, project_root):
        """权重就绪时 auto 应选 yolo。"""
        weights = project_root / "models" / "yolov8n-seg.pt"
        if not weights.exists():
            pytest.skip("YOLO 权重未就绪")
        p = build_perception("auto", weights=str(weights))
        assert p.name == "yolo"

    def test_kwargs_filtered_per_class(self):
        """不支持的参数应被过滤而非引发 TypeError。"""
        p = build_perception("rule", saliency_work_size=120, unsupported_param=999)
        assert p.name == "rule"


# ---------------------------------------------------------------------------
# YOLO 后端契约（不强制加载权重）
# ---------------------------------------------------------------------------
class TestYoloContract:
    @_requires_yolo
    def test_eager_loading_at_construction(self, project_root):
        """权重加载必须在**构造期**完成（急于加载）。

        理由：把约 83ms 的加载成本放在启动阶段，避免首帧延迟尖刺污染
        P95 指标（NFR-P1 要求稳定，而非"平均达标"）。
        """
        from aicg.perception.detector import YoloPerception

        weights = project_root / "models" / "yolov8n-seg.pt"
        if not weights.exists():
            pytest.skip("YOLO 权重未就绪")
        p = YoloPerception(weights=str(weights), device="cpu")
        assert p.available is True, "构造后模型应已就绪"
        assert p._model is not None

    def test_missing_weights_degrades(self, tmp_path):
        """权重路径无效时 infer 应返回降级结果，而非抛异常。"""
        from aicg.perception.detector import YoloPerception

        p = YoloPerception(weights=str(tmp_path / "no.pt"), device="cpu")
        img = np.full((240, 320, 3), 100, np.uint8)
        r = p.infer(img, frame_id=0, timestamp_ms=0)

        assert isinstance(r, PerceptionResult)
        assert r.degraded is True
        assert r.backend == "yolo"

    def test_available_false_when_missing(self, tmp_path):
        from aicg.perception.detector import YoloPerception

        p = YoloPerception(weights=str(tmp_path / "no.pt"), device="cpu")
        assert p.available is False

    def test_infer_never_raises_on_weird_input(self, project_root):
        """异常输入不得让感知层抛异常（实时回路的硬要求）。"""
        from aicg.perception.detector import YoloPerception

        weights = project_root / "models" / "yolov8n-seg.pt"
        if not weights.exists():
            pytest.skip("YOLO 权重未就绪")
        p = YoloPerception(weights=str(weights), device="cpu")
        weird = np.zeros((1, 1, 3), np.uint8)
        r = p.infer(weird, frame_id=0, timestamp_ms=0)
        assert isinstance(r, PerceptionResult)


# ---------------------------------------------------------------------------
# 设备解析
# ---------------------------------------------------------------------------
class TestDeviceResolution:
    def test_auto_resolves_to_concrete_device(self):
        """device=auto 必须解析为具体设备名，不能把 "auto" 传给 torch。"""
        from aicg.perception.detector import YoloPerception

        p = YoloPerception(weights="x.pt", device="auto")
        dev = p._resolve_device("auto")
        assert dev.startswith("cuda") or dev == "cpu", f"未解析为具体设备: {dev}"

    def test_explicit_cpu_respected(self):
        from aicg.perception.detector import YoloPerception

        p = YoloPerception(weights="x.pt", device="cpu")
        assert p._resolve_device("cpu") == "cpu"

    def test_explicit_cuda_respected(self):
        from aicg.perception.detector import YoloPerception

        p = YoloPerception(weights="x.pt", device="cuda")
        assert p._resolve_device("cuda").startswith("cuda")
