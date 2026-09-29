"""夹具与感知后端配对测试（"双源分流"的契约锁定）。

对应文档：``tests/fixtures/README.md`` §双源分流
对应需求：FR-02、FR-06

**背景（实测约束，必须锁定以免回退）**：

YOLOv8n-seg 无法检出任何"代码画出来的人形"——纯椭圆、火柴人、
带渐变背景+噪声的写实化版本我们全部试过，都是 0 主体。因此：

- 合成素材（``walk_towards`` / ``handheld_jitter``）**只能**配 ``rule``；
- 真实素材（``real_photo_zoom``）**只能**配 ``yolo``。

若有人日后把后端配对弄错，防抖指标会退化成"0% 降幅"的假结果，
而且**不会报错**。本文件就是为了让这种错误立刻暴露。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aicg.perception import build_perception, infer_backend_for_source
from aicg.perception.detector import YoloPerception

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


class TestInferBackendForSource:
    """素材名 → 后端推断。"""

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("tests/fixtures/walk_towards.mp4", "rule"),
            ("tests/fixtures/handheld_jitter.mp4", "rule"),
            ("tests/fixtures/synthetic_portrait.jpg", "rule"),
            ("tests/fixtures/real_photo_zoom.mp4", "yolo"),
            (r"C:\some\path\walk_towards.mp4", "rule"),
        ],
    )
    def test_known_fixtures(self, source: str, expected: str):
        assert infer_backend_for_source(source) == expected

    def test_camera_source_has_no_hint(self):
        """摄像头源无推荐——真实拍摄当然要用真实模型。"""
        assert infer_backend_for_source("camera:0") is None

    def test_unknown_source_has_no_hint(self):
        """未知素材不硬猜，返回 None 交由配置决定。"""
        assert infer_backend_for_source("some/random/clip.mp4") is None

    def test_synthetic_and_real_never_share_backend(self):
        """合成与真实素材的后端推荐必须不同（分流的核心不变量）。"""
        syn = infer_backend_for_source("tests/fixtures/handheld_jitter.mp4")
        real = infer_backend_for_source("tests/fixtures/real_photo_zoom.mp4")
        assert syn != real


class TestRuleDetectsSynthetic:
    """规则后端必须能检出合成人形——否则防抖素材失效。"""

    @pytest.mark.skipif(
        not (FIXTURE_DIR / "handheld_jitter.mp4").exists(),
        reason="夹具未生成；先跑 python scripts/make_fixtures.py",
    )
    def test_rule_detects_handheld_jitter(self):
        import cv2

        cap = cv2.VideoCapture(str(FIXTURE_DIR / "handheld_jitter.mp4"))
        ok, frame = cap.read()
        cap.release()
        assert ok, "无法读取夹具首帧"

        p = build_perception("rule")
        r = p.infer(frame, 0, 0.0)
        assert not r.degraded, "规则后端应能检出合成人形"
        assert r.primary_subject is not None

    @pytest.mark.skipif(
        not (FIXTURE_DIR / "handheld_jitter.mp4").exists(),
        reason="夹具未生成",
    )
    def test_synthetic_fixture_has_real_motion(self):
        """夹具必须产生足够大的运动幅度，否则防抖无从体现。

        实测教训：第一版夹具让主体在理想位置附近小幅抖动，
        结果全程落在决策层的"静默区"内，防抖对照得到 0% 降幅——
        不是防抖无效，而是素材没制造出需要压制的抖动。

        这里锁定"主体确实在画面中大范围移动"这一前提。
        """
        import cv2

        from aicg.utils.image import bbox_center

        p = build_perception("rule")
        cap = cv2.VideoCapture(str(FIXTURE_DIR / "handheld_jitter.mp4"))
        xs, hs = [], []
        i = 0
        while i < 120:
            ok, frame = cap.read()
            if not ok:
                break
            r = p.infer(frame, i, i * 80.0)
            if r.primary_subject is not None:
                cx, _ = bbox_center(r.primary_subject.bbox)
                xs.append(cx)
                hs.append(r.primary_subject.height)
            i += 1
        cap.release()

        assert len(xs) > 50, "检出帧数过少，夹具可能失效"
        # 横向位移跨度应显著（夹具设计为在左右建议区之间摆动）
        assert max(xs) - min(xs) > 0.35, f"横向摆幅不足: {max(xs)-min(xs):.3f}"
        # 尺度跨度应跨过占比容差（0.08），能触发前进/后退建议
        assert max(hs) - min(hs) > 0.15, f"尺度摆幅不足: {max(hs)-min(hs):.3f}"


class TestYoloCannotDetectSynthetic:
    """把"YOLO 检不出合成人形"这一**已知限制**固化为可执行的事实。

    这不是我们希望的行为，而是模型能力的客观边界。写进测试的价值：

    1. 若某天换了更强的模型/更真的合成素材，本测试会失败 →
       提示我们"分流约束可以放松了"，是**正向**信号；
    2. 若有人误以为"合成素材配 yolo 也行"，本测试直接给出反例。
    """

    @staticmethod
    def _yolo_weights_available() -> bool:
        weights = Path(__file__).resolve().parents[1] / "models" / "yolov8n-seg.pt"
        return weights.exists()

    @pytest.mark.skipif(
        not (FIXTURE_DIR / "handheld_jitter.mp4").exists(),
        reason="夹具未生成",
    )
    def test_yolo_misses_synthetic_figure(self):
        if not self._yolo_weights_available():
            pytest.skip("YOLO 权重不存在")
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            pytest.skip("ultralytics 未安装")

        import cv2

        weights = Path(__file__).resolve().parents[1] / "models" / "yolov8n-seg.pt"
        p = YoloPerception(weights=str(weights), device="cpu")
        if not p.available:
            pytest.skip("YOLO 权重加载失败")

        cap = cv2.VideoCapture(str(FIXTURE_DIR / "handheld_jitter.mp4"))
        ok, frame = cap.read()
        cap.release()
        assert ok

        r = p.infer(frame, 0, 0.0)
        # 记录当前事实：合成人形检不出。
        # 若此断言失败，说明模型或素材已变强 —— 请顺手更新 fixtures/README.md。
        assert r.primary_subject is None, (
            "YOLO 竟然检出了合成人形！说明分流约束可以放松，"
            "请更新 tests/fixtures/README.md 与 scripts/make_fixtures.py 的说明。"
        )


class TestRealPhotoFixture:
    """真实照片素材必须能被 YOLO 检出——否则 yolo 轨也是假的。"""

    @pytest.mark.skipif(
        not (FIXTURE_DIR / "real_photo_zoom.mp4").exists(),
        reason="夹具未生成；先跑 python scripts/make_fixtures.py",
    )
    def test_yolo_detects_real_photo(self):
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            pytest.skip("ultralytics 未安装")

        import cv2

        weights = Path(__file__).resolve().parents[1] / "models" / "yolov8n-seg.pt"
        if not weights.exists():
            pytest.skip("YOLO 权重不存在")

        p = YoloPerception(weights=str(weights), device="cpu")
        if not p.available:
            pytest.skip("YOLO 权重加载失败")

        cap = cv2.VideoCapture(str(FIXTURE_DIR / "real_photo_zoom.mp4"))
        ok, frame = cap.read()
        cap.release()
        assert ok

        r = p.infer(frame, 0, 0.0)
        assert not r.degraded, "真实照片应能被检出"
        assert r.primary_subject is not None
        assert r.primary_subject.label == "person"
