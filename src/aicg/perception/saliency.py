"""规则版感知后端（无模型依赖的兜底实现）。

对应需求：FR-02（主体确认）、NFR-R2（降级路径）
对应文档：《开发计划.md》§6 诚实简化处清单

本后端的定位必须说清楚：

**它不是"占位桩"，而是一条真实可用的降级路径。** 当 YOLO 权重不可用时，
引导引擎仍需给出可用的主体框，否则整条 FR-05 链路断裂。算法选择：

1. **显著性**（频谱残差，纯 FFT）→ 定位画面视觉重点区域；
2. **肤色 + 中心先验**（YCrCb 阈值）→ 近似人体/人脸区域；
3. **形态学清理 + 连通域** → 得到稳定的候选框。

诚实标注的局限：
- 无法区分语义类别（一律标为 ``person``，若肤色命中；否则 ``object``）；
- 多人重叠时倾向于合并为一个框；
- 逆光/复杂背景误检率明显高于 YOLO。

以上限制会写入 ``PerceptionResult.extras["rule_backend_limitations"]``，
供上层与报告如实披露。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..observability import get_logger
from ..schemas import PerceptionResult, Subject, SubjectSource
from ..utils.image import bbox_px_to_norm, spectral_residual_saliency, to_gray
from .base import BasePerception, SubjectSelector

log = get_logger("perception.rule")

# 规则后端已知局限，随结果一并输出（诚实原则）
_RULE_LIMITATIONS = [
    "无语义分类能力，主体类别为启发式推断",
    "多人重叠时倾向合并为单一框",
    "逆光/复杂背景误检率高于 YOLO",
]


class RulePerception(BasePerception):
    """基于经典计算机视觉的无依赖感知后端。"""

    name = "rule"

    def __init__(
        self,
        saliency_work_size: int = 160,
        prefer_person: bool = True,
        min_area_ratio: float = 0.02,
        max_area_ratio: float = 0.90,
    ) -> None:
        """
        Args:
            saliency_work_size: 显著性计算尺寸，越小越快。
            prefer_person: 是否优先选择肤色命中的区域为主体。
            min_area_ratio: 候选框最小面积占比，低于此值的连通域被丢弃（滤噪点）。
            max_area_ratio: 候选框最大面积占比，高于此值视为误检整幅背景。
        """
        self.saliency_work_size = saliency_work_size
        self.selector = SubjectSelector(prefer_person=prefer_person)
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def model_versions(self) -> dict[str, str]:
        return {"rule.algo": "saliency+skin+cc", "opencv": cv2.__version__}

    def infer(self, image: np.ndarray, frame_id: int, timestamp_ms: int) -> PerceptionResult:
        h, w = image.shape[:2]
        t0 = time.perf_counter()

        result = PerceptionResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            frame_size=(w, h),
            backend=self.name,
            model_versions=self.model_versions(),
        )

        try:
            boxes, sal_ref, skin_mask = self._detect(image)
        except Exception as e:  # noqa: BLE001 - 兜底后端更不允许抛异常
            log.warning("规则后端异常（帧 %d）: %s", frame_id, e)
            result.degraded = True
            result.extras["error"] = f"{type(e).__name__}: {e}"
            result.perception_ms = (time.perf_counter() - t0) * 1000.0
            return result

        candidates: list[tuple[str, tuple[float, float, float, float], float]] = []
        for bbox, conf, is_person in boxes:
            label = "person" if is_person else "object"
            candidates.append((label, bbox, conf))

        if candidates:
            primary_idx = self.selector.select(candidates)
            result.subjects = [
                Subject(
                    subject_id=f"r{i}",
                    label=label,
                    bbox=bbox,
                    confidence=conf,
                    is_primary=(i == primary_idx),
                    source=SubjectSource.AUTO,
                )
                for i, (label, bbox, conf) in enumerate(candidates)
            ]
        else:
            result.degraded = True
            result.extras["no_subject"] = True

        result.saliency_map_ref = sal_ref
        result.extras["rule_backend_limitations"] = list(_RULE_LIMITATIONS)
        result.extras["saliency_peak"] = sal_ref
        result.extras["skin_ratio"] = (
            float(np.count_nonzero(skin_mask)) / float(skin_mask.size)
            if skin_mask is not None
            else 0.0
        )
        result.perception_ms = (time.perf_counter() - t0) * 1000.0
        return result

    # ------------------------------------------------------------------
    # 内部算法
    # ------------------------------------------------------------------
    def _detect(
        self, image: np.ndarray
    ) -> tuple[list[tuple[tuple[float, float, float, float], float, bool]], str | None, np.ndarray | None]:
        """返回 ``[(bbox_norm, confidence, is_person)]``、显著性引用与肤色掩码。"""
        h, w = image.shape[:2]

        # --- 1. 肤色掩码：YCrCb 空间阈值 + RGB 三重门限 ---
        # 只用 YCrCb 会把大量暖色背景误判为肤色（实测背景梯度色亦落入
        # 宽阈值区间），故叠加 BGR 通道的相对关系门限：
        #   肤色满足 R > G > B，且 R-B 落在合理区间。
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        skin_ycc = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 130))

        b, g, r = cv2.split(image.astype(np.int16))
        skin_rgb = (
            (r > g) & (g > b) & ((r - b) > 12) & ((r - b) < 90) & (r > 60) & (r < 250)
        )
        skin = cv2.bitwise_and(skin_ycc, (skin_rgb.astype(np.uint8) * 255))

        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        # 注意：uint8 掩码求 .mean() 会溢出得到 255，必须用 count_nonzero
        skin_ratio = float(np.count_nonzero(skin)) / float(skin.size)

        # --- 2. 显著性图 ---
        gray = to_gray(image)
        saliency = spectral_residual_saliency(gray, self.saliency_work_size)
        sal_u8 = np.clip(saliency * 255.0, 0, 255).astype(np.uint8)
        _, sal_mask = cv2.threshold(sal_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        boxes: list[tuple[tuple[float, float, float, float], float, bool]] = []

        # --- 3. 肤色区域优先作为主体（近似人体） ---
        if skin_ratio > 0.005:
            boxes.extend(self._mask_to_boxes(skin, (w, h), source="skin"))

        # --- 4. 显著性区域补位（无肤色时给出"视觉重点"框） ---
        if not boxes:
            boxes.extend(self._mask_to_boxes(sal_mask, (w, h), source="saliency"))

        # --- 5. 仍为空时退化为画面中心固定框，保证链路不断（NFR-R2） ---
        if not boxes:
            boxes.append(((0.28, 0.18, 0.72, 0.92), 0.20, False))

        sal_ref = f"inline:saliency:{frame_id_of(saliency)}"
        return boxes, sal_ref, skin

    def _mask_to_boxes(
        self, mask: np.ndarray, size: tuple[int, int], source: str
    ) -> list[tuple[tuple[float, float, float, float], float, bool]]:
        """把二值掩码转为主体框列表。"""
        w, h = size
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        total_area = float(w * h)
        scored: list[tuple[float, tuple[int, int, int, int]]] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area_ratio = (bw * bh) / total_area
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue
            if bw < 8 or bh < 8:
                continue
            scored.append((area_ratio, (x, y, bw, bh)))

        if not scored:
            return []

        # 只保留最显著的前 2 个区域，避免碎片框干扰构图判断
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for area_ratio, (x, y, bw, bh) in scored[:2]:
            bbox = bbox_px_to_norm((x, y, x + bw, y + bh), size)
            # 置信度用面积占比做代理：占画面越大越可能是主体
            conf = float(np.clip(0.35 + area_ratio * 0.6, 0.0, 0.95))
            is_person = source == "skin"
            out.append((bbox, conf, is_person))
        return out


def frame_id_of(arr: np.ndarray) -> str:
    """为数组生成一个稳定的短标识，用作内联引用名（不存大数组）。"""
    import hashlib

    return hashlib.md5(arr.tobytes()).hexdigest()[:8]
