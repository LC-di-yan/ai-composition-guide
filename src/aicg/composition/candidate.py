"""候选画框搜索。

对应需求：FR-03（构图评估）
对应文档：《技术方案.md》§1 构图决策层 —— "候选框搜索（sliding window / grid anchor）"

思路来源：调研指出 Doka 的"构图评分模型"本质是
"把画面裁成多个候选框打分选最优"。本模块负责**生成候选**，
打分由 :mod:`aicg.composition.scorer` 完成。

候选生成策略（subject-anchored grid）：

1. **尺度维度**：按"主体应占候选框高度的比例"（occupancy）枚举若干倍率，
   覆盖从更紧到更松的景别；
2. **位置维度**：保持候选框尺寸不变，把主体框在候选框内部平移，
   平移范围由"候选框比主体框大出多少"决定；
3. **显式锚点**：额外构造"主体落在三分点/画面中心"的候选，避免网格
   采样错过构图学上公认的最优位置。

之所以用主体锚点而非全画面穷举网格：候选数量可控（护住时延 NFR-P1），
且生成的候选在语义上都合理（主体必然完整在框内）。

**诚实标注**：候选框宽高比统一取自当前画面，因此不生成"改变画幅比例"
的候选（如从 16:9 改成 4:3）。这类建议在手机上通常意味着用户需要
改变持机方向，属于产品层决策而非算法层输出。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.perception import BBox
from ..utils.image import bbox_center, bbox_height, bbox_width, clamp_bbox

# 主体在候选框中的目标占比（occupancy）。
# 覆盖区间说明：0.40 意味着"更松的景别"（主体更小、环境更多），
# 0.92 意味着"更紧的景别"（主体更大）。全区间覆盖是为了在主体本就
# 很大时仍能给出"再退一点会更好"的候选。
_OCCUPANCY_TARGETS: tuple[float, ...] = (0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.92)


@dataclass(frozen=True)
class Candidate:
    """一个候选画框。"""

    bbox: BBox
    """归一化画框。"""

    origin: str
    """生成来源，便于追溯：``center`` / ``thirds`` / ``scale``。"""

    occupancy: float = 0.0
    """主体在该候选框内的高度占比，用于诊断与解说。"""


class CandidateGenerator:
    """基于主体锚点的候选框生成器。"""

    def __init__(
        self,
        min_area_ratio: float = 0.10,
        max_area_ratio: float = 0.99,
        grid_x: int = 9,
        grid_y: int = 7,
        min_aspect: float = 0.55,
        max_aspect: float = 1.80,
        max_candidates: int = 240,
    ) -> None:
        """
        Args:
            min_area_ratio: 候选框面积占原图的最小比例。默认 0.10 对应
                "主体很小、大量环境"的宽景；低于此值主体会太小而失去引导意义。
            max_area_ratio: 最大比例，接近 1.0 即接近整幅画面。
            grid_x / grid_y: 主体在候选框内的平移采样密度。
            min_aspect / max_aspect: 候选框宽高比范围（宽/高），需容纳横竖幅。
            max_candidates: 候选数量上限，超出则按"尺度接近度"裁剪（护住时延）。
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.max_candidates = max_candidates

    # ------------------------------------------------------------------
    def generate(
        self,
        subject_bbox: BBox | None,
        frame_aspect: float = 0.75,
    ) -> list[Candidate]:
        """生成候选画框。

        Args:
            subject_bbox: 主体框；为 None 时退化为"以画面中心为锚点"的网格。
            frame_aspect: 画幅宽高比（宽/高）。竖幅人像约 0.75，横向约 1.33。
                候选框严格保持该比例，因为改变画幅比例不是本层该给的建议。

        Returns:
            候选列表（已去重、已裁剪到画面内）。**保证非空**：无可用候选时
            返回整幅画面作为兜底，避免上层出现空值分支。
        """
        if frame_aspect <= 0:
            frame_aspect = 0.75

        if subject_bbox is None:
            out = self._center_grid(frame_aspect)
        else:
            out = self._anchored_candidates(subject_bbox, frame_aspect)
            out.extend(self._explicit_anchors(subject_bbox, frame_aspect))

        out = self._dedup_and_limit(out)
        if not out and subject_bbox is not None:
            # 无可用候选的**唯一真实成因**：主体本身已超出画面比例能容纳的范围
            # （典型场景：横幅画面里主体占满高度 0.77，而横幅最大可用高度仅
            # 0.75）。此时"重新取景"在物理上不可行，唯一正确的建议是
            # **后退拉开距离**，而不是伪造一个整幅画面当候选。
            # 因此这里返回"维持现状 + 记为收紧"的候选，由上层 differ 判定
            # 需要后退；occupancy 记为 1.0 以标记"主体已贴满"。
            out = [
                Candidate(
                    bbox=(0.0, 0.0, 1.0, 1.0),
                    origin="subject_overflow",
                    occupancy=1.0,
                )
            ]
        elif not out:
            out = [Candidate(bbox=(0.0, 0.0, 1.0, 1.0), origin="fallback", occupancy=0.0)]
        return out

    # ------------------------------------------------------------------
    def _anchored_candidates(self, subject_bbox: BBox, frame_aspect: float) -> list[Candidate]:
        """核心：尺度枚举 × 主体平移采样。"""
        out: list[Candidate] = []
        su_w = bbox_width(subject_bbox)
        su_h = bbox_height(subject_bbox)
        su_cx, su_cy = bbox_center(subject_bbox)
        if su_h <= 1e-6 or su_w <= 1e-6:
            return out

        for occupancy in _OCCUPANCY_TARGETS:
            # 由目标占比反推候选框尺寸
            cand_h = su_h / occupancy
            cand_w = cand_h * frame_aspect

            # 尺寸超出画面：按受限维度等比收缩（保证宽高比不变）
            if cand_h > 1.0:
                cand_h = 1.0
                cand_w = cand_h * frame_aspect
            if cand_w > 1.0:
                cand_w = 1.0
                cand_h = cand_w / frame_aspect

            # 收缩后主体可能仍放不下（主体本身就很宽/很高）
            if cand_h < su_h - 1e-6 or cand_w < su_w - 1e-6:
                continue

            area = cand_w * cand_h
            if not (self.min_area_ratio <= area <= self.max_area_ratio):
                continue
            aspect = cand_w / cand_h
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue

            # 主体中心在候选框内可平移的范围
            max_dx = max(0.0, (cand_w - su_w) / 2.0)
            max_dy = max(0.0, (cand_h - su_h) / 2.0)

            for ix in range(self.grid_x):
                fx = (ix / max(1, self.grid_x - 1)) * 2.0 - 1.0 if self.grid_x > 1 else 0.0
                for iy in range(self.grid_y):
                    fy = (iy / max(1, self.grid_y - 1)) * 2.0 - 1.0 if self.grid_y > 1 else 0.0
                    cx = su_cx + fx * max_dx
                    cy = su_cy + fy * max_dy

                    bbox = clamp_bbox(
                        (cx - cand_w / 2, cy - cand_h / 2, cx + cand_w / 2, cy + cand_h / 2)
                    )
                    if not _contains(bbox, subject_bbox, margin=1e-3):
                        continue
                    out.append(
                        Candidate(bbox=bbox, origin="scale", occupancy=su_h / max(1e-6, cand_h))
                    )
        return out

    def _explicit_anchors(self, subject_bbox: BBox, frame_aspect: float) -> list[Candidate]:
        """显式构造"主体落在三分点/画面中心"的候选。

        这些是构图学上公认的重点位置，直接构造可避免网格采样恰好错过。
        """
        out: list[Candidate] = []
        su_w = bbox_width(subject_bbox)
        su_h = bbox_height(subject_bbox)
        anchors = (
            (1.0 / 3.0, 1.0 / 3.0, "thirds"),
            (2.0 / 3.0, 1.0 / 3.0, "thirds"),
            (1.0 / 3.0, 0.45, "thirds"),
            (2.0 / 3.0, 0.45, "thirds"),
            (0.5, 0.42, "center"),
            (0.5, 0.5, "center"),
        )

        for occupancy in (0.50, 0.62, 0.72, 0.82):
            if su_h <= 1e-6:
                continue
            cand_h = su_h / occupancy
            cand_w = cand_h * frame_aspect
            if cand_h > 1.0:
                cand_h = 1.0
                cand_w = cand_h * frame_aspect
            if cand_w > 1.0:
                cand_w = 1.0
                cand_h = cand_w / frame_aspect
            if cand_h < su_h - 1e-6 or cand_w < su_w - 1e-6:
                continue

            area = cand_w * cand_h
            if not (self.min_area_ratio <= area <= self.max_area_ratio):
                continue

            for ax, ay, origin in anchors:
                bbox = clamp_bbox(
                    (ax - cand_w / 2, ay - cand_h / 2, ax + cand_w / 2, ay + cand_h / 2)
                )
                if not _contains(bbox, subject_bbox, margin=1e-3):
                    continue
                out.append(Candidate(bbox=bbox, origin=origin, occupancy=su_h / max(1e-6, cand_h)))
        return out

    def _center_grid(self, frame_aspect: float) -> list[Candidate]:
        """无主体时的降级候选：常规取景框。"""
        out: list[Candidate] = []
        for occ in (0.45, 0.60, 0.75, 0.90):
            w = min(1.0, occ * frame_aspect if frame_aspect < 1 else occ)
            h = min(1.0, w / frame_aspect)
            for cx in (0.5, 1.0 / 3.0, 2.0 / 3.0):
                for cy in (0.40, 0.50, 0.58):
                    bbox = clamp_bbox((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                    out.append(Candidate(bbox=bbox, origin="center"))
        return out

    def _dedup_and_limit(self, candidates: list[Candidate]) -> list[Candidate]:
        """按 bbox 四舍五入去重，并限制总数。"""
        seen: set[tuple[float, float, float, float]] = set()
        uniq: list[Candidate] = []
        for c in candidates:
            key = (round(c.bbox[0], 3), round(c.bbox[1], 3), round(c.bbox[2], 3), round(c.bbox[3], 3))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        if len(uniq) > self.max_candidates:
            # 保留"主体占比适中"的候选：过大过小都不是好的取景
            uniq.sort(key=lambda c: abs(c.occupancy - 0.68))
            uniq = uniq[: self.max_candidates]
        return uniq


def _contains(outer: BBox, inner: BBox, margin: float = 0.0) -> bool:
    """判断 inner 是否被 outer 包含（允许 margin 容差）。"""
    return (
        outer[0] - margin <= inner[0]
        and outer[1] - margin <= inner[1]
        and outer[2] + margin >= inner[2]
        and outer[3] + margin >= inner[3]
    )
