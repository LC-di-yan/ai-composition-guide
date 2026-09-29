"""滤镜推荐（FR-08）。

对应需求：FR-08（滤镜推荐）
对应文档：《需求说明.md》FR-08

**决策依据（可溯源）：**

滤镜推荐**不靠 VLM 拍脑袋**，而是从构图结果里抽出可解释的信号，
按规则匹配：

| 信号 | 取值来源 | 推荐倾向 |
|------|----------|----------|
| 环境亮度 | ``composition.sub_scores['brightness']`` | 暗 → 高对比/暖调；亮 → 低饱和 |
| 主体高度占比 | ``perception.primary_subject.height`` | 大（特写）→ 人像柔和；小（远景）→ 风景浓郁 |
| 构图模式 | ``composition.pattern`` | 对称/居中 → 经典；对角线 → 高对比 |
| 显著重点 | ``perception.extras['saliency_emphasis_label']`` | 存在 → 强调局部色彩 |

**为什么用高度占比而非面积占比**：人像景别（特写/半身/全身）由主体
**高度**占画面的比例决定，与人体宽高比无关。若用面积，瘦高个会被误判
成远景、宽体被误判成特写——语义就错了。

每个推荐都返回 ``reason``，指明**是哪个信号触发的**——满足 NFR-O2
「建议可溯源」，也让使用者能反驳"为什么推这个滤镜"。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..observability import get_logger
from ..schemas import CompositionPattern, FrameSnapshot

log = get_logger("postprocess.filter")


@dataclass
class FilterRecommendation:
    """滤镜推荐结果。"""

    name: str
    reason: str
    confidence: float
    """0~1，规则命中强度；不是概率，是"信号明确度"。"""
    signals: dict[str, str]
    """触发本次推荐的原始信号，保证可溯源。"""


# 当人格资产未提供滤镜列表时的内置兜底预设
_DEFAULT_FILTERS: list[dict[str, str]] = [
    {"name": "Classic Chrome", "tone": "low_saturation", "scene": "portrait"},
    {"name": "PROVIA", "tone": "standard", "scene": "general"},
    {"name": "Velvia", "tone": "high_saturation", "scene": "landscape"},
    {"name": "S400", "tone": "warm_contrast", "scene": "lowlight"},
]

# 主体面积分级阈值
_CLOSEUP_AREA = 0.35
"""面积超过此值视为特写/近景。"""

_WIDE_AREA = 0.12
"""面积低于此值视为远景。"""

_BRIGHT_LOW = 0.35
_BRIGHT_HIGH = 0.68


class FilterRecommender:
    """基于构图信号的规则滤镜推荐器。

    刻意保持**纯规则、零依赖、可离线**：滤镜推荐发生在拍摄之后，
    对延迟不敏感，但必须**可解释**——这是 NFR-O2 的要求。
    """

    def __init__(self, filters: list[dict] | None = None) -> None:
        self.filters = filters or _DEFAULT_FILTERS
        self._by_tone = {f.get("tone", ""): f for f in self.filters}
        self._by_scene = {f.get("scene", ""): f for f in self.filters}

    # ------------------------------------------------------------------
    def recommend(self, snapshot: FrameSnapshot) -> FilterRecommendation:
        """根据快照推荐滤镜。**保证返回**（最差返回通用预设）。"""
        comp = snapshot.composition
        subject = snapshot.perception.primary_subject
        signals: dict[str, str] = {}

        # 信号 1：亮度
        brightness = comp.sub_scores.get("brightness")
        if brightness is None:
            scenes = snapshot.perception.extras.get("scene_brightness")
            brightness = float(scenes) if scenes is not None else 0.5
        brightness = float(brightness)
        signals["brightness"] = f"{brightness:.2f}"

        # 信号 2：主体占比
        #
        # 用「高度占比」而非「面积占比」：人像摄影的景别（特写/半身/全身）
        # 由**高度**决定，与人体宽高比无关。用面积会让"瘦高的人"被判成
        # 远景、"宽的人"被判成特写，语义错误。
        occupancy = float(subject.height) if subject else 0.0
        signals["subject_occupancy"] = f"{occupancy:.2f}"

        # 信号 3：构图模式
        pattern = comp.pattern
        signals["pattern"] = pattern.value

        # 信号 4：显著重点
        emphasis = snapshot.perception.extras.get("saliency_emphasis_label")
        if emphasis:
            signals["saliency_emphasis"] = str(emphasis)

        # --- 规则链（先特殊后一般，命中即返回）---
        if brightness < _BRIGHT_LOW:
            rec = self._pick("lowlight", "S400")
            return FilterRecommendation(
                name=rec,
                reason=f"画面偏暗（亮度 {brightness:.2f}），暖调高对比能提亮主体、压住噪点观感",
                confidence=round(1.0 - brightness, 2),
                signals=signals,
            )

        if pattern in (CompositionPattern.DIAGONAL,):
            rec = self._pick("low_saturation", "Classic Chrome")
            return FilterRecommendation(
                name=rec,
                reason="对角线构图自带张力，降低饱和度可突出线条与几何关系，不与构图抢戏",
                confidence=0.7,
                signals=signals,
            )

        if occupancy >= _CLOSEUP_AREA:
            rec = self._pick("portrait", "Classic Chrome")
            return FilterRecommendation(
                name=rec,
                reason=f"主体占画面 {occupancy:.0%}（近景/特写），低饱和柔和调更耐看，不会让肤色过曝",
                confidence=0.75,
                signals=signals,
            )

        if occupancy <= _WIDE_AREA or emphasis:
            rec = self._pick("landscape", "Velvia")
            label = "远景" if occupancy <= _WIDE_AREA else "局部显著重点"
            return FilterRecommendation(
                name=rec,
                reason=f"{label}需要色彩张力支撑（主体占比 {occupancy:.0%}），高饱和能强化层次",
                confidence=0.65,
                signals=signals,
            )

        if brightness > _BRIGHT_HIGH:
            rec = self._pick("low_saturation", "Classic Chrome")
            return FilterRecommendation(
                name=rec,
                reason=f"画面偏亮（亮度 {brightness:.2f}），降饱和可避免高光溢出后色彩发灰",
                confidence=0.55,
                signals=signals,
            )

        rec = self._pick("general", "PROVIA")
        return FilterRecommendation(
            name=rec,
            reason="常规光比与主体占比，标准色调能如实还原肤色与材质",
            confidence=0.5,
            signals=signals,
        )

    # ------------------------------------------------------------------
    def _pick(self, scene: str, fallback_name: str) -> str:
        """按场景取滤镜名；人格资产未配置时退回内置名。"""
        item = self._by_scene.get(scene)
        if item and item.get("name"):
            return str(item["name"])
        plain = self._by_tone.get(scene)
        if plain and plain.get("name"):
            return str(plain["name"])
        return fallback_name


def build_filter_recommender(persona_filters: list[dict] | None = None) -> FilterRecommender:
    """按人格资产中的滤镜列表构建推荐器。"""
    return FilterRecommender(persona_filters)
