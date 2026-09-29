"""构图规则：三分线、留白、平衡。

对应需求：FR-03（构图评估）
对应文档：《技术方案.md》§1 构图决策层 —— "主体框位置规则校验（三分线/中心/留白比例）"

设计取向：本模块是**显式规则**而非学习模型，目的是让每一条构图建议都能
被解释和追溯（NFR-O2）。调研指出 Doka 的解说中包含"右三分七分身构图"
这类术语，其位置部分是规则可判定的。

所有函数输入输出均为**归一化坐标**。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.perception import BBox
from ..utils.image import bbox_center, bbox_height, bbox_width

# 三分线位置（归一化）
THIRDS = (1.0 / 3.0, 2.0 / 3.0)


@dataclass
class ThirdsAlignment:
    """主体与三分线的对齐情况。"""

    horizontal: str
    """水平位置描述：``left_third`` / ``right_third`` / ``center`` / ``off``。"""

    vertical: str
    """垂直位置描述：``upper_third`` / ``lower_third`` / ``middle`` / ``off``。"""

    score: float
    """对齐得分 0~1，越接近三分点越高。"""

    distance: float
    """到最近三分交叉点的归一化距离。"""


def thirds_alignment(bbox: BBox, tolerance: float = 0.10) -> ThirdsAlignment:
    """评估主体框与三分线的对齐程度。

    评分逻辑：取主体中心到最近三分交叉点的距离，越近分越高。
    ``tolerance`` 内视为完全对齐。

    Args:
        bbox: 主体框（归一化）。
        tolerance: 容差（归一化坐标）。

    Returns:
        对齐描述与得分。
    """
    cx, cy = bbox_center(bbox)

    # 到最近三分线的距离（水平/垂直分别计算）
    dx = min(abs(cx - THIRDS[0]), abs(cx - THIRDS[1]))
    dy = min(abs(cy - THIRDS[0]), abs(cy - THIRDS[1]))

    # 交叉点距离
    dist = (dx**2 + dy**2) ** 0.5
    # 单一方向对齐也有价值，故取"两轴分别归一化后再合成"的宽松版本
    axis_score = 1.0 - min(1.0, (min(dx, dy) + 0.5 * max(dx, dy)) / max(tolerance * 3.0, 1e-6))
    score = max(0.0, min(1.0, axis_score))

    # 位置描述
    if dx <= tolerance:
        horizontal = "left_third" if abs(cx - THIRDS[0]) < abs(cx - THIRDS[1]) else "right_third"
    elif abs(cx - 0.5) <= tolerance:
        horizontal = "center"
    else:
        horizontal = "off"

    if dy <= tolerance:
        vertical = "upper_third" if abs(cy - THIRDS[0]) < abs(cy - THIRDS[1]) else "lower_third"
    elif abs(cy - 0.5) <= tolerance:
        vertical = "middle"
    else:
        vertical = "off"

    return ThirdsAlignment(horizontal=horizontal, vertical=vertical, score=score, distance=dist)


def headroom_score(
    subject_bbox: BBox,
    candidate_bbox: BBox,
    ideal_ratio: float = 0.12,
    frame_bbox: BBox | None = None,
) -> float:
    """头顶留白评分。

    人像构图中，合适留白既不能顶头（压迫感）也不能过大（主体过小）。
    评分基于主体框顶部到**参照画面上边缘**的距离占参照画面高度的比例，
    与理想值的偏差越小分越高。

    **设计修正（与 balance_score 同源的真实缺陷）**：早期实现以
    ``candidate_bbox`` 上边缘为参照。评估"当前构图"时
    ``subject_bbox == candidate_bbox``，``gap`` 恒为 0，ratio 恒为 0，
    于是 headroom 对**所有**画面返回同一个常数 0.6（= 1 - 0.12/0.30）。
    该分项（权重 0.16）因此完全丧失区分度。修正方式同上：改用独立的
    ``frame_bbox`` 作为参照系。

    Args:
        subject_bbox: 主体框。
        candidate_bbox: 候选画框；主体越出该框顶部视为严重问题。
        ideal_ratio: 理想留白比例（占参照画面高度）。
        frame_bbox: 参照画面范围，默认整幅画面。

    Returns:
        0~1 的得分。
    """
    ref = frame_bbox if frame_bbox is not None else (0.0, 0.0, 1.0, 1.0)
    ref_h = bbox_height(ref)
    if ref_h <= 1e-6:
        return 0.0
    # 主体被候选框裁切 → 严重问题
    if subject_bbox[1] < candidate_bbox[1] - 1e-6:
        return 0.0
    gap = subject_bbox[1] - ref[1]
    if gap < 0:
        # 主体被参照画面上缘裁切
        return 0.0
    ratio = gap / ref_h
    deviation = abs(ratio - ideal_ratio)
    # 偏差 0 -> 1 分；偏差达 0.3 以上 -> 0 分
    return max(0.0, 1.0 - deviation / 0.30)


def lead_room_score(
    subject_bbox: BBox,
    candidate_bbox: BBox,
    face_yaw: float | None = None,
) -> float:
    """前方留白（lead room）评分。

    人物朝向侧应留出更多空间，避免"面壁"构图。朝向未知时退化为
    居中对称留白（不给方向奖励也不惩罚）。

    Args:
        subject_bbox: 主体框。
        candidate_bbox: 候选画框。
        face_yaw: 偏航角；正值表示面向画面右侧。None 表示未知。

    Returns:
        0~1 的得分。
    """
    if face_yaw is None:
        return 0.5  # 未知朝向：中性分，不参与优劣区分

    cand_w = bbox_width(candidate_bbox)
    if cand_w <= 1e-6:
        return 0.0

    left_room = subject_bbox[0] - candidate_bbox[0]
    right_room = candidate_bbox[2] - subject_bbox[2]

    if face_yaw > 5.0:  # 面向右
        front, back = right_room, left_room
    elif face_yaw < -5.0:  # 面向左
        front, back = left_room, right_room
    else:  # 正脸
        balance = 1.0 - abs(left_room - right_room) / cand_w
        return max(0.0, min(1.0, balance))

    if front + back <= 1e-6:
        return 0.0
    ratio = front / (front + back)
    # 理想前方留白约 65%
    deviation = abs(ratio - 0.65)
    return max(0.0, min(1.0, 1.0 - deviation / 0.35))


def balance_score(
    subject_bbox: BBox,
    candidate_bbox: BBox,
    frame_bbox: BBox | None = None,
) -> float:
    """画面平衡评分。

    用"主体框在**画面**中分割出的左右/上下空间比例"衡量平衡感：
    极端失衡（主体贴边）得分低，适度偏心得分高。

    **设计修正（对应一处真实缺陷）**：早期实现以 ``candidate_bbox`` 作为
    参照系，即 ``left = subject[0] - candidate[0]``。这在"评估候选框"时
    成立，但在"评估当前构图"时 ``subject_bbox == candidate_bbox``，
    于是 ``left = right = top = bottom = 0``，四个边距全部退化为 0，
    ``axis_score`` 无条件返回 ``0.0``，``balance`` 子项被**恒定置零**——
    权重 0.20 的分项对所有画面贡献同一个常数，既失去区分度，也让
    "当前构图分"被人为压低。这是一个只在特定调用角色下才暴露的非对称
    bug（同一个规则函数因调用位置不同而行为不一致）。

    修正方式：引入独立的 ``frame_bbox`` 作为参照系（默认整幅画面
    ``(0,0,1,1)``）。这样无论评估的是候选框还是当前构图，边距衡量的
    都是"主体相对画面"的位置关系，语义稳定。

    Args:
        subject_bbox: 主体框（归一化）。
        candidate_bbox: 候选画框；仅用于判断主体是否越出该框。
        frame_bbox: 参照画面范围，默认整幅画面。传入更小的框可评估
            "主体在某个画框内是否平衡"（例如评估候选框构图质量）。

    Returns:
        0~1 的得分。
    """
    ref = frame_bbox if frame_bbox is not None else (0.0, 0.0, 1.0, 1.0)
    ref_w = bbox_width(ref)
    ref_h = bbox_height(ref)
    if ref_w <= 1e-6 or ref_h <= 1e-6:
        return 0.0

    # 越出参照画面 → 严重失衡
    if (
        subject_bbox[0] < ref[0] - 1e-6
        or subject_bbox[1] < ref[1] - 1e-6
        or subject_bbox[2] > ref[2] + 1e-6
        or subject_bbox[3] > ref[3] + 1e-6
    ):
        return 0.0

    # 越出候选框（当候选框比画面小时）→ 同样视为构图缺陷
    if (
        subject_bbox[0] < candidate_bbox[0] - 1e-6
        or subject_bbox[1] < candidate_bbox[1] - 1e-6
        or subject_bbox[2] > candidate_bbox[2] + 1e-6
        or subject_bbox[3] > candidate_bbox[3] + 1e-6
    ):
        return 0.0

    left = subject_bbox[0] - ref[0]
    right = ref[2] - subject_bbox[2]
    top = subject_bbox[1] - ref[1]
    bottom = ref[3] - subject_bbox[3]

    def axis_score(a: float, b: float) -> float:
        if a + b <= 1e-6:
            return 0.0
        ratio = a / (a + b)
        # 理想区间 [0.2, 0.45] 或对称的 [0.55, 0.8]，即主体略偏
        if 0.20 <= ratio <= 0.45 or 0.55 <= ratio <= 0.80:
            return 1.0
        if ratio < 0.20:
            return max(0.0, ratio / 0.20)
        if ratio > 0.80:
            return max(0.0, (1.0 - ratio) / 0.20)
        # 落在正中 0.45~0.55：中心构图也有价值，给中等分
        return 0.75

    return 0.5 * (axis_score(left, right) + axis_score(top, bottom))


def subject_scale_score(subject_bbox: BBox, ideal_height: float = 0.55) -> float:
    """主体占比评分（"主体是不是太小/太大"）。

    **为什么必须有这一项（对应一处真实缺陷）**：早期评分器只衡量主体
    "落在哪里"（三分点、中心、留白），完全不衡量主体"有多大"。后果是
    一个占画面 1% 的"小人"和一个占画面 44% 的"半身人像"拿到**完全相同的
    分数**。而 FR-01（距离建议）的整个前提恰恰是"主体太小 → 该走近一点"——
    评分器无法表达这个缺陷，评分与产品主线需求脱节。

    这是通过与人工标注做 SRCC 相关性分析时暴露出来的：自动分在 45~50
    的窄带内聚集，而人工分横跨 34~88，相关性恒为 0。根因不是"人打分
    主观"，而是**评分函数缺失了一整个维度**。

    评分曲线（以主体框高度占画面的比例衡量，"越大越近"）：
        - ``>= ideal``        满分（主体足够大）
        - ``0.15 ~ ideal``    线性上升（越小分越低）
        - ``< 0.15``          陡降（远景/小人，人像构图几乎无意义）

    Args:
        subject_bbox: 主体框（归一化）。
        ideal_height: 理想主体高度占比，默认 0.55（大致对应"半身"）。

    Returns:
        0~1 的得分。
    """
    h = bbox_height(subject_bbox)
    if h <= 1e-6:
        return 0.0
    if h >= ideal_height:
        return 1.0
    floor = 0.15
    if h < floor:
        # 远景/小人：在 floor 以下继续陡降，最低 0
        return max(0.0, 0.25 * (h / floor))
    # floor ~ ideal 之间线性
    return 0.25 + 0.75 * (h - floor) / max(ideal_height - floor, 1e-6)


def subject_center_score(
    subject_bbox: BBox, candidate_bbox: BBox, tolerance: float = 0.22
) -> float:
    """主体是否被候选框合理包含（主体在框内的相对位置）。

    与 :func:`balance_score` 的区别：本函数只看主体中心相对候选框中心的
    偏移，用于过滤"框跑偏"的候选。

    Args:
        subject_bbox: 主体框。
        candidate_bbox: 候选画框。
        tolerance: 允许的中心偏移（归一化）。

    Returns:
        0~1 的得分。
    """
    sx, sy = bbox_center(subject_bbox)
    bx, by = bbox_center(candidate_bbox)
    dist = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5
    if dist <= tolerance:
        return 1.0
    return max(0.0, 1.0 - (dist - tolerance) / max(tolerance, 1e-6))


def saliency_center_score(saliency: "object", candidate_bbox: BBox) -> float:
    """候选框内显著性质量占比评分。

    Args:
        saliency: 0~1 的显著图（numpy 数组）。
        candidate_bbox: 候选画框。

    Returns:
        0~1 的得分：框内显著性均值相对全图均值的提升程度。
    """
    import numpy as np

    arr = np.asarray(saliency, dtype="float32")
    if arr.size == 0:
        return 0.0
    h, w = arr.shape[:2]
    x1 = int(max(0, min(w - 1, round(candidate_bbox[0] * w))))
    y1 = int(max(0, min(h - 1, round(candidate_bbox[1] * h))))
    x2 = int(max(x1 + 1, min(w, round(candidate_bbox[2] * w))))
    y2 = int(max(y1 + 1, min(h, round(candidate_bbox[3] * h))))

    inside = float(arr[y1:y2, x1:x2].mean())
    overall = float(arr.mean())
    if overall <= 1e-6:
        return 0.0
    # 框内显著性达到全图 1.6 倍以上视为满分
    ratio = inside / overall
    return max(0.0, min(1.0, (ratio - 1.0) / 0.6))


# --------------------------------------------------------------------------
# 构图模式判定
# --------------------------------------------------------------------------
def classify_pattern(
    subject_bbox: BBox,
    saliency: "object | None" = None,
    frame_shape: tuple[int, int] | None = None,
) -> tuple[str, dict[str, float]]:
    """判定构图模式（FR-03）。

    判定顺序（前者优先，因为其条件更强）：
        1. ``framing``    —— 主体被前后景夹住（需分割掩码，当前实现不启用）
        2. ``symmetric``  —— 主体水平居中且左右显著性近似对称
        3. ``center``     —— 主体中心接近画面中心
        4. ``diagonal``   —— 主体长轴明显倾斜
        5. ``rule_of_thirds`` —— 主体中心接近三分点
        6. ``unknown``    —— 无法判定

    Args:
        subject_bbox: 主体框。
        saliency: 显著图，用于对称性判定（可选）。
        frame_shape: ``(h, w)``，暂未使用，保留给后续扩展。

    Returns:
        ``(模式名, 各模式置信度)``。模式名为
        :class:`~aicg.schemas.composition.CompositionPattern` 的取值。
    """
    cx, cy = bbox_center(subject_bbox)
    w = bbox_width(subject_bbox)
    h = bbox_height(subject_bbox)

    conf: dict[str, float] = {}

    # 中心距离
    center_dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    conf["center"] = max(0.0, 1.0 - center_dist / 0.22)

    # 三分点距离
    dx = min(abs(cx - THIRDS[0]), abs(cx - THIRDS[1]))
    dy = min(abs(cy - THIRDS[0]), abs(cy - THIRDS[1]))
    thirds_dist = (dx**2 + dy**2) ** 0.5
    conf["rule_of_thirds"] = max(0.0, 1.0 - thirds_dist / 0.25)

    # 对称性：水平居中 + 显著图左右近似镜像
    sym = 0.0
    if abs(cx - 0.5) <= 0.08:
        sym = 0.45
        if saliency is not None:
            import numpy as np

            arr = np.asarray(saliency, dtype="float32")
            if arr.size:
                left = arr[:, : arr.shape[1] // 2]
                right = arr[:, arr.shape[1] // 2 :][:, ::-1]
                n = min(left.shape[1], right.shape[1])
                if n > 0:
                    l = left[:, :n].mean()
                    r = right[:, :n].mean()
                    denom = max(l + r, 1e-6)
                    sym = 0.45 + 0.55 * (1.0 - abs(l - r) / denom)
    conf["symmetric"] = sym

    # 对角线：主体框长宽比明显偏离 1 且倾斜（用框形状做代理）
    aspect = h / w if w > 1e-6 else 1.0
    diag = 0.0
    if abs(cx - 0.5) > 0.12 and 1.15 < aspect < 2.2:
        # 主体既偏离中心又呈纵向长条，常见于对角线引导
        diag = 0.35 + 0.25 * min(1.0, abs(cx - 0.5) / 0.25)
    conf["diagonal"] = diag

    conf["framing"] = 0.0  # 需要分割掩码，当前实现不启用（诚实标注）

    best = max(conf, key=lambda k: conf[k])
    if conf[best] < 0.30:
        return "unknown", conf
    return best, conf


def describe_pattern(pattern: str, bbox: BBox) -> str:
    """把构图模式转成中文术语，用于自然语言解说（FR-07）。

    对应调研中"Doka 输出右三分七分身构图"这类表述的结构化来源。

    Args:
        pattern: 模式名。
        bbox: 主体框，用于补充位置信息。

    Returns:
        中文描述，如 ``"右三分"``。
    """
    cx, _ = bbox_center(bbox)

    position = ""
    if abs(cx - THIRDS[0]) < 0.10:
        position = "左"
    elif abs(cx - THIRDS[1]) < 0.10:
        position = "右"

    name_map = {
        "rule_of_thirds": "三分法",
        "center": "中心",
        "symmetric": "对称",
        "diagonal": "对角线",
        "framing": "框架式",
        "unknown": "待定",
    }
    base = name_map.get(pattern, "待定")
    if pattern == "rule_of_thirds" and position:
        return f"{position}三分"
    return base


def describe_shot_size(subject_bbox: BBox) -> str:
    """按主体框高度占比推断景别（人像摄影术语）。

    对应调研中"七分身"这类术语——它属于**景别**，由主体在画面中的
    占比决定，而非构图位置。

    占比区间参考人像摄影惯例：
        - ``>=0.90`` 大头/特写
        - ``0.60~0.90`` 半身
        - ``0.30~0.60`` 七分身（头到膝）
        - ``0.15~0.30`` 全身
        - ``<0.15`` 远景

    Args:
        subject_bbox: 主体框。

    Returns:
        景别名词。
    """
    h = bbox_height(subject_bbox)
    if h >= 0.90:
        return "特写"
    if h >= 0.60:
        return "半身"
    if h >= 0.30:
        return "七分身"
    if h >= 0.15:
        return "全身"
    return "远景"
