"""向量特征：从 FrameSnapshot 提取 8 维可解释构图特征（FR-09）。

设计见 ``docs/research/Milvus本地方案调研.md`` §4：

- **不用深度 embedding**（CLIP 双塔 ≈ 数百 MB 重依赖，违背"镜像最小化"
  纪律），改用感知层 + 评分器**已经产出**的构图特征——零新增模型、
  维度全部有界 [0,1]、每一维都有明确语义（面试可讲清楚每一维是什么）。
- 维度名是**契约**：建库与查询必须用同一份 ``DIM_NAMES``，
  次序错位会让相似度完全失真（比"缺一维"更隐蔽）。

**键名核对记录（实现时实测核对，非沿用文档初稿）**：
评分器 ``scorer.py._score_candidate`` 实际产出键为 ``thirds``（非
rule_of_thirds）与 ``subject_center``；本模块以代码为准并在此注释留痕。
"""

from __future__ import annotations

from ..schemas.snapshot import FrameSnapshot

# 向量维度契约：名称 + 次序。建库 / 检索共用，禁止各自硬编码。
DIM_NAMES: tuple[str, ...] = (
    "thirds",            # 三分法贴合度（评分器子项）
    "subject_scale",     # 主体占比与理想占比的贴近度（评分器子项）
    "balance",           # 画面平衡（评分器子项）
    "headroom",          # 头部留白（评分器子项）
    "lead_room",         # 视线留白（评分器子项）
    "saliency_center",   # 显著性居中（评分器子项）
    "subject_height",    # 主体框高占比（感知层原始几何量）
    "center_offset_x",   # 主体中心水平偏移 ×2 归一化（0=居中，1=贴边）
)

DIM = len(DIM_NAMES)
assert DIM == 8, f"向量维度契约应为 8，实际 {DIM}"


def _clamp01(v: float) -> float:
    """防御性裁剪到 [0,1]。上游已保证有界，这里兜底浮点抖动。"""
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))


def features_from_snapshot(snapshot: FrameSnapshot) -> list[float] | None:
    """从单帧快照提取 8 维向量。

    Args:
        snapshot: 帧管线输出（含感知与评分结果）。

    Returns:
        8 维浮点列表（次序与 ``DIM_NAMES`` 一致）；
        **未检出主体时返回 None**——此时几何维（subject_height /
        center_offset_x）没有真实来源，评分器用的是兜底框，
        拿它去检索会把"没有信息"伪装成"有相似度"，属于不诚实行为，
        由调用方走 ``no_subject`` 降级。
    """
    subs = snapshot.composition.sub_scores or {}
    subj = snapshot.perception.primary_subject
    if subj is None:
        return None

    x1, y1, x2, y2 = subj.bbox
    subject_height = y2 - y1
    cx = (x1 + x2) / 2.0
    # 水平偏移归一化：|cx-0.5| 天然 ∈ [0, 0.5]，×2 后铺满 [0, 1]，
    # 避免该维在余弦度量下天然权重减半。
    center_offset_x = abs(cx - 0.5) * 2.0

    vec = [
        subs.get("thirds", 0.0),
        subs.get("subject_scale", 0.0),
        subs.get("balance", 0.0),
        subs.get("headroom", 0.0),
        subs.get("lead_room", 0.0),
        subs.get("saliency_center", 0.0),
        subject_height,
        center_offset_x,
    ]
    return [_clamp01(v) for v in vec]
