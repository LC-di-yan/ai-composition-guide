"""配置加载与校验（NFR-M2）。

设计要点：
- 所有可调参数集中在 ``configs/default.yaml``，代码中禁止魔法数字；
- 环境变量可覆盖关键项（API Key 等敏感信息只从环境变量读取，禁止入库，NFR-S1）；
- 全局单例，避免各层重复读盘。

用法::

    from aicg.settings import get_settings
    cfg = get_settings()
    print(cfg.pipeline.target_fps)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

# 项目根目录：src/aicg/settings.py -> 上溯三级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


# --------------------------------------------------------------------------
# 各配置段
# --------------------------------------------------------------------------
class AppConfig(BaseModel):
    name: str = "aicg"
    env: str = "dev"
    log_level: str = "INFO"
    # 是否在启动时打印配置摘要
    verbose_startup: bool = True


class PipelineConfig(BaseModel):
    """实时引导循环参数（FR-05）。"""

    target_fps: float = Field(default=3.0, gt=0.0, le=30.0)
    frame_downsample_width: int = Field(default=480, gt=0)
    frame_timeout_ms: int = Field(default=2000, gt=0)


class DetectorConfig(BaseModel):
    weights: str = "models/yolov8n-seg.pt"
    device: str = "auto"
    conf_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    subject_labels: list[str] = Field(default_factory=lambda: ["person"])
    prefer_person: bool = True

    def resolved_weights(self) -> Path:
        """把权重相对路径解析为绝对路径。"""
        p = Path(self.weights)
        return p if p.is_absolute() else (PROJECT_ROOT / p)


class SaliencyConfig(BaseModel):
    work_size: int = Field(default=160, gt=0)


class PerceptionConfig(BaseModel):
    """感知层配置（FR-02）。backend=auto 时按权重是否存在自动选择。"""

    backend: str = "auto"
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    saliency: SaliencyConfig = Field(default_factory=SaliencyConfig)

    def effective_backend(self) -> str:
        """解析生效后端：yolo / rule。"""
        if self.backend == "auto":
            return "yolo" if self.detector.resolved_weights().exists() else "rule"
        return self.backend


class CandidateConfig(BaseModel):
    """候选框搜索范围（FR-03）。"""

    min_area_ratio: float = Field(default=0.36, gt=0.0, lt=1.0)
    max_area_ratio: float = Field(default=0.94, gt=0.0, le=1.0)
    grid_x: int = Field(default=9, gt=0)
    grid_y: int = Field(default=7, gt=0)
    min_aspect: float = Field(default=0.7, gt=0.0)
    max_aspect: float = Field(default=1.6, gt=0.0)
    top_k: int = Field(default=5, gt=0, description="保留的候选数量")


class ScoringConfig(BaseModel):
    """构图评分权重。所有子项得分归一化到 0~1 后按权重加权，最终 ×100。"""

    weight_rule_of_thirds: float = 0.30
    weight_balance: float = 0.18
    weight_headroom: float = 0.16
    weight_lead_room: float = 0.12
    weight_saliency_center: float = 0.12
    weight_subject_scale: float = 0.22
    """主体占比权重。

    加入原因（真实缺陷修正）：原评分体系缺失"主体大小"维度，导致
    "主体过小的远景"与"主体饱满的半身"得分相同，评分器无法服务
    FR-01 的距离建议主线。SRCC 实测为 0 即由此暴露。
    """

    thirds_tolerance: float = 0.10
    subject_center_tolerance: float = 0.22
    ideal_headroom_ratio: float = 0.12
    ideal_subject_height: float = 0.55
    """**「理想构图」的唯一真源（single source of truth）。**

    理想主体高度占比（大致对应"半身"景别），见 :func:`rules.subject_scale_score`。

    [D-10 修复 2026-09-29] 该值此前只是评分层的私有参数，而"理想构图"
    在另外两层各有一份**互相矛盾**的副本：

    ==================  ==========  ====================================
    层                  常量        量纲/含义
    ==================  ==========  ====================================
    评分层（本字段）     0.55        主体占画面高度比例
    差分层（旧）         0.68        "理想占比"（自称七分身~半身中心）
    距离层（旧）         2.6 m       理想距离（反推约等于占比 0.71）
    ==================  ==========  ====================================

    后果是差分层的 ``hold`` 窗口只有 0.60~0.76（0.68±0.08），而真实人像
    素材的主体占高普遍落在 0.30~0.60 —— 于是 ``move_closer`` 变成了
    **默认输出**而非判断结果；更糟的是同帧内距离层说"太近，请后退"、
    差分层却输出 ``move_closer``，**自相矛盾**。

    现在差分层的目标占比由本字段派生（见
    :meth:`derived_occupancy_tolerance`），三层共用同一口径。
    """

    occupancy_tolerance_ratio: float = 0.15
    """占比容差，以 ``ideal_subject_height`` 的**相对比例**表达。

    取值依据：理想占比现在从 0.68 下调到 0.55（半身景别），绝对容差
    不应保持不变——否则容差会从"占理想的 12%"放大到"占 26%"，把
    "距离已合适"的判据放宽到几乎任何距离都算合适。

    0.15 的含义：允许主体占高在 ``0.55 × (1 ± 0.15)`` = **0.4675 ~ 0.6325**
    区间内视为"距离已合适"。该区间反推距离约 2.9~4.3 m，落在
    ``[min_distance_m, max_distance_m] = [2.0, 5.0]`` 内且不与下界冲突。

    与距离层标称误差（±25%）的关系：占高 15% 偏差对应的距离偏差约
    ±25%，与 :class:`~aicg.composition.distance.DistanceEstimator` 的
    标称误差同量级——即小于该量级的差异**本来就在估算误差之内**，
    不应据此要求用户移动。
    """

    @property
    def derived_occupancy_tolerance(self) -> float:
        """占比容差的绝对值（从 :attr:`occupancy_tolerance_ratio` 派生）。

        差分层的尺度判据直接消费本属性，从而与评分层共用同一个
        ``ideal_subject_height``，不再存在第二份硬编码。
        """
        return self.ideal_subject_height * self.occupancy_tolerance_ratio

    @property
    def total_weight(self) -> float:
        return (
            self.weight_rule_of_thirds
            + self.weight_balance
            + self.weight_headroom
            + self.weight_lead_room
            + self.weight_saliency_center
            + self.weight_subject_scale
        )

    @model_validator(mode="after")
    def _check_occupancy_window(self) -> "ScoringConfig":
        """校验"理想占比 ± 容差"是一个**可达且非全包**的窗口。

        [D-10 修复] 这类约束必须在**配置加载时**失败，而不是延迟到运行时
        表现为"某个分支永远不触发"。D-10 正是这样逃过测试的：容差窗口
        (0.60, 0.76) 与真实素材分布 (0.30, 0.60) 几乎不相交，但代码本身
        没有任何地方会因此报错——它只是持续输出一个错误的默认值。

        两条硬约束：

        1. 窗口上界不得超过 1.0。``ideal + tolerance > 1.0`` 意味着
           "理想占比"本身就超出了画面，是参数标定错误。
        2. 容差必须为正。容差为 0 会让尺度判据退化为浮点相等的精确比较，
           在真实标注抖动下几乎恒为"偏离"，等于取消了容差。

        **不**校验"窗口下界必须大于真实素材分布的众数"——那属于数据
        分布假设，不该写进配置校验（素材会换、景别会变）。
        """
        if self.occupancy_tolerance_ratio <= 0.0:
            raise ValueError(
                f"occupancy_tolerance_ratio({self.occupancy_tolerance_ratio}) 必须为正；"
                "为 0 会让尺度判据退化成浮点精确比较，在真实标注抖动下恒判为偏离。"
            )
        upper = self.ideal_subject_height * (1.0 + self.occupancy_tolerance_ratio)
        if upper > 1.0:
            raise ValueError(
                f"理想占比窗口上界 {upper:.4f} > 1.0："
                f"ideal_subject_height({self.ideal_subject_height}) 与 "
                f"occupancy_tolerance_ratio({self.occupancy_tolerance_ratio}) "
                "标定冲突，理想占比本身已超出画面范围。"
            )
        return self


class DistanceConfig(BaseModel):
    """距离估算参数（FR-01）。

    **[待确认] 焦距与身高假设需按机型标定。**

    **可达区间的物理约束（实测校准，非臆测）**：

    本实现用「等效视角」推导距离::

        FOV_v = 2*atan(sensor_height_mm / (2 * focal_constant))
        distance = (assumed_person_height_m / 2) / tan(box_ratio * FOV_v / 2)

    在默认参数（f_eq=26mm、sensor_h=24mm、H=1.65m）下实测得到::

        框高占比 0.999 → 1.79m   ← 硬下界
        框高占比 0.700 → 2.64m
        框高占比 0.300 → 6.32m
        框高占比 0.050 → 38.2m   ← 已超出可信区间

    因此：**``min_distance_m`` 必须大于 1.79m**，否则"距离太近"分支永远
    不可达（早期设 1.2m 即为该缺陷）。下界的存在原因是主体验证公式在
    θ→180° 时发散，故对 θ 做了 170° 截断。

    同理，``max_distance_m`` 不应设得过大：框高占比 < 0.25 时（约 7.6m）
    主体已不足画面四分之一，估算误差急剧放大，"距离偏远"的提示意义不大。

    生产环境做法：改用深度图（Depth-Anything-V2 / MiDaS）或双摄测距，
    可真正覆盖 0.5~15m 区间。详见 ``EstimationMethod.DEPTH_MAP``。
    """

    assumed_person_height_m: float = Field(default=1.65, gt=0.0)
    focal_constant: float = Field(default=26.0, gt=0.0)
    sensor_height_mm: float = Field(default=24.0, gt=0.0)

    min_distance_m: float = Field(
        default=2.0,
        gt=0.0,
        description="建议最近距离。须大于公式硬下界（默认参数下约 1.79m）",
    )
    max_distance_m: float = Field(
        default=5.0, gt=0.0, description="建议最远距离"
    )
    ideal_distance_m: float = Field(default=2.6, gt=0.0)
    """**仅用于 ``position_ratio`` 的滑条归位点，不代表"理想构图"。**

    [D-10 修复 2026-09-29] 本字段此前被当作"理想距离"的独立定义，反推
    约等于主体占高 0.71，与评分层的 0.55 冲突。"理想构图"的唯一真源是
    :attr:`ScoringConfig.ideal_subject_height`；本字段降级为**纯可视化
    参数**（决定用户看到的距离滑条在中段哪个位置被标为"中间"）。

    它**不参与**动作决策：差分层的前进/后退由占高偏离判定，距离层只负责
    给出可读的米数文案。因此二者即使数值不同也不会产生矛盾建议。
    """

    @model_validator(mode="after")
    def _check_range(self) -> "DistanceConfig":
        """跨字段校验：区间必须有序，且理想值必须落在区间内。

        这类校验必须在**配置加载时**失败，而不是等到运行时产生
        "永远为 False 的判断分支"——后者是静默缺陷，最难排查。
        """
        if self.max_distance_m <= self.min_distance_m:
            raise ValueError(
                f"max_distance_m({self.max_distance_m}) 必须大于 "
                f"min_distance_m({self.min_distance_m})"
            )
        if not (self.min_distance_m <= self.ideal_distance_m <= self.max_distance_m):
            raise ValueError(
                f"ideal_distance_m({self.ideal_distance_m}) 必须落在 "
                f"[{self.min_distance_m}, {self.max_distance_m}] 内"
            )
        return self

    def formula_floor_m(self) -> float:
        """公式可达距离的硬下界（框高占比 = 1.0 时）。

        用于在配置校验与文档中显式暴露该约束，避免再次出现
        "min_distance_m 设得比物理下界还小"的静默缺陷。
        """
        import math

        fov_v_deg = 2.0 * math.degrees(
            math.atan(self.sensor_height_mm / (2.0 * self.focal_constant))
        )
        theta = math.radians(min(fov_v_deg, 170.0))
        return (self.assumed_person_height_m / 2.0) / math.tan(theta / 2.0)


class CompositionConfig(BaseModel):
    candidate: CandidateConfig = Field(default_factory=CandidateConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    distance: DistanceConfig = Field(default_factory=DistanceConfig)

    @model_validator(mode="after")
    def _check_cross_layer_consistency(self) -> "CompositionConfig":
        """跨层一致性检查：把 D-10 那类缺陷变成**加载期硬错误**。

        D-10 的教训是：三层各自"内部自洽"，但**互相矛盾**，而且没有任何
        单一模块会因此报错——它只会在真实素材上表现为持续输出错误动作。
        因此这里显式做一次跨层核对：

        1. **理想占比窗口必须落在距离层可达区间内**。若 ``ideal`` 对应的
           距离已经超出 ``[min_distance_m, max_distance_m]``，说明两条
           建议主线（"该走近"与"距离合适"）在打架。
        2. **理想占比不得高于 0.90**。主体占满整个画面在物理上意味着
           ``tan`` 发散区，距离估算会失去意义。

        本检查**只覆盖可静态判定的冲突**；"素材分布是否落在窗口内"属于
        数据问题，由评测脚本而非配置校验负责（见
        ``scripts/record_web_samples.py`` 的分布统计）。
        """
        ideal_h = self.scoring.ideal_subject_height
        if ideal_h > 0.90:
            raise ValueError(
                f"ideal_subject_height({ideal_h}) > 0.90：主体几乎占满画面，"
                "距离估算公式在 tan 发散区失去意义，无法给出可信的米数建议。"
            )

        # 用与 DistanceEstimator 同源的公式，把"理想占比"翻译成米数，
        # 验证它确实落在建议区间内。
        import math

        d = self.distance
        fov_v_deg = 2.0 * math.degrees(
            math.atan(d.sensor_height_mm / (2.0 * d.focal_constant))
        )
        theta = min(max(math.radians(ideal_h * min(fov_v_deg, 170.0)), 1e-4),
                    math.radians(170.0))
        ideal_m = (d.assumed_person_height_m / 2.0) / math.tan(theta / 2.0)

        if not (d.min_distance_m <= ideal_m <= d.max_distance_m):
            raise ValueError(
                f"跨层不一致（D-10 同类缺陷）：评分层的理想占高 "
                f"{ideal_h} 反推距离约 {ideal_m:.2f}m，落在距离建议区间 "
                f"[{d.min_distance_m}, {d.max_distance_m}] 之外。"
                "这意味着评分器认定的'最佳构图'是差分层永远不会建议的构图。"
                f"请调整 ideal_subject_height，或扩宽 distance 区间。"
            )
        return self


class EmaConfig(BaseModel):
    enabled: bool = True
    alpha: float = Field(default=0.35, gt=0.0, le=1.0)


class DebounceConfig(BaseModel):
    enabled: bool = True
    required_consecutive_frames: int = Field(default=3, ge=1)
    min_interval_ms: int = Field(default=1500, ge=0)


class StabilizationConfig(BaseModel):
    """防抖参数（FR-06）。[待确认] 三参数需 M5 实测调参。"""

    ema: EmaConfig = Field(default_factory=EmaConfig)
    debounce: DebounceConfig = Field(default_factory=DebounceConfig)


class LanguageConfig(BaseModel):
    """语言层配置（FR-07）。provider=mock 时走模板兜底，保证流程不中断。

    **provider 取值（M4 起）**

    - ``mock``    模板兜底，零外部依赖（默认，保证任何环境都能跑通）
    - ``keypool`` 经本机 keypool 本地代理（推荐）。**本进程不持有密钥**，
      由 keypool 负责 27 把密钥的轮询与 429 冷却；模型由
      ``language/model_registry.py`` 按角色决定，可用 ``model`` 强制指定。
    - ``openai`` / ``dashscope`` 等：直连任意 OpenAI 兼容端点，
      需要 ``api_key_env`` 指向的环境变量已设置。
    """

    provider: str = "mock"
    model: str = ""
    base_url: str = ""
    api_key_env: str = "AICG_VLM_API_KEY"
    timeout_s: float = Field(default=30.0, gt=0.0)
    max_calls_per_session: int = Field(default=20, ge=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # glm-5.3 是重推理模型，实测单次约需 456~525 completion tokens；
    # 512 会把预算耗尽在推理上、导致正文被截断（实测只剩"一看"两个字），
    # 因此默认放宽到 2048。
    max_tokens: int = Field(default=2048, gt=0)
    prompt_file: str = "configs/prompts/photographer.yaml"
    keypool_dir: str = ""
    """keypool 目录（留空则自动探测桌面默认位置）。"""

    def api_key(self) -> str | None:
        """从环境变量读取密钥。密钥永不写入配置文件或版本库（NFR-S1）。"""
        return os.environ.get(self.api_key_env) or None

    def resolved_prompt_file(self) -> Path:
        p = Path(self.prompt_file)
        return p if p.is_absolute() else (PROJECT_ROOT / p)


class FilterConfig(BaseModel):
    enabled: bool = True


class RetouchConfig(BaseModel):
    enabled: bool = False


class PostprocessConfig(BaseModel):
    filter: FilterConfig = Field(default_factory=FilterConfig)
    retouch: RetouchConfig = Field(default_factory=RetouchConfig)


class ObservabilityConfig(BaseModel):
    latency_tracking: bool = True
    latency_window: int = Field(default=300, gt=0)
    report_dir: str = "outputs/reports"

    def resolved_report_dir(self) -> Path:
        p = Path(self.report_dir)
        return p if p.is_absolute() else (PROJECT_ROOT / p)


class RetrievalConfig(BaseModel):
    """案例检索配置（FR-09）。

    host/port 拆开放是给 docker compose 用的（服务名做 host）；
    ``uri`` 是 pymilvus 实际消费的完整端点，默认由 host/port 拼接，
    直接设置 ``uri`` 可整体覆盖（如带鉴权参数的场景）。

    Milvus **不在时检索自动降级**（200 + degraded），因此本配置不存在
    也完全不影响其余功能——这是 NFR-R1"运行类问题不 5xx"的配置面体现。
    """

    host: str = "127.0.0.1"
    port: int = Field(default=19530, gt=0, le=65535)
    collection: str = "composition_cases"
    uri: str = ""
    """完整端点（如 ``http://milvus:19530``）；留空则由 host/port 拼接。"""
    hnsw_m: int = Field(default=16, ge=4, le=64)
    ef_construction: int = Field(default=200, ge=8)
    ef: int = Field(default=64, ge=1)
    timeout_s: float = Field(default=5.0, gt=0.0)

    @property
    def resolved_uri(self) -> str:
        """完整端点。环境变量 ``AICG_RETRIEVAL_URI`` 最高优先——docker
        compose 里 api 与 milvus 同网络，用服务名做 host（与语言层
        ``AICG_LLM_BASE_URL`` 的覆盖模式一致）。"""
        env_uri = os.environ.get("AICG_RETRIEVAL_URI")
        if env_uri:
            return env_uri
        if self.uri:
            return self.uri
        return f"http://{self.host}:{self.port}"


# --------------------------------------------------------------------------
# 根配置
# --------------------------------------------------------------------------
class Settings(BaseModel):
    """全项目配置根对象。"""

    app: AppConfig = Field(default_factory=AppConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    composition: CompositionConfig = Field(default_factory=CompositionConfig)
    stabilization: StabilizationConfig = Field(default_factory=StabilizationConfig)
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    # 运行时注入，不来自 yaml
    project_root: Path = Field(default=PROJECT_ROOT, exclude=True)

    # --- 覆盖项（供 CLI / 测试临时改参，不落盘）---
    def with_overrides(self, **kwargs: Any) -> "Settings":
        """返回按嵌套路径覆盖后的新配置，例：``with_overrides(**{"composition.scoring.weight_rule_of_thirds": 0.5})``"""
        data = self.model_dump()
        for dotted, value in kwargs.items():
            node = data
            parts = dotted.split(".")
            for key in parts[:-1]:
                node = node[key]
            node[parts[-1]] = value
        return Settings(**data)

    def summary(self) -> str:
        """启动时的配置摘要，便于复现与排障。"""
        return (
            f"env={self.app.env} "
            f"fps={self.pipeline.target_fps} "
            f"perception={self.perception.effective_backend()} "
            f"vlm={self.language.provider} "
            f"ema={self.stabilization.ema.enabled}@{self.stabilization.ema.alpha} "
            f"debounce={self.stabilization.debounce.enabled}"
            f"({self.stabilization.debounce.required_consecutive_frames}frames/"
            f"{self.stabilization.debounce.min_interval_ms}ms)"
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并字典，override 优先。"""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_dotted_key(node: dict[str, Any], dotted: str, value: Any) -> None:
    """把 ``"a.b.c": v`` 展开成 ``{"a": {"b": {"c": v}}}`` 并原地合并进 node。"""
    parts = dotted.split(".")
    cur = node
    for key in parts[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _normalize_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """把覆盖字典统一成嵌套结构。

    **为什么必须做这一步（真实 bug 复盘）**：

    调用方习惯写扁平的点号键（``{"pipeline.target_fps": 5}``），
    但 ``_deep_merge`` 只认嵌套字典。早期实现直接把扁平键丢给
    ``_deep_merge``，于是 ``"pipeline.target_fps"`` 成了一个**字面
    顶层键**——而 pydantic 默认忽略未声明的字段，覆盖被**静默丢弃**。
    症状极其隐蔽：配置看着传进去了，实际一点没生效（例如
    ``--backend rule`` 无声无息地失效，日志仍显示 yolo）。

    因此这里显式展开点号键，保证"所见即所得"。
    """
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        if "." in key:
            _expand_dotted_key(out, key, value)
        elif isinstance(value, dict):
            out[key] = _deep_merge(out.get(key, {}), value)
        else:
            out[key] = value
    return out


def load_settings(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """从 yaml 加载配置，可选叠加覆盖字典。

    ``overrides`` 同时支持嵌套字典与扁平点号键：
    ``{"perception": {"backend": "rule"}}`` 与 ``{"perception.backend": "rule"}``
    等价。
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _deep_merge(raw, _normalize_overrides(overrides))
    return Settings(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """全局配置单例。"""
    return load_settings()


def reset_settings_cache() -> None:
    """清空配置缓存，供测试使用。"""
    get_settings.cache_clear()
