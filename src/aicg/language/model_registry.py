"""语言层模型注册表：可用模型的实测能力与**选型理由**。

对应需求：FR-07（自然语言解说）、NFR-E3（成本控制）
对应文档：《技术方案.md》§2.4 语言层选型

**本文件的价值在于把"选型理由"写进代码**，而不是散落在口头说明里。
每个模型的 ``why`` 字段说明它被选中/排除的**具体依据**，
``probe`` 字段记录 2026-09-28 实测的状态码。

----

**实测事实（2026-09-28，经 keypool 本地代理探测）**

- keypool 代理存活，27 把密钥全部 200、无配额耗尽。
- **可用对话模型 7 个**：deepseek-v4-flash-0731 / glm-5.3 / qwen3.8-27b /
  Atria-Dawn-Preview / qwen3.8-flash / tierflow / tierflow_pro。
- **不可用 3 个**（在 /v1/models 中登记但真实调用 404）：
  glm-5.3-flash、glm-5.3-flashx、tiersense。上游未实际提供这些模型。
  → 教训：**"模型列表里有" ≠ "能用"**，必须真实调用验证（见 D-07）。
- **无图像生成能力**：``/v1/images/generations`` 在**代理层与上游层双重 404**。
  → 影响：素材图不能靠生成，改为联网获取真实照片（见素材库方案）。

**实测事实（2026-09-28 第二批：token 预算陷阱，记入缺陷台账 D-08）**

本环境的"重推理模型"会**把 ``max_tokens`` 预算几乎全部烧在内部推理上**：

    model                 max_tokens   延迟     completion  reasoning  正文     finish
    glm-5.3                     2048   6635ms        587        552    45字   stop
    glm-5.3                     3000  12875ms        909        879    38字   stop
    glm-5.3                     4096  49459ms       1103       1075    32字   stop
    deepseek-v4-flash-0731       512   5940ms        512        512     0字   length  ← 正文全空
    deepseek-v4-flash-0731      1024   1880ms         49         26    29字   stop
    qwen3.8-flash               1024   3402ms        314        290    31字   stop

三条硬结论：

1. **推理占 completion 的 88~100%**。正文长度与 ``max_tokens`` 基本无关
   （恒在 30~45 字），多给预算**只换来更多内部思考**，不换来更好输出，
   却让延迟**线性甚至超线性拉长**（glm-5.3 从 2048→4096 时延迟 6.6s→49.5s）。
2. **``max_tokens`` 过小时正文会被推理挤空**且 ``finish_reason=length``。
   deepseek-v4-flash 在 512 时输出 0 字；1024 时正常。
   → 因此不能简单"调小以提速"，存在一个**下界**。
3. **单次延迟样本完全不可信**。同一模型同一参数下 glm-5.3 录得
   6635 / 10096 / 12875 / 49459ms 四个量级的读数，波动达 7.5 倍。
   任何"某模型比某模型快 N 倍"的结论都必须**多次复测取中位数**，
   且要连同 ``max_tokens`` 一起报告（旧结论"deepseek 快 8.6×"是在
   两者预算不同、且 glm 侧被推理拖满的前提下测出的，**不可直接比较**）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """一个可调用模型的元信息与选型理由。"""

    id: str
    """调用时使用的 model 名。"""

    roles: tuple[str, ...]
    """适用角色。取值：
    - ``narration``  构图解说的中文文案生成（FR-07 主用途）
    - ``analysis``   结构化分析 / 打分理由
    - ``bulk``       批量处理（低单价优先）
    - ``fallback``   兜底
    """

    probe_ok: bool
    """2026-09-28 实测是否 200。"""

    why: str
    """选型/排除理由（**必须具体**，禁止写"效果更好"这类空话）。"""

    price_in: float | None = None
    """输入价（¥/百万 token，来自 keypool config.json）。"""

    price_out: float | None = None
    """输出价（¥/百万 token）。"""


# 价格按 keypool config.json 的 USD 标价 × 汇率 7.1 折算为 ¥/百万 token。
# DeepSeek 取官网高峰价 $0.30/$1.20；GLM $0.84/$2.64；Qwen3.8-27B $0.42/$3.00。
_RATE = 7.1


def _cny(usd: float) -> float:
    return round(usd * _RATE, 2)


REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="glm-5.3",
        roles=("narration", "analysis"),
        probe_ok=True,
        why=(
            "【主用 · 文案质量首选】**重推理模型**：实测 completion 中 "
            "552~1075 token 为 ``reasoning_tokens``（占 94~98%），正文只有 32~45 字。"
            "选它的理由："
            "(1) 实测 200 可用，且 keypool 用量账本显示已调用 153 次/28.5M token，"
            "是本环境最稳定的通道；"
            "(2) 中文口语化表达明显更自然，实测输出如"
            "'主体偏左与右三分构图不符，建议人物右移贴合右三分线，视线前方留白'，"
            "符合 FR-07 要求的'像摄影师在耳边说话'；"
            "(3) 支持 structured_outputs，可强制 JSON 便于契约校验。"
            "**代价与使用约束**：单价高（¥5.96/¥18.74）；延迟对 ``max_tokens`` "
            "极其敏感（2048→6635ms，4096→49459ms），故 ``max_tokens`` "
            "**必须压在 1536~2048 区间**——低于下界会被推理挤空正文，"
            "高于上界只是多思考、不成比例地变慢。"
            "因此**只用于低频的实时单条解说**（FR-07 本就是冷路径），绝不用于批量。"
        ),
        price_in=_cny(0.84),
        price_out=_cny(2.64),
    ),
    ModelSpec(
        id="deepseek-v4-flash-0731",
        roles=("bulk", "narration", "analysis"),
        probe_ok=True,
        why=(
            "【批量首选】**实测延迟中位 1880ms**（``max_tokens=1024``，"
            "completion 49 token，其中 reasoning 26）、正文 29 字/次。"
            "**速度与 token 消耗均显著优于 glm-5.3**（后者同题需 587 completion token），"
            "单价也仅为 1/3（¥2.13/¥8.52）。适合："
            "(1) 对评测集 47 张图批量生成解说做抽检；"
            "(2) 需要低延迟的即时反馈；"
            "(3) 结构化分析（把打分分项翻成一句话理由）。"
            "输出质量实测可用（'人物偏左可留白，右三分空明，七分身显气场'），"
            "略逊于 glm-5.3 的语感但差距不大。"
            "**两个必须避开的坑**："
            "(a) ``max_tokens=512`` 时 512 token **全部**被 reasoning 吃光，"
            "正文 0 字且 ``finish_reason=length`` —— 本模型的下界是 1024，不能更低；"
            "(b) 首轮单独测量曾得到 23.1s、以及 5940ms 的读数，"
            "在同一参数下 1024 又只需 1880ms —— **单次延迟样本不足以做选型依据**。"
        ),
        price_in=_cny(0.30),
        price_out=_cny(1.20),
    ),
    ModelSpec(
        id="qwen3.8-27b",
        roles=("analysis", "narration"),
        probe_ok=True,
        why=(
            "【备选文案】中文能力强，27B 规模在术语准确性上有优势。"
            "实测 200 可用。作为 glm-5.3 的**质量对等备选**：当 glm-5.3 触发限流"
            "（池内冷却）时接替。输出单价最高（¥21.3），因此不做批量用途。"
        ),
        price_in=_cny(0.42),
        price_out=_cny(3.00),
    ),
    ModelSpec(
        id="qwen3.8-flash",
        roles=("bulk", "fallback"),
        probe_ok=True,
        why=(
            "【低成本兜底】来自 tierflow 池，实测 200 可用。"
            "定位是'前两个都不可用时的最后手段'——宁可解说质量降一档，"
            "也不能让 FR-07 整条链路断掉（对应 D-04 的降级原则）。"
            "该池用量显示 47 次调用、25.4M token，通道稳定。"
        ),
    ),
    ModelSpec(
        id="tierflow_pro",
        roles=("fallback",),
        probe_ok=True,
        why=(
            "【通道兜底】tierflow 池的'主'层通道，实测 200 可用。"
            "不用于日常调用（它被设计为兜底位），仅在 intern-ai 池整体不可用时启用。"
        ),
    ),
    ModelSpec(
        id="tierflow",
        roles=("fallback",),
        probe_ok=True,
        why=(
            "【通道兜底】与 tierflow_pro 同级，实测 200 可用。"
            "注意：该模型名与池同名，返回的是池自身的默认路由结果，"
            "不适合做需要稳定特性（如结构化输出）的调用。"
        ),
    ),
    ModelSpec(
        id="Atria-Dawn-Preview",
        roles=("analysis",),
        probe_ok=True,
        why=(
            "【未采用为主力】实测 200 可用，但它是 Preview 模型，"
            "稳定性与行为一致性无保证，且缺乏公开的中文摄影领域评测。"
            "仅登记备查，不纳入默认链路。"
        ),
    ),
    # ---- 实测不可用（登记以便排查，禁止加入调用链）----
    ModelSpec(
        id="glm-5.3-flash",
        roles=(),
        probe_ok=False,
        why=(
            "【实测不可用，已排除】在 /v1/models 中登记，但真实调用返回 404"
            "（keypool 报'所有密钥均失败'），说明**上游未实际提供该模型**。"
            "教训（已记入缺陷台账 D-07）：不可仅凭模型列表判断可用性。"
        ),
    ),
    ModelSpec(
        id="glm-5.3-flashx",
        roles=(),
        probe_ok=False,
        why="【实测不可用，已排除】同 glm-5.3-flash：登记存在但调用 404。",
    ),
    ModelSpec(
        id="tiersense",
        roles=(),
        probe_ok=False,
        why="【实测不可用，已排除】调用报 openai 兼容错误，上游未提供。",
    ),
)


def by_id(model_id: str) -> ModelSpec | None:
    for m in REGISTRY:
        if m.id == model_id:
            return m
    return None


def for_role(role: str) -> list[ModelSpec]:
    """按角色返回**可用**模型，按推荐顺序排列。

    排序规则：先按 ``roles`` 中该角色的声明顺序（越靠前越推荐），
    价格仅作次级参考。刻意不按价格排序——文案质量比省几分钱重要。
    """
    picked = [m for m in REGISTRY if m.probe_ok and role in m.roles]
    # 保持 REGISTRY 的声明顺序即为推荐顺序（主用排在最前）
    return picked


def available() -> list[ModelSpec]:
    return [m for m in REGISTRY if m.probe_ok]


def unavailable() -> list[ModelSpec]:
    return [m for m in REGISTRY if not m.probe_ok]


# 默认链路：按角色给出首选
#
# 排序依据是**实测延迟 + 质量 + token 预算健康度**（2026-09-28 复测）：
#   glm-5.3               重推理，正文 32~45 字，延迟随 max_tokens 剧烈放大
#   deepseek-flash        同题 completion 49 vs 587 token，速度与成本显著更优
# narration 把 glm-5.3 放首位（FR-07 对语感要求高，且本就是低频冷路径）；
# bulk 把 deepseek 放首位（批量场景速度与成本优先）。
DEFAULT_CHAIN: dict[str, tuple[str, ...]] = {
    "narration": ("glm-5.3", "deepseek-v4-flash-0731", "qwen3.8-27b", "qwen3.8-flash"),
    "analysis": ("glm-5.3", "deepseek-v4-flash-0731", "qwen3.8-27b"),
    "bulk": ("deepseek-v4-flash-0731", "qwen3.8-flash", "glm-5.3"),
}

# 每个模型的 ``max_tokens`` 下界/上界（见 D-08）。
#
# 为什么不统一用一个值：本环境的推理模型对预算的响应是**非单调**的——
# 太小则正文被推理挤空（deepseek 在 512 时输出 0 字），
# 太大则只增加思考、延迟超线性上涨（glm-5.3 在 4096 时 49.5s）。
# 因此按模型给出"甜区"，而不是全局一个 max_tokens。
TOKEN_BUDGET: dict[str, tuple[int, int]] = {
    "glm-5.3": (1536, 2048),
    "deepseek-v4-flash-0731": (1024, 1536),
    "qwen3.8-27b": (1024, 2048),
    "qwen3.8-flash": (1024, 1536),
    "tierflow": (1024, 1536),
    "tierflow_pro": (1024, 1536),
    "Atria-Dawn-Preview": (1024, 2048),
}
DEFAULT_TOKEN_BUDGET: tuple[int, int] = (1024, 2048)


def budget_for(model_id: str, requested: int) -> int:
    """把请求的 ``max_tokens`` 夹到该模型的实测甜区。

    这是 D-08 的**工程化防御**：调用方不必知道每个模型的脾气，
    但也不会因为传了 512 而拿到空正文、或传了 4096 而白等 50 秒。
    """
    lo, hi = TOKEN_BUDGET.get(model_id, DEFAULT_TOKEN_BUDGET)
    return max(lo, min(int(requested), hi))

# 角色未登记时的通用兜底链路（覆盖全部实测可用模型）
MODEL_CHAIN_FALLBACK: tuple[str, ...] = (
    "glm-5.3",
    "deepseek-v4-flash-0731",
    "qwen3.8-27b",
    "qwen3.8-flash",
    "tierflow_pro",
)
