# CHANGELOG

> 项目：AI 实时构图指导 Agent
>
> 本文件记录所有**实质性变更**。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
>
> **变更类型**：`新增` / `变更` / `废弃` / `移除` / `修复` / `安全`

---

## 版本规划说明

| 阶段 | 版本号策略 | 说明 |
| --- | --- | --- |
| 文档阶段 | `0.x.0` | 仅文档，无可用产物 |
| M1–M3 | `0.x.0` | 功能逐步补齐，API 与契约不稳定 |
| M4–M5 | `0.x.0` | 加入语言层与实时工程优化 |
| M6 及以后 | `1.0.0` | 可部署 Demo 完成，契约冻结 |

---

## [Unreleased]

### 计划中

- M4-5 `language/tts.py` 接入真实 TTS 引擎
- M6 Milvus 案例检索（`retrieval/`，当前 `/v1/cases/search` 返回 501）
- Docker 一键启动实测
- INT8 量化前后推理耗时对比
- 架构图与关键示意图（`docs/assets/`）
- 演示录屏产出

> ⚠️ **依赖风险（诚实说明）**：成本指标在补齐上述前置条件前**不得写入简历**。
> 所有实测值以 [测试与验收.md](./测试与验收.md) §0 为准。
>
> ✅ **D-10 / D-11 已于 [0.6.1](#061---2026-09-29) 修复**。演示不再出现
> "该后退却在喊靠近"的指令，Web 页已把该卡片改造成**缺陷复发哨兵**
> （若再次出现会主动标红告警，而不是当作功能亮点展示）。

---

## [0.6.1] - 2026-09-29

**修复一个「三个常量互不自洽」的跨层缺陷（D-10），并在修复过程中发现
该缺陷并非主因——真正的根因是防抖层闩锁（D-11）。** 本版含一次公开更正。

### 修复

- **D-10：三个常量分属三层却描述同一件「理想构图」**

  `ideal_subject_height = 0.55`（评分层）、`_TARGET_OCCUPANCY_IN_FRAME = 0.68`
  （差分层）、`ideal_distance_m = 2.6`（距离层，折算 ≈ 0.71）三者互不一致，
  导致差分层判「保持」的窗口只有 0.60~0.76。

  修复方式：**以评分层为唯一真源**，差分层不再自带常量，改为从配置派生：

  - 新增 `ScoringConfig.occupancy_tolerance_ratio = 0.15`（容差以**相对比例**表达，
    绝对值由 `derived_occupancy_tolerance` 派生 = 0.55 × 0.15 = 0.0825）；
  - 差分层 `CommandDiffer` 新增 `target_occupancy` / `occupancy_tolerance`
    构造参数，由 `FrameProcessor` 从 `comp.scoring` 注入；
    模块级常量降级为**仅兜底**（`_TARGET_OCCUPANCY_IN_FRAME_FALLBACK = 0.55`）；
  - 新增**加载期跨层校验** `CompositionConfig._check_cross_layer_consistency`：
    把 `ideal_subject_height` 按竖直 FOV 反算成物距，若落在
    `[min_distance_m, max_distance_m]` 之外则**启动即报错**，不再让不自洽的
    配置静默跑起来。

  结果：保持窗口由 **0.60~0.76** 收敛为 **0.4675~0.6325**。

- **D-11：防抖层闩锁（「10 帧全 `move_closer`」的**真因**）**

  修完 D-10 后输出**没有任何变化**，说明归因错了。转去查 `raw_action`，
  发现原始序列 `closer, back, hold, back, back, right, back, closer, back, closer`
  **从来没有连续 3 帧相同**，而 `CommandDebouncer` 第 2 级要求
  **严格连续 N(=3) 帧一致**才能切换 → `_pending_count` 永远到不了阈值 →
  管线**永久锁死在首帧的动作**上。

  证据：`raw_action` 其实**大部分是对的**（10 帧里 6 帧 `move_back`），
  但全部被 `suppressed_by = n_frame_consensus` 拦下，
  `_should_bypass_interval` 里为 `hold`/`move_back` 写的安全方向豁免**从未被执行到**
  ——是一段**不可达的死代码**。

  修复方式（**刻意收窄**，见 §8.1 的取舍说明）：让**安全方向**
  （`hold` / `move_back`）在同时满足三个条件时可穿透闩锁：

  1. 卡住时长 ≥ `min_interval_ms`（用 `now_ms - _last_change_ms`，
     即**距上次被接受的切换**的时长——早期的 `_pending_since_ms` 会被摆动重置，
     等于没修）；
  2. **长窗占多数**：最近 12 帧里该动作出现次数 > 一半（`_LONG_WINDOW_SIZE = 12`）
     ——单看「出现过一次」不够，摆动信号会反复触发（实测 4 动作循环下
     曾反弹到 33 次切换）；
  3. **4 秒冷却**（`_SAFETY_RELEASE_COOLDOWN_MS = 4000`）且本方向未释放过。

  **激进方向（`move_closer` 等）仍严格要求连续 N 帧一致**，不放开——
  FR-06 要的是抑制无谓抖动，不是让指令乱跳。

  > 曾试过「窗口多数表决 + 容忍 1 个离群」，会打破 4 条既有测试，
  > 且经实测验证：真实序列与纯交替序列的 3 窗多数表决分布**完全相同**，
  > 该方法**无法区分**二者，故放弃。这条记录留档是为了说明收窄不是偷懒。

### 变更

- **AC-06 抖动抑制指标显著改善**（240 帧，同一夹具）：

  | 指标 | 官方基线 | 收窄修复前 | 收窄修复后 |
  | --- | --- | --- | --- |
  | 指令切换次数 | 84 → 19 | 86 → 6 | 86 → **6** |
  | 切换降幅 | 77.38% | — | **93.02%** |
  | 平均持续帧数 | 2.82 → 12.00（4.3×） | 2.76 → 34.29 | 2.76 → **34.29（12.4×）** |

- `ideal_distance_m` 的定位从「理想构图的定义」**降级**为
  「距离滑条归位点」，不再冒充评分层的真源。
- 真实素材回归：`walk_towards.mp4` 原始 44/34/12 → 最终 42/36/12（忠实）；
  `handheld_jitter.mp4` 原始 back 152 → 最终 back 225；
  `real_photo_zoom.mp4` 原始 back 118 → 最终 back 120。

### 测试

- 全量 `pytest`：331 → **334 passed, 1 skipped**（无回归）。
- 新增 `tests/test_debouncer.py::TestLivenessUnderOscillation` 3 条活性回归防线：
  安全方向可在间隔后被释放 / 激进方向仍要求连续 N 帧 /
  单帧离群仍被过滤。
- Web 样本用 `--order scale`（按实测主体占比排序）重新录制，
  模拟**物理连续**的取景过程而非 10 张无关照片；
  `verify_web_demo.py` 8 项断言全绿，构图分 74.7 与录制原值逐项相等。

### 文档

- `docs/research/构图指令一致性与防抖闩锁缺陷_D-10_D-11.md` 重写，新增：

  - **§0 复核结论（诚实更正）**：0.6.0 把「10/10 指令方向错误」归因于
    常量不自洽是**错的**。实测 22 张真实人像的主体框高占比中位数是
    **0.794**（区间 0.266~0.992），并非当时声称的 0.30~0.60；
    新旧保持窗口的重叠率只从 13.6% 升到 18.2%，**不足以解释现象**。
    当时漏掉的关键动作是**先看 `raw_action` 再看最终 `action`**。
  - **§7** D-10 修复（单一真源）；**§8** D-11 根因与修复
    （含原始 vs 最终对照表、最小复现、§8.1 为何刻意收窄、§8.2 实测效果、
    §8.3 回归防线）；**§9** 修订版结论；**§10** 复现命令。
  - 初版的归因分析**保留**并标注「已被 §0 推翻」——删掉证据等于毁尸灭迹，
    留着才能说明判断是怎么错的。
  - **文件已重命名**：`距离建议一致性缺陷_D-10.md` →
    `构图指令一致性与防抖闩锁缺陷_D-10_D-11.md`（原文件名只提 D-10，
    会误导读者以为根因是常量问题；改名后与 §8 的真因对齐）。
    全仓 10 处引用（CHANGELOG / 目录结构 / 技术方案 / 测试与验收 /
    `web/index.html` / `differ.py` / 项目卡）已同步。

### 新增：文档口径一致性门禁（AC-DOC）

- 新增 `scripts/check_doc_consistency.py`：扫描全仓行内路径引用查**死链**，
  并对「易漂移事实」声明合法取值集合（测试数 `{334}`、防抖降幅
  `{77.38, 93.02}`、缺陷文档路径白名单），出现集合外的值即失败并返回非零退出码。
- **接入 `acceptance_report.py` 作为第 4 项验收门禁**（AC-DOC），与延迟 / 抖动 /
  SRCC 并列输出 PASS/WARN/FAIL。
- 首次运行即抓出 **5 处真实文档缺陷**（全部已修）：
  - `src/aicg/pipeline/frame_processor.py` docstring 指向不存在的
    `tests/test_frame_processor.py` → 改为 `tests/test_pipeline.py`；
  - `数据模型与接口.md` 声称契约在 `schemas/command.py`，实际在 `schemas/session.py`；
  - `目录结构.md` 把从未创建的 `configs/env.example` 列为「必须进版本库」；
  - `素材图源调研.md` 引用了已清理的临时产物 `outputs/_probe/tpdne_preview.jpg`；
  - `目录结构.md` 多处引用已更名的 `gen_mock_frames.py` / `build_case_index.py`。
- 排除规则（避免误报）：`outputs/` 历史快照不改写、CHANGELOG 历史段整文豁免、
  「方案 vs 实际」对照行（`更名为` / `未实现` / `⏸` 等）不算死链。

### 修复：Docker 编排的 3 处必然失败项 + 新增静态校验门禁（AC-DOCKER）

`docker/` 目录长期标注「✅ 已备（未实测启动）」。做静态审查后发现
**"已备"不等于"写对了"** —— 3 处必错项：

1. **`CMD ["uvicorn","aicg.api.app:app",...]` → 容器启动必然失败**。
   `aicg/api/app.py` 只暴露工厂函数 `create_app()`，**没有模块级 `app`**，
   启动会报 `Attribute "app" not found`。
   已改为 `["uvicorn","aicg.api.app:create_app","--factory",...]`，
   与 `cli/__main__.py` 的 serve 子命令（`uvicorn.run(..., factory=True)`）对齐。
2. **compose 的 `test` profile 跑 `python -m pytest`，但镜像里没有 pytest**。
   Dockerfile 只装运行依赖。已新增构建参数 `WITH_TEST`（默认 0），
   并在 test profile 传 `WITH_TEST=1`；同时**移除 `../tests` 挂载**——
   那是用本地代码遮蔽镜像内 COPY 的 tests，会导致"测的不是镜像里那份"。
3. **缺 `.dockerignore`**：整个项目目录（**114MB**，含 outputs 76M / assets 21M /
   models 6.8M）全量作为构建上下文传给 daemon。
   已新增 `.dockerignore`（放在构建上下文根，因 compose 用 `context: ..`），
   实测 **114MB → 9.4MB（排除 94%）**，且 Dockerfile 需要的路径全部保留。

**新增 `scripts/check_docker_static.py`**：不构建镜像的前提下做 6 类静态校验
（COPY 源存在性 / uvicorn 入口与导出一致性 / 构建参数匹配 / 命令依赖完整性 /
挂载遮蔽 / ignore 规则），并**接入 `acceptance_report.py` 作为第 5 项门禁**。

**该脚本在开发过程中自身踩到的 4 个坑**（已全部修复，记录以便复用）：
- **注释里的示例代码被当成真实入口**：我在 Dockerfile 注释里写了
  `uvicorn.run("aicg.api.app:create_app", factory=True)` 作为说明，
  结果解析器把它当成真入口并误报"缺 `--factory`"。→ 解析前必须 `strip_comments`。
- **CMD 跨行续行**：`CMD [...] \` 折行导致只匹配到前半段，误报缺 `--factory`。
  → 先把 `\\\n` 折成一行。
- **`configs/*.local.yaml` 被误判为排除了整个 `configs/`**：
  朴素前缀匹配的错。→ 改用 `fnmatch` 语义。
- **`CMD_REQUIREMENTS` 同时写 `pytest` 与 `python -m pytest`**，
  同一条命令命中两次产生重复报告。→ 每个包只保留一条最宽松规则。
- **只查"包名是否出现在 Dockerfile"不够**：包可能只出现在注释里，
  或只在一个 `if [ "$ARG" = "1" ]` 条件块里装而 ARG 传的是 0。
  → 必须判断**是否无条件安装**，并回读 compose 传入的 arg 值。

**负向验证**：注入这 3 个原始缺陷 → 门禁精确报出 3 项且退出码 1；
修复后 → 0 项、退出码 0。

> **⚠️ 诚实标注**：静态校验**不能替代实测构建**。它只能证明"没有必然失败项"，
> 无法证明"能构建成功"。实测需启动 Docker daemon 后
> `docker compose -f docker/docker-compose.yml build api`。
> **在真机构建通过前，不得声称"Docker 一键部署可用"。**
> 已在《开发计划.md》把 M6-5 从「✅ 已备」更正为「⚠️ 已修必错项，仍未实测构建」。
- `web/index.html`：已知缺陷卡片拆成两块（① D-10 已修复 + 明确的「诚实更正」；
  ② D-11 防抖闩锁才是真因），常量改为 `D10_HOLD_LO/HI = 0.4675/0.6325`，
  横幅由「D-10 提示」改为**缺陷复发哨兵**（再次出现即标红）。

---

## [0.6.0] - 2026-09-29

**Web 轨升级为可交互引导体验**，并在过程中**发现一个此前被夹具掩盖的真实缺陷**。

### 新增

- `scripts/record_web_samples.py`：从**真实运行中的 API** 录制逐帧响应，
  落到 `web/samples/frames.json`。核心动机是**防止回放样本与后端脱节**——
  手写样本会在字段改名、数值范围变化时静默失真，页面看起来正常，
  实际展示的是一个不存在的系统。录制的字段与 `POST /v1/frame` **同源同构**，
  因此回放与联机**共用同一条渲染路径**。

- `scripts/verify_web_demo.py`：用**真实无头 Chromium（CDP）**做渲染验证。
  `node --check` 只能证明语法没错，证明不了页面能加载、图能画出、
  数据能落到 DOM。本脚本打开页面、推进到指定帧、**断言 DOM 文本与录制原值逐项相等**，
  并截图留证。不新增项目依赖（直接连 CDP）。

- `web/samples/`：10 帧真实响应 + `_meta.json`（记录来源与录制方式）。

- `web/index.html` 交互增强：
  - **时间轴**：10 个帧点，按状态着色（绿=保持 / 红=降级 / 蓝=当前），可点击跳帧
  - **上一帧 / 下一帧**：逐帧查看，替换原来只有"单帧推进"
  - **动作箭头**：在主体框上叠加方向箭头，直观表达"往哪走"
  - **子项分值条**：6 个维度各自带迷你进度条与色阶（原为纯数字）
  - **已知缺陷卡片**：主动暴露 D-10，**不做掩盖**

### 发现并记录

- **D-10：指令方向被钉死为 `move_closer`（真实缺陷，0.6.0 时未修复）**

  > ⚠️ **本节的归因在 [0.6.1](#061---2026-09-29) 被推翻，保留原文作为证据。**
  > 真因是 D-11 防抖层闩锁，不是下面说的常量不自洽。详见
  > `docs/research/构图指令一致性与防抖闩锁缺陷_D-10_D-11.md` §0。

  录制 10 帧真实素材后发现**全部**返回 `move_closer`。排查确认根因是
  **三个常量分属三层、描述同一件"理想构图"却互不自洽**：

  | 常量 | 所在层 | 换算为「框高占比」 |
  |---|---|---|
  | `ideal_subject_height = 0.55` | `scoring` | 0.55 |
  | `_TARGET_OCCUPANCY_IN_FRAME = 0.68` | `differ` | 0.68 |
  | `ideal_distance_m = 2.6` | `distance` | 0.71 |

  差分层判「保持」的窗口只有 **0.60~0.76**（目标 0.68 ± 容差 0.08）。

  > ~~而真实人像多落在 0.30~0.60，于是「靠近」成了默认输出而非判断结果。~~
  > **这句话是错的，已在 0.6.1 更正**：实测 22 张真实人像的主体框高占比
  > 中位数为 **0.794**（区间 0.266~0.992），窗口重叠率仅从 13.6% 升到 18.2%。

  更严重的是**同一帧内自相矛盾**：第 4、9 帧距离层明说「约 1.9 m，太近，
  建议后退」，而下发的 `action` 却是 `move_closer`。

  影响 **FR-04（指令方向）**；**不影响 FR-03（构图评分）**，
  故 Task #14 的 SRCC +0.9113 结论仍然成立。详见
  `docs/research/构图指令一致性与防抖闩锁缺陷_D-10_D-11.md`。

- **为什么之前没发现**：既有测试用 `handheld_jitter.mp4` 等合成/近景夹具，
  主体占比多在 0.45~0.90，恰好落在 `hold` 窗口附近，**夹具覆盖不到真实分布，
  就会把缺陷藏住**。这条结论在 0.6.1 依然成立（D-11 同样是被夹具掩盖的）。

- **退化框**：帧 5 的 `best_bbox = [0.0, 0.0, 0.664, 0.951]` 是锚在原点的
  退化框；页面已加保护，不再把它当作"建议框"画出来误导观众。

### 修复

- **覆盖层错位**：`.overlay` 原先钉在弹性容器 `.stage` 上，而 `canvas` 在其中
  居中且可能更小，导致检测框画到画面外的黑边上。改为把 `canvas` 与
  `.overlay` 一起放进 `.frame` 包裹层，覆盖层贴合 canvas 本身。
  由浏览器渲染验证发现（截图肉眼可见），**语法检查查不出这类问题**。

### 测试

- `pytest` → **331 passed, 1 skipped**（无回归）
- Web 渲染验证 → **通过**（构图分 64.8 与录制原值逐项相等，8 项断言全绿）

> 本版遗留 `move_closer` 缺陷（当时判定为 D-10）已于 [0.6.1](#061---2026-09-29)
> 修复，且真因更正为 D-11。

---

## [0.5.2] - 2026-09-28

**主题素材库**：补齐 0.5.1 明确缺失的"按主题检索"能力。上一版只能拿到
Picsum 随机图（有人像占比 3.4%、无任何主题标签），因此**无法兑现用户
要求的「自拍/咖啡馆/街拍/旅行」场景**。本版找到可行路径并建成 23 张主题素材。

### 新增

- `scripts/fetch_topic_photos.py`：**Bing 图片检索 → 免费图库白名单直链**
  采集。上版实测结论是"Pexels/Unsplash 网页 403、API 401，拿不到主题素材"，
  本版发现两条被忽略的路径：
  1. Bing 图片 async 接口 **200**，返回结构化 JSON（`murl`/`purl`/`desc`）
     ——**这提供了真正的主题检索能力**；
  2. 免费图库**网页**被 Cloudflare 挡，但 **CDN 直链 200**，图片本身可下。

  把两者结合即可按主题拿到合规素材。脚本内置 `FREE_HOSTS` 白名单（10 个
  免费/公共领域域）作为**硬性合规闸门**——**实测不带白名单时，Bing 首轮
  35 条里 28 条来自付费图库**（Freepik 266 / Dreamstime 132 / Vecteezy 97 /
  Alamy 67 / Getty 36），直接抓取有明确版权风险。

- `scripts/curate_topic_photos.py`：**把人工复核结论固化成可复跑脚本**。
  `REVIEW` 表逐张记录「保留/剔除 + 具体理由 + 主题」。剔除文件**移动**到
  `_rejected_with_reason/`（保留证据，不删除）。关键设计取舍：复核结论写在
  代码里而不是手工删文件，否则后人无法区分"漏抓"与"人筛掉了"。

- `scripts/screen_topic_photos.py`：把素材过**本项目自己的 YOLO 链路**，
  判断能否作为演示素材。只读不改，输出 `topic_photos_screen.json`。

- `scripts/make_topic_sheet.py`：生成人工复核拼图（ASCII 标注，含 `OK`/`DEGRADED` 判定）。

- `assets/topic_photos/`：**23 张主题素材**（cafe 6 / street 9 / selfie 3 /
  travel 3 / lifestyle 1 / outdoor 1），含 `PROVENANCE.md` 完整溯源。

- `outputs/reports/topic_photos_contact_sheet.png`（采集后全量 30 张）、
  `topic_photos_curated_sheet.png`（复核后 23 张 + 链路判定）、
  `topic_photos_screen.json`（逐张检测明细）。

### 修复

- **`screen_topic_photos.py` 首版踩的坑：感知后端静默降级导致假结论。**
  首版写成 `YoloPerception(settings)`，但该类的首个参数是 `weights: str`，
  传整个 `AppConfig` 会让权重路径变成一坨 `repr` 字符串 →
  `FileNotFoundError` → **静默降级为规则后端**，`subjects` 恒为 `[]`。
  结果是脚本报出"23 张全都没检出人"，**看起来像严重发现，实际是假的**。
  改用 `perception_from_settings(settings)`（项目文档已就此坑写过 `warning`）
  后，真实检出率 **22/23 = 96%**。同时在脚本里加了**降级即报错退出**，
  避免同类假结论再次出现。

  > 这是本项目第三次因"配置写着 yolo、实际跑着 rule"而拿到失真数据，
  > 前两次记录在 `benchmark_latency.py` 与 `guiding_loop`。

### 数据（实测，非估计）

采集三道漏斗：

| 阶段 | 数量 | 说明 |
|---|---|---|
| 白名单唯一直链 | 58 | 原始命中 400+ 条，白名单后仅 58 条唯一 |
| 下载 + 尺寸达标 | 33 / 58 | 57% —— 未达标多为 404/502 或长边 < 900 |
| 像素级去重后 | 33 | 无重复 |
| 入库 | 30 | 按主题配额选取 |
| **人工复核保留** | **23 / 30** | **剔除率 23.3%** |
| **链路可用** | **22 / 23** | 1 张逆光剪影检测降级（诚实保留） |

人工剔除的 7 张按原因分类：

| 原因 | 张数 | 例 |
|---|---|---|
| 主题误报（检索词 ≠ 内容） | 2 | 用 `woman` 搜出**男性**主体；不可辨的礁石剪影 |
| 合规从严（低俗风险） | 1 | 露肤度较高的街拍 |
| 质量/冗余 | 4 | 同源连拍、玻璃反光、长边不足 900 |

> **这批数据本身即结论**：自动采集的**主题准确率仅 ~93%**（28/30 中 2 张主题错），
> **质量可用率 76.7%**。"爬了就入库"必然把这两类错误带进素材库。

主题检出面积比（实测区间）：cafe 0.39~0.83、street 0.21~0.58、
selfie 0.22~0.99、travel 0.03~0.71（0.03 那张是极远景，作为难度样本保留）。

### 诚实标注

- **`topic_requested` 来自检索关键词，不是图库官方标签**。每条索引都带
  `note` 复述此点；任何基于 topic 的统计都须带此前提，不能当权威分类。
- Pixabay Content License 免费商用、无需署名，但**不允许将图片作为主要
  商品再分发**；本项目仅作演示/评测素材，符合授权范围。
- Bing 检索结果随地域与时间变化，**重跑不保证拿到相同图片**；但已入库
  素材的 `source_url` / `source_page` / `license` 已逐条留存，可独立溯源。

---

## [0.5.1] - 2026-09-28

**演示素材库**：建成 34 张素材（14 张经人工复核的真实照片 + 20 张 AI 人脸），
并**如实记录图源能力的硬限制**——不假装有"咖啡馆/街拍"主题素材。

### 新增

- `scripts/fetch_demo_photos.py`：真实照片采集 + 自动筛选。两阶段
  （缩略图筛选 → 全图下载），按 (景别, 人脸朝向) 网格做**多样性配额**，
  含质量门槛（清晰度 / 曝光）与合规门槛（必须检出人脸）。
- `scripts/organize_demo_assets.py`：按**实测属性**归档 + 生成主题清单。
  索引字段与文件名对齐（修复了一处两者不一致的缺陷）。
- `scripts/curate_demo_assets.py`：**人工复核清单固化为可复跑脚本**。
  逐张记录保留/剔除与**具体理由**，剔除文件保留不删（可复核对）。
- `scripts/fetch_face_samples.py`：AI 人脸采集，**串行 + 限速 + 像素指纹去重**
  （实测并发会命中服务端缓存导致重复）。
- `scripts/make_demo_sheet.py`：生成人工复核拼图（ASCII 标注）。
- `assets/demo_photos/`：素材库本体（含 `README.md` / `PROVENANCE.md` /
  `topics.json` / `review_manifest.json`），逐图可溯源到作者与原始页。
- `docs/research/素材图源调研.md`：**图源可用性实测记录**，12 个渠道逐一探测。

### 变更

- `scripts/acceptance_report.py`："简化处"清单更新——移除已不成立的
  "VLM 解说默认走 mock"（现已接真实模型），新增语言层延迟不在本报告范围、
  keypool 依赖、素材库受图源限制三条。
- `目录结构.md`：补充 `assets/` 资源层与 `docs/research/` 的规范。

### 实测结果

| 项目 | 数值 |
| --- | --- |
| 图源全库 | **993 张**（Picsum，分页第 11 页结束，已是天花板） |
| 可检出主体 | 186 / 993（18.7%） |
| 可检出人脸（合规门槛） | 34 / 993（3.4%） |
| 自动筛选通过 | 19 / 993（1.9%） |
| **人工复核剔除** | **5 / 19（26%）** |
| 最终保留 | 14 张 |
| AI 人脸 | 20 张（串行 3s 间隔 → 有效产出率 **100%**，0 重复） |
| 素材总数 | **34 张** |

### 修复

- **索引字段与文件名不一致**：`_index.json` 存的是内部分桶名
  （`closeup`/`medium`/`wide`/`micro`），而文件名用的是映射后的
  `big_closeup`/`half_body`/...，导致归档脚本按目录名查不到、归档为空。
  已统一为人类可读档名，并保留 `shot_bucket` 以便追溯分档阈值。
- **`cv2.imread` 在中文路径上静默返回 None**：本脚本的兜底函数原先只捕获
  `OSError`，实际 `np.fromfile` 的异常类型不固定，导致兜底失效、
  9 张图读不出来。已改为捕获 `Exception` 并加 `exists()` / 空缓冲检查。
  （同样的问题在本项目已出现两次，见"教训"。）

### 诚实性说明（必读）

- **素材库无法完全满足原始需求**。用户要求「自拍/他拍/咖啡馆/街拍」主题，
  但**可用图源没有主题检索能力**：Pexels/Unsplash API 返回 401（需 Key），
  Wikimedia/Openverse 网络不可达，唯一可用的 Picsum 是**随机图库、
  无任何主题标签**。因此目录名只用**实测属性**，场景主题另出
  `topics.json` 并**显式标注为人工判断、可信度有限**。
- **自动检测误判率 26%**。19 张通过自动筛选的图中，5 张实际不可用：
  包括**压根不是人像的静物**（一串葡萄、办公桌、室内空间）、
  运动模糊、只拍到腿部、男性背影。**人工复核不能省。**
- **AI 人脸源构图多样性为零**。`thispersondoesnotexist` 输出恒为
  正面免冠照，与本项目"构图指导"核心价值冲突，因此**单独存放**在
  `faces/`，并在索引中写明"不可用于构图多样性演示"。
- **并发采集会拿到重复图**。实测并发 6 次仅 2 张唯一内容，
  串行间隔 3s 则 4/4 唯一。脚本因此强制串行 + 像素指纹去重。
  注意**不能用文件字节哈希判重**——同一图字节 md5 不同（EXIF 差异）
  但像素一致，必须降到像素空间比对。

---

## [0.5.0] - 2026-09-28

**语言层真实接入**：从 `provider: mock` 切到 `provider: keypool`，接的是本机
8790 端口的 OpenAI 兼容代理池（27 把第三方密钥池化）。**本项目因此不需要
持有或读取任何真实密钥**——密钥只在 keypool 目录里，代码里一个都没有。

### 新增

- `src/aicg/language/credentials.py`：**运行时**凭据加载器（绝不硬编码、绝不回写）。
  两种模式：`proxy`（默认，**本进程零密钥**，只连本地代理）/ `direct`（直读密钥文件）。
  实现 keypool 的「主/子账号」规则（含「主」标记的仅作兜底），并**去重保序**。
- `src/aicg/language/model_registry.py`：把**选型理由写进代码**。每个模型带
  `probe_ok` / `roles` / `price` / `why`，`why` 必须引用**实测证据**。
  另含 `TOKEN_BUDGET` + `budget_for()` 逐模型 token 甜区（见下）。
- `src/aicg/language/client.py`：链路降级客户端。按角色选模型链，
  逐次尝试记录到 `attempts`，**任何失败都不抛异常**，转成
  `LLMResult(ok=False, degraded=True, reason=...)`。
- `scripts/probe_keypool.py`：只读能力探测（10 个对话模型 + 12 个图像模型）。
- `scripts/verify_language.py`：真实照片跑完整「感知 → 构图 → 真实 LLM 解说」链路，
  如实打印是否降级、实际模型、耗时、token。

### 变更

- `configs/default.yaml`：`language.provider` 由 `mock` 改为 `keypool`；
  `max_tokens` 512 → 2048（详见 D-08）。
- `configs/prompts/photographer.yaml`：`system_prompt` **收紧为硬约束清单**。
  旧版只写"生成 1~2 句构图理念解说"，实测出现啰嗦（1145 token）与截断（只剩"人"字）
  两类失败；改为 7 条硬约束 + 正反例后，同一困难样本上稳定输出 31~33 字。
  **结论：对重推理模型，软性建议不生效，必须写成硬约束。**
- 三处语言层测试改为**显式** `provider: mock`，不再寄生全局默认值（详见"修复"）。

### 实测结果

| 项目 | 结论 |
| --- | --- |
| keypool 存活 | 27/27 密钥健康（deep health-check） |
| 可用对话模型 | **7 个**；另有 3 个"登记但 404"（见 D-07） |
| 图像生成 | **无**。`/v1/images/generations` 在代理层与上游层双重 404 |
| glm-5.3 实测输出 | "右三分站位很稳，画面平衡也正，就差头顶留白这一口气"（40 字） |
| 困难样本（远景/低分） | 收紧提示词后 3/3 成功，25~32 字 |
| 测试 | **331 passed, 1 skipped** |

### 修复

- **[D-07]「模型列表里有」≠「能用」**：`/v1/models` 登记 16 个模型，其中
  `glm-5.3-flash`、`glm-5.3-flashx`、`tiersense` 真实调用返回 404/502，
  上游根本未提供。已登记为 `probe_ok=False`、`roles=()`，链路自动跳过。
- **[D-08] token 预算陷阱（重推理模型必读）**：本环境的模型把 `max_tokens`
  几乎全烧在**内部推理**上——glm-5.3 的 completion 中 94~98% 是
  `reasoning_tokens`，正文只有 32~45 字。后果有三：
  1. 预算太小（512）会让正文**被挤空**、`finish_reason=length`；
  2. 预算太大只增加思考，延迟**超线性上涨**（2048→6.6s，4096→49.5s）；
  3. 正文长度与预算**无关**，多给不换来更好输出。
  已加入 `TOKEN_BUDGET` / `budget_for()`，把每个模型的预算夹到实测甜区。
- **[D-09] 截断是随机且隐蔽的**：困难样本（远景+低分）上 glm-5.3 会返回
  **单个"人"字**，HTTP 200、非空、看起来"成功"。只判空的检查会漏掉它。
  已加入 `min_chars` 门槛 + **原地重试一次**（截断随机，换模型要付完整延迟，
  重试更划算），并把 `finish_reason` / `reasoning_tokens` 写进失败原因。
- **测试寄生于全局默认值**：`test_reports_fallback_status` 等三处注释写着
  "默认 provider=mock"，实际依赖 `configs/default.yaml`。默认值改 keypool 后，
  这些单测变成**真的去打网络**，随机失败且慢（单次 63s）。
  已改为显式传 `provider: mock`。**教训：测试必须显式声明前提。**
- **`_call_via_keypool` 空错误信息**：HTTP 200 但正文空时，
  `RuntimeError(f"...失败: {res.reason}")` 会打印出冒号后什么都没有。
  已改为分支诊断（区分"链路失败/空正文/截断"，并带上 token 明细）。

### 诚实性说明

- 早期结论"deepseek 比 glm-5.3 快 8.6 倍"**已撤回**。该读数是在两者
  `max_tokens` 不同、且 glm 侧被推理拖满的前提下测出的，**不可直接比较**。
  同一模型同一参数下 glm-5.3 实测录得 6635 / 10096 / 12875 / 49459ms
  四个量级（波动 7.5 倍）。**任何模型延迟对比都必须多次复测取中位数，
  且连同 `max_tokens` 一起报告。**
- 语言层延迟（glm-5.3 约 7~15s）**只在冷路径**（按快门后生成解说）。
  热路径仍是规则打分，实测 P95 约 28~37ms。两者不可混为一谈。

---

## [0.4.0] - 2026-09-28

**评测集重建**：把 SRCC 从「n=5、不具统计效力」升级为「n=47、统计有效」，
并同步校正一处**延迟数字记录缺陷**（详见"修复"节）。

### 新增

- `scripts/fetch_eval_photos.py`：真实素材采集（两阶段：Stage1 缩略图+YOLO 筛图，
  Stage2 按主体占比**分桶配额**取全图；自动产出 provenance 记录）。
  从 800 张候选筛出 179 张含主体图（22.4%），落地 48 张。
- `scripts/build_principle_annotations.py`：**原则驱动**标注生成。
  与评分器 `rules.py` **完全独立重写**（不共享代码/权重，主体占比曲线刻意改为单峰），
  以避免循环论证。
- `scripts/analyze_srcc.py`：SRCC 深度分析（退化检查 / 逐维度相关 / **共线性诊断** /
  分层 SRCC / **偏相关**），把"这个数字虚不虚"的检验过程固化为可复跑脚本。
- `scripts/make_review_sheet.py`：生成 18 张人工抽检表 + 拼图（ASCII 标注，
  规避 Hershey 字体无 CJK 字形限制），用于后续补齐主观标注。
- `configs/eval/images_real/`：48 张真实素材 + `PROVENANCE.md`（逐图作者/原始页/授权）
  + `_index.json`。
- `configs/eval/annotations_real.json`：47 条原则驱动标注（含逐项分与局限声明）。
- `outputs/reports/构图评分相关性评测报告.md`：完整方法学 + 检验过程 + 简历表述条件。

### 变更

- `scripts/evaluate_composition.py`：结果 JSON 增加 `annotation_method` /
  `is_subjective_human_rating` / `known_limitation` 三个字段，使标注性质随结果落盘。
- `scripts/acceptance_report.py`：`measure_srcc()` 改为**优先使用真实素材集**，
  缺失时回退构造集并在报告中标注，不再默认输出 n=5 的旧结论。
  "简化处"清单更新为 SRCC 的准确局限（原则驱动 / 共线性 / balance 弱）。
- 文档同步：README / 测试与验收 / 开发计划 / 技术方案 / 需求说明 / CHANGELOG
  中的 SRCC 与延迟数字全部更新至最新实测。

### 实测结果

| 指标 | 旧 | 新 |
| --- | --- | --- |
| SRCC | +0.8000（n=5） | **+0.9113（n=47）** |
| 偏相关（控制主体大小） | 未测 | **+0.8216** |
| 分层 SRCC | 未测 | +0.9564 / +0.8737 / +0.6667（n=20/19/8） |
| 统计效力 | ❌ | ✅ |

### 修复

- **[D-06] 延迟数字记录了最低读数而非稳定值（文档缺陷）**：
  早期文档统一引用 **P95 = 24.4ms（7.3%）**，该值在 2026-09-28 的三次独立复测中
  **未能复现**（复测为均值 29.5–31.6ms、P95 33.7–37.2ms）。
  排查确认**非代码改动所致**（同版本代码），属跨次测量的正常波动
  （CPU/GPU 调度、系统负载），而文档当初只登记了最有利的一次读数。
  已改为**区间表述**并注明成因，同步新增 `scripts/sync_latency_docs.py` 做批量校正。
  教训：**性能指标应记录多次运行的区间与条件，不得只记单次最优值。**

### 诚实性说明（本次新增，必读）

- SRCC = +0.9113 **含共线放大**：`headroom` 与主体大小相关系数 r = −0.8278，
  两套实现相当程度上都在测"主体有多大"。**剔除后偏相关 +0.8216 才是保守读数。**
- `balance` 维度有效性弱（与原则 SRCC 仅 +0.1517），属真实待改进项。
- 标注为**原则驱动**，**不等于主观人工评分**。本指标支撑
  「评分器与构图学原则同向」，**不支撑**「符合人类审美」。
- 评测集含非人像主体 4/48（8.3%，dog×2/bird×1/无主体×1），属评分器
  `subject_labels` 规格内行为，但若结论仅针对人像需注意该污染率。

---

## [0.3.1] - 2026-09-28

**文档一致性校正**：以运行时 schema 为基准，逐项核对 9 份文档与代码的一致性。

### 修复（文档 vs 代码不一致）

| 项 | 文档原先写的 | 实际代码 | 处置 |
| --- | --- | --- | --- |
| API 路径形态 | `/v1/sessions/{sid}/frames` | `/v1/frame`（扁平，`session_id` 走请求体） | 已更正 4 处 |
| `/metrics` 端点 | 被 `/readyz` 取代 | **两者都存在**（职责不同） | 已更正 |
| 图像请求字段 | `image_base64` + `image_path` 双字段互斥 | 单字段 `image_ref`（路径 / base64 前缀判别） | 已更正 |
| 验收调用方式 | `python -m aicg accept` | 包未安装，须 `PYTHONPATH=src python -m aicg.cli accept` 或 `python scripts/acceptance_report.py` | 已全量更正 |
| 延迟指标 | P95 28.2ms（8.4%） | **P95 33.7–37.2ms**（均值 29.5–31.6ms，占 ~10–11%） | 已按复测校正：不再引用 24.4ms 点值，改为区间 |
| `degradation_reason` 取值 | `no_subject_detected` / `limited_candidate_space` | `subject_detection_failed` 等 6 个封闭枚举 | 已按枚举重写 |
| `LatencyBreakdown` | 含 `language_ms` | 含 `stabilization_ms`；`language_ms` 挂在 `narration` 下 | 已更正 |
| REST 端点数 | "10 个 REST" | 7 业务 + 3 运维探针 | 已更正 |

### 变更

- `README.md` 重写为 **v0.2**：新增 §0 实测指标表、§3.2 冷热路径隔离、§7 诚实性声明（含缺陷表与简化对照表）、§9 待确认清单（已决/未决分区）
- `技术方案.md` 升级 **v0.2**：M0 选型倾向全部回填为实测结论；新增 §2.1.1（FastAPI `def` vs `async def` 陷阱）、§2.3.1（为何不直接用 SAMPNet）、§2.4.1（降级层两个真实设计点）
- `数据模型与接口.md` 升级 **v0.2**：契约草案 → 实现契约；新增 §1.2（坐标归一化的参考系陷阱）、`subject_scale` 评分维度说明、§7 变更记录
- `测试与验收.md`：§0 结论表更新为最新实测；标注运行间波动区间
- `需求说明.md` 升级 **v0.2**：NFR-P1/P2/P3/O1/O3/R4 回填实测；FR-06 防抖三参数定稿（α=0.35 / N=3 / 1500ms）；§6 待确认清单分区
- `目录结构.md` 升级 **v0.2**：新增 **§2.5 实现偏差核对**（规划蓝图 → 落地结构的 11 项差异及原因），并保留 M0 蓝图不改
- `编码规范.md` 升级 **v0.2**：新增 **§9 实战补充条款**（多参考系命名、降级不抛异常、指标可复现、FastAPI `def` 陷阱、契约以运行时 schema 为准），每条对应一个真实缺陷编号

> **方法论沉淀**：契约文档必须以 **运行时 schema**（`GET /openapi.json`、`app.routes`）为准，不得凭设计意图或记忆书写。本次校正发现 **8 处**文档与代码不符，全部源自"按计划书写"而非"按实现核对"。

---

## [0.3.0] - 2026-09-28

**里程碑：M0 → M3 全部出口**（M5 指标提前达标）。从"仅有文档"推进到
"可运行、可演示、指标可复跑"的完整系统。

### 新增

- **感知层**（FR-02）
  - `perception/detector.py`：YOLOv8n-seg 封装，含 `warmup()`（NFR-P1）
  - `perception/saliency.py`：显著性图
  - `perception/rule_backend.py`：规则兜底后端（无权重/无 GPU 可用）
  - `perception/factory.py`：`perception_from_settings()` + **按素材名推断后端**（双源分流）
  - `perception/base.py`：`PerceptionBackend` Protocol（NFR-M1 可替换性）
- **构图层**（FR-01 / FR-03 / FR-04）
  - `composition/scorer.py`：`HeuristicCompositionScorer`（默认）+ `SAMPNetScorer`（接入桩）
  - `composition/candidate.py`：网格搜索候选框
  - `composition/rules.py`：三分线 / 平衡 / 留白 / **主体占比** / 模式判定
  - `composition/distance.py`：等效视角距离估算
  - `composition/differ.py`：差分 → 七类动作指令
- **防抖层**（FR-06）
  - `stabilization/ema.py`：EMA 平滑（α=0.35）
  - `stabilization/debouncer.py`：N 帧一致 + 最小切换间隔
- **管线层**
  - `pipeline/frame_processor.py`：单帧热路径编排
  - `pipeline/guiding_loop.py`：引导循环 + 防抖对照实验
  - `pipeline/postshot.py` / `calibration.py` / `subject.py`：冷路径
- **语言层**（FR-07 / FR-08）
  - `language/vlm_client.py`：真实 VLM + 模板兜底**双路**（无 Key 不断流程）
  - `language/tts.py`：占位（未接引擎）
  - `postprocess/filter_recommend.py`：滤镜推荐
- **API 层**（NFR-O3）
  - **7 个业务 REST 端点** + **3 个运维探针** + **1 个 WebSocket**：
    - 业务：`POST /v1/session`、`DELETE /v1/session/{id}`、`POST /v1/calibrate`、`POST /v1/subject`、`POST /v1/frame`、`POST /v1/shot/report`、`POST /v1/cases/search`（501 占位）
    - 运维：`GET /healthz`、`GET /readyz`、`GET /metrics`（均不带 `/v1` 前缀）
    - 流式：`WS /v1/stream`
  - 路径采用**扁平形态**（M0 草案即如此），`session_id` 走请求体
  - `/healthz`（不碰感知）/ `/readyz`（就绪探测，会构建感知）/ `/metrics`（Prometheus 文本，无额外依赖）
  - 图像传输：单字段 `image_ref`（本地路径 **或** base64 data URI，按前缀判别）
  - `api/session_store.py`：线程安全会话存储（TTL + LRU，可换 Redis）
- **CLI**（8 个子命令）
  - `demo` / `score` / `serve` / `make-fixtures` / `bench` / `eval` / **`accept`** / `doctor`
- **Web 轨**（本次新增）
  - `web/index.html`：自包含演示页，三模式（演示样本 / 联机 API / 待机）
  - 挂载于 `GET /`（同源，免 CORS 配置）
  - **中文文案层**：录屏轨受 OpenCV 字体限制只能英文，中文由 Web 轨承担
- **脚本**
  - `scripts/run_demo.py`：录屏轨主交付（`--backend` / `--compare`）
  - `scripts/make_fixtures.py`：**双源夹具**生成器
  - `scripts/benchmark_latency.py`：延迟基准（冷/稳态分离）
  - `scripts/evaluate_composition.py`：SRCC 评测（含样本量警示）
  - **`scripts/acceptance_report.py`**：端到端验收报告（延迟+抖动+SRCC 一次跑完）
- **配置**
  - `configs/default.yaml`：全量阈值外置（NFR-M2）
  - `configs/prompts/photographer.yaml`：人格化 prompt
  - `configs/eval/annotations.json`：SRCC 标注集（含 `known_limitation`）
- **测试**：**331 用例通过**（+1 skipped），覆盖分层单元 / 契约 / 集成 / 验收逻辑

### 变更

- **评分权重调整**（契约影响：评分数值变化，字段未变）
  - `weight_rule_of_thirds` 0.34 → **0.30**
  - `weight_balance` 0.20 → **0.18**
  - `weight_headroom` 0.18 → **0.16**
  - `weight_lead_room` 0.14 → **0.12**
  - `weight_saliency_center` 0.14 → **0.12**
  - **新增** `weight_subject_scale` = **0.22**
  - **新增** `ideal_subject_height` = **0.55**
  - 原因见"修复"D-01/D-02/D-03；调整后同一标注集 SRCC 由 0.0000 → **0.8000**
- **`balance_score` / `headroom_score` 新增 `frame_bbox` 参数**（非破坏性：有默认值）
  - 参照系由"候选框"改为"画面"，语义在两种调用角色下一致
- **`perception.base.warmup()` 签名扩展**：新增 `image_shape` 可选参数
- **API 启动新增感知预热**：首帧延迟 6296ms → **204ms**
- 文档版本 `0.1（草案）` → `0.2`，`待确认` 项按实测回填为**实测值**

### 修复

| 编号 | 描述 | 级别 |
| --- | --- | --- |
| D-01 | `balance_score` 参照系退化，该分项**恒为 0.0** | P1 |
| D-02 | `headroom_score` 参照系退化，该分项**恒为 0.6** | P1 |
| D-03 | 评分器缺失"主体占比"维度，主体过小与饱满**同分** | P1 |
| D-04 | **解说误报"未识别到主体"**——对已检出 `person` 的画面输出与事实相反的内容 | **P0** |
| D-05 | settings 扁平点号覆盖被**静默丢弃**（`--backend rule` 无声失效） | **P0** |
| D-06 | 合法 base64 被误判为文件路径（Base64 含 `/`） | P1 |
| D-07 | YOLO `warmup()` 参数与 `infer()` 不一致，走不同代码路径（首帧 7099ms） | P1 |
| D-08 | API 未预热感知后端，冷启动成本落在用户等待（首帧 6296ms） | P1 |
| D-09 | `debouncer` 契约缺陷：`is_changed=True` 却返回旧指令 | **P0** |
| D-10 | CLI `bench` 与脚本参数名不一致（`--runs` vs `--repeat`），子命令直接崩 | P2 |

> 每个缺陷均对应一条**回归测试**锁死，清单见 [测试与验收.md](./测试与验收.md) §6。
> 其中 D-01~D-03 是排查 SRCC=0 时**顺藤摸瓜发现的真实 bug**，而非测试写错；
> 修复带来的指标提升不是调参凑数。

### 指标回填（NFR-O3 可复跑）

| 指标 | 实测值 | 来源脚本 |
| --- | --- | --- |
| 端到端延迟 P95 | **33.7–37.2 ms**（均值 29.5–31.6ms） | `benchmark_latency.py` |
| 延迟分段 | 感知 ~19ms（**占 ~99%**）/ 决策 0.14ms / 防抖 0.04ms | 同上 |
| 首帧冷启动 | **6296ms → 204ms**（API 启动预热，**31×**）；预热后稳态势 <30ms | 实测 |
| 指令切换降幅 | **77.38%**（84→19 次/240 帧） | `run_demo.py --compare` |
| 平均指令持续帧数 | 2.82 → 12.00（**4.3×**） | 同上 |
| SRCC | **+0.9113**（n=47，真实素材集，**统计有效**） | `evaluate_composition.py` + `analyze_srcc.py` |
| 验收门禁 | **PASS 2 / WARN 1 / FAIL 0** | `acceptance_report.py` |
| 测试用例 | **331 passed / 1 skipped** | `pytest` |

### 备注

- **双源分流（重要工程约束）**：实测确认 **YOLOv8n-seg 无法检出任何
  "代码画出来的人形"**（纯椭圆 / 火柴人 / 带噪声的写实化版本全部返回 0 主体）。
  因此夹具与后端**配对使用**：合成素材 → `rule` 后端；真实素材 → `yolo` 后端。
  落地在 `make_fixtures.py` / `tests/fixtures/README.md` /
  `perception/factory.infer_backend_for_source()`，并有回归测试守护。
- **SRCC 的构造式标注局限**：5 张裁切图在感知层被"扩张至近全幅"，
  导致想验证的"主体占比"维度被抹平。详见 `configs/eval/annotations.json`
  的 `known_limitation` 字段与 [测试与验收.md](./测试与验收.md) §4.1。
- 新增 `requirements.lock.txt`：精确版本锁，用于复现实验指标
  （`requirements.txt` 仍为下限锁定，适合开发）。

---

## [0.2.0] - 2026-09-28

### 新增

- M0 决策落地（AskUserQuestion 确认）：
  - 交付形态：**Web + 录屏双轨**
  - 开发深度：**一口气推到 M3**
  - 模型策略：**真实模型 + 可降级桩并存**
  - VLM 集成：**抽象接口 + 模板兜底**
- M0-3 结论：SAMPNet/CADB 权重未接入，先用**可解释的启发式评分**，保留替换接口
- M0-4 结论：YOLOv8n-seg 可用，但**检不出合成人形** → 引出双源分流设计
- `schemas/`：`FrameSnapshot` 等核心契约冻结（NFR-O1）
- 仓库骨架落地：`src/aicg/` 按五层架构分目录

### 备注

- 本版起**包含业务代码**（此前仅文档）

---

## [0.1.0] - 2026-09-28

### 新增

- 建立项目文档体系（9 份），作为后续开发的唯一依据：
  - `README.md` — 项目定位、目标用户、核心功能概述
  - `需求说明.md` — 功能/非功能需求，P0/P1/P2 分级
  - `技术方案.md` — 技术选型、依赖与环境要求
  - `目录结构.md` — 文件与模块组织规范
  - `开发计划.md` — 分阶段里程碑与当前阶段任务清单
  - `编码规范.md` — 命名、注释与提交信息格式
  - `数据模型与接口.md` — 核心数据结构与接口约定
  - `测试与验收.md` — 验证方式与验收标准
  - `CHANGELOG.md` — 本文件
- 确立项目定位：面向普通拍照者的实时构图指导 Agent，护城河为**实时工程 + 稳定性 + 产品兜底设计**
- 确立 6 周分期路线（M0–M6），对齐调研建议

### 说明

- 本版**仅含文档**，不含任何业务代码
- 所有未定信息统一标注 `待确认` / `TODO`，未做细节臆造
- 技术选型中仅 Python / FastAPI / Milvus / Docker 标注为 `已定`，其余均待验证

---

## 变更记录模板

> 复制以下模板追加新版本。

```markdown
## [x.y.z] - YYYY-MM-DD

### 新增
- 新增内容（关联需求编号：FR-xx / NFR-xx）

### 变更
- 变更内容
- **破坏性变更**：说明影响范围与迁移方式

### 废弃
- 废弃内容及原因

### 移除
- 移除内容及原因

### 修复
- 修复内容（关联缺陷级别：P0/P1/P2/P3）

### 安全
- 安全相关修复或加固

### 备注
- 补充说明、遗留问题、后续计划
```

---

## 变更记录要求

| 要求 | 说明 |
| --- | --- |
| 面向读者 | 变更描述面向使用者，说明"影响什么"，而非"改了哪个文件" |
| 关联需求 | 功能性变更须标注 FR / NFR 编号 |
| 契约变更 | 涉及 `数据模型与接口.md` 的变更须标注**是否破坏性** |
| 指标回填 | 实测得到的指标数值须在此记录，便于追溯 |
| 简化处登记 | 新增简化实现须在此登记，并同步至 [开发计划.md](./开发计划.md) §6 |
| 日期格式 | `YYYY-MM-DD` |
| 倒序排列 | 最新版本在最上方（`Unreleased` 之后） |
