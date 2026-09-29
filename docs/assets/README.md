# docs/assets —— 架构图与示意图

> 全部由 `scripts/make_architecture_diagrams.py` **代码生成**，不是手绘。
>
> 复跑：`python scripts/make_architecture_diagrams.py`
> 只生成某张：`python scripts/make_architecture_diagrams.py --only arch_03_debounce.png`
> 列出全部：`python scripts/make_architecture_diagrams.py --list`

## 为什么用代码画图

架构图会随代码漂移。手绘的图在「热路径边界改了」「常量统一了」之后**不会报错**，
只会安静地变成一句谎话。本项目在 D-10 上已经吃过一次同类教训
（配置写着 yolo、实际跑着 rule，指标全失真）——因此把图也纳入可复跑范围：
图中每个模块路径、每个数字、每条边都从源码与配置里读。

## 图清单

| 文件 | 内容 | 对应文档 |
| --- | --- | --- |
| `arch_01_layers.png` | 五层架构 + 热/冷路径隔离 | 《技术方案.md》§1 |
| `arch_02_frame_flow.png` | 单帧数据流（取帧→防抖）与 4 条降级出口 | 《技术方案.md》§1.3、《数据模型与接口.md》§2.6 |
| `arch_03_debounce.png` | 三级防抖状态机 + D-11 安全方向穿透 | 《技术方案.md》§3、《需求说明.md》FR-06 |
| `arch_04_scoring.png` | 6 维加权 + 门控项 + 单帧六维剖面 | 《需求说明.md》FR-03 |
| `arch_05_latency.png` | 热/冷路径延迟量级对照（含预算条） | 《测试与验收.md》AC-N1 |
| `arch_06_gantt.png` | M0–M6 里程碑与当前进度 | 《开发计划.md》 |
| `arch_07_api.png` | API 端点清单 + 典型调用时序 | 《数据模型与接口.md》§3 |

## 生成环境约束（踩过的坑）

1. **必须用 matplotlib，不能用 OpenCV 画中文。**
   OpenCV 的 Hershey 字体**没有中文字形**（本项目早期拼图因此只能写 ASCII 标签）。
   matplotlib 走系统字体，实测可用 `Microsoft YaHei` / `SimHei` / `SimSun`。
   脚本在找不到任何中文字体时**直接报错**，而不是画出一堆豆腐块——
   「图生成了」和「图可读」是两件事。

2. **`U+2713`（✓）等符号在 Microsoft YaHei 中缺字形。**
   实测 matplotlib 会打印 `Glyph 10003 missing` 并渲染成空白。
   已改用纯文字「已完成」。建议生成时加 `-W error::UserWarning` 让这类问题**直接失败**：

   ```bash
   python -W error::UserWarning scripts/make_architecture_diagrams.py
   ```

3. **matplotlib 的 `arrowstyle` 不含 `--|>`。**
   虚线要拆成 `arrowstyle="-|>"` + `linestyle="--"`，否则抛
   `ValueError: Unknown style`。

4. **`--order scale` 与图表数值同源。**
   `arch_05` 会去 `outputs/reports/acceptance_*.json` 里找最近的验收报告读真实延迟；
   找不到就回填 M3 基线值并在图内注明来源。**不编造数字。**
