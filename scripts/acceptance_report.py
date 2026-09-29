#!/usr/bin/env python
"""端到端验收报告（NFR-O3）。

对应需求：NFR-O3（指标可复跑）、NFR-P1（延迟）、NFR-P2（抖动）
对应文档：《测试与验收.md》§3 验收标准、§4 效果评测方案

**这个脚本存在的意义**：把散落在多个实验脚本里的指标，一次性跑完并
汇总成一份可复跑的验收报告。它不产生新的测量方法，只是把已有的三路
测量**编排在一起**：

  1. 延迟（`benchmark_latency` 的同一套打点）→ NFR-P1 / AC-N1
  2. 抖动（防抖前后对照）              → FR-06 / AC-06 / AC-N2
  3. 构图评分相关性 SRCC                → FR-03 / AC-03

**关于"诚实"的设计约束**（贯穿本项目）：
  - 报告里每个数字都标注 **来源**（哪个脚本、哪份原始 JSON）；
  - 未达标的项**如实标 FAIL**，不调参凑数；
  - 样本量不足的指标（如 SRCC n=5）**强制打上"不具统计效力"**；
  - 所有"简化处"在报告末尾集中列出，不藏在正文里。

用法::

    python scripts/acceptance_report.py                  # 全套（约 1~2 分钟）
    python scripts/acceptance_report.py --quick          # 跳过 SRCC（无模型依赖）
    python scripts/acceptance_report.py --backend rule   # 规则后端跑抖动
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aicg.settings import PROJECT_ROOT  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"

# 单帧预算：目标 3 FPS
FRAME_BUDGET_MS = 1000.0 / 3.0


# --------------------------------------------------------------------------
# 结果容器
# --------------------------------------------------------------------------
@dataclass
class CheckResult:
    """一条验收项的结论。"""

    ac_id: str
    """验收编号（对应《测试与验收.md》）。"""

    name: str
    status: str  # PASS / FAIL / WARN / INFO
    detail: str
    value: dict = field(default_factory=dict)


def _run(cmd: list[str], *, timeout: int = 600) -> tuple[int, str]:
    """跑一个子进程脚本，返回 (returncode, stdout+stderr)。"""
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace"
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _extract_summary(text: str) -> dict | None:
    """从脚本输出里抓 ``SUMMARY_JSON {...}`` 行。"""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("SUMMARY_JSON"):
            try:
                return json.loads(line[len("SUMMARY_JSON") :].strip())
            except json.JSONDecodeError:
                return None
    return None


def _latest(pattern: str) -> Path | None:
    """取 outputs/reports 下匹配 pattern 的最新文件。"""
    files = sorted(REPORTS_DIR.glob(pattern))
    return files[-1] if files else None


def _load(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------
# 三路测量
# --------------------------------------------------------------------------
def measure_latency(py: str, backend: str) -> tuple[CheckResult, dict]:
    """延迟验收（NFR-P1 / AC-N1）。"""
    print("  [1/3] 延迟基准 ...", flush=True)
    before = _latest("benchmark_*.json")
    rc, out = _run(
        [py, str(PROJECT_ROOT / "scripts" / "benchmark_latency.py"),
         "--backend", backend, "--repeat", "3", "--frames", "40"]
    )
    path = _latest("benchmark_*.json")
    if path == before:  # 脚本失败，没有新报告
        return (
            CheckResult("AC-N1", "端到端引导延迟", "FAIL", f"基准脚本执行失败：{out[-300:]}"),
            {},
        )
    data = _load(path) or {}
    total = data.get("stages", {}).get("total", {})
    p95 = float(total.get("p95", 0.0))
    mean = float(total.get("mean", 0.0))
    ratio = p95 / FRAME_BUDGET_MS * 100.0

    status = "PASS" if p95 <= FRAME_BUDGET_MS else "FAIL"
    # 瓶颈定位
    stages = data.get("stages", {})
    bottleneck = max(
        ((k, v.get("mean", 0.0)) for k, v in stages.items() if k != "total"),
        key=lambda kv: kv[1],
        default=("-", 0.0),
    )
    detail = (
        f"P95={p95:.1f}ms（预算 {FRAME_BUDGET_MS:.0f}ms，占 {ratio:.1f}%）；"
        f"均值 {mean:.1f}ms；瓶颈 {bottleneck[0]}"
    )
    return (
        CheckResult(
            "AC-N1", "端到端引导延迟", status, detail,
            {
                "p50_ms": total.get("p50"), "p95_ms": p95, "p99_ms": total.get("p99"),
                "mean_ms": mean, "budget_ms": FRAME_BUDGET_MS, "p95_budget_ratio_pct": round(ratio, 1),
                "first_frame_ms": data.get("first_frame_ms"),
                "bottleneck": bottleneck[0], "source": path.name, "backend": backend,
            },
        ),
        data,
    )


def measure_debounce(py: str, backend: str, source: str) -> tuple[CheckResult, dict]:
    """抖动验收（FR-06 / AC-06 / AC-N2）。"""
    print(f"  [2/3] 防抖对照（{Path(source).name}, backend={backend}）...", flush=True)
    src = FIXTURES / source
    if not src.exists():
        return CheckResult("AC-06", "指令抖动抑制", "FAIL", f"夹具缺失：{src}"), {}
    rc, out = _run(
        [py, str(PROJECT_ROOT / "scripts" / "run_demo.py"),
         "--source", str(src), "--backend", backend, "--compare", "--no-render",
         "--out", str(REPORTS_DIR)],
        timeout=600,
    )
    data = _load(_latest(f"debounce_{Path(source).stem}_*.json")) or {}
    if not data:
        return CheckResult("AC-06", "指令抖动抑制", "FAIL", f"对照脚本失败：{out[-300:]}"), {}

    red = float(data.get("switch_reduction_pct", 0.0))
    off = data.get("off", {}).get("switch_stats", {})
    on = data.get("on", {}).get("switch_stats", {})
    run_before = float(data.get("mean_run_frames_before", 0.0))
    run_after = float(data.get("mean_run_frames_after", 0.0))
    gain = (run_after / run_before) if run_before > 0 else 0.0

    # 本项目自定门槛：降幅 >= 50% 视为有效（基线由实测确定，见文档）
    status = "PASS" if red >= 50.0 else "WARN"
    detail = (
        f"切换 {off.get('switch_count'):.0f} → {on.get('switch_count'):.0f} 次/240帧，"
        f"降幅 {red:.2f}%；平均持续帧数 {run_before:.2f} → {run_after:.2f}（{gain:.1f}×）"
    )
    return (
        CheckResult(
            "AC-06", "指令抖动抑制", status, detail,
            {
                "frames": data.get("frames"),
                "switches_off": off.get("switch_count"),
                "switches_on": on.get("switch_count"),
                "switches_per_min_off": off.get("switches_per_minute"),
                "switches_per_min_on": on.get("switches_per_minute"),
                "reduction_pct": red,
                "mean_run_frames_before": run_before,
                "mean_run_frames_after": run_after,
                "mean_run_gain_x": round(gain, 2),
                "source_fixture": source, "backend": backend,
                "source": (_latest(f"debounce_{Path(source).stem}_*.json") or Path("-")).name,
            },
        ),
        data,
    )


def measure_srcc(py: str, backend: str) -> tuple[CheckResult, dict]:
    """构图评分相关性（FR-03 / AC-03）。

    评测集有两套，**优先用真实素材集**：

    - ``configs/eval/annotations_real.json``（n≈47，真实拍摄整图，
      原则驱动标注）—— 样本量达标，结论具备统计效力；
    - ``configs/eval/annotations.json``（n=5，裁切构造）—— 仅保留用于
      验证评测链路本身可用，**不得作为结论**。

    若真实集不存在（例如未执行 fetch 脚本），自动回退到构造集并
    在报告中标注，而不是假装指标已达标。
    """
    real_ann = PROJECT_ROOT / "configs" / "eval" / "annotations_real.json"
    if real_ann.exists():
        ann_path = real_ann
        out_json = PROJECT_ROOT / "outputs" / "reports" / "srcc_real.json"
        label = "真实素材集"
    else:
        ann_path = PROJECT_ROOT / "configs" / "eval" / "annotations.json"
        out_json = PROJECT_ROOT / "outputs" / "reports" / "srcc_evaluation.json"
        label = "构造集（回退）"

    print(f"  [3/3] 构图评分 SRCC（{label}）...", flush=True)
    rc, out = _run(
        [
            py,
            str(PROJECT_ROOT / "scripts" / "evaluate_composition.py"),
            "--annotations",
            str(ann_path),
            "--backend",
            backend,
            "--out",
            str(out_json),
        ]
    )
    data = _load(out_json) or {}
    if not data:
        return CheckResult("AC-03", "构图评分相关性", "FAIL", f"评测脚本失败：{out[-300:]}"), {}

    srcc = float(data.get("srcc", 0.0))
    n = int(data.get("n_samples", 0))
    meaningful = bool(data.get("statistically_meaningful", False))
    method = data.get("annotation_method", "")

    if not meaningful:
        status, detail = (
            "WARN",
            f"SRCC={srcc:+.4f}（n={n}，样本量不足，**不具统计效力**，仅证明脚本可用）",
        )
    else:
        status = "PASS" if srcc >= 0.5 else "FAIL"
        note = "，原则驱动标注" if method == "principle_based" else ""
        detail = f"SRCC={srcc:+.4f}（n={n}{note}）"
    return (
        CheckResult(
            "AC-03", "构图评分相关性", status, detail,
            {
                "srcc": srcc, "n_samples": n, "statistically_meaningful": meaningful,
                "min_meaningful_n": data.get("min_meaningful_n"),
                "backend": backend,
                "annotation_set": label,
                "annotation_method": method,
            },
        ),
        data,
    )


# --------------------------------------------------------------------------
# 报告渲染
# --------------------------------------------------------------------------
_STATUS_MARK = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "INFO": "INFO"}


def render(results: list[CheckResult], *, generated_at: str, env: dict) -> str:
    lines: list[str] = []
    w = 72
    lines.append("=" * w)
    lines.append("  AI 实时构图指导 Agent —— 端到端验收报告")
    lines.append("=" * w)
    lines.append(f"  生成时间 : {generated_at}")
    lines.append(f"  硬件/环境: {env.get('device', '-')} / torch {env.get('torch', '-')}")
    lines.append(f"  单帧预算 : {FRAME_BUDGET_MS:.0f} ms（目标 3 FPS）")
    lines.append("")

    passed = sum(1 for r in results if r.status == "PASS")
    warned = sum(1 for r in results if r.status == "WARN")
    failed = sum(1 for r in results if r.status == "FAIL")
    lines.append(f"  结论汇总 : PASS {passed} / WARN {warned} / FAIL {failed}")
    lines.append("")
    lines.append("-" * w)
    lines.append(f"  {'编号':<10}{'验收项':<20}{'结论':<6}详情")
    lines.append("-" * w)
    for r in results:
        lines.append(f"  {r.ac_id:<10}{r.name:<20}{_STATUS_MARK.get(r.status, r.status):<6}")
        # 详情折行
        for chunk in _wrap(r.detail, w - 36):
            lines.append(f"  {'':<34}{chunk}")
    lines.append("-" * w)
    lines.append("")
    lines.append("  原始数据（可复跑，脚本见 scripts/）")
    for r in results:
        if r.value.get("source"):
            lines.append(f"    {r.ac_id} → outputs/reports/{r.value['source']}")
    lines.append("")
    lines.append("  ⚠ 简化处与生产差距（必读，详见《测试与验收.md》§5）")
    for s in SIMPLIFICATIONS:
        lines.append(f"    · {s}")
    lines.append("=" * w)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    """按显示宽度折行（中文字符按 2 计）。"""

    def dw(s: str) -> int:
        return sum(2 if ord(c) > 0x2E80 else 1 for c in s)

    out, cur = [], ""
    for ch in text:
        if dw(cur + ch) > width:
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out or [""]


SIMPLIFICATIONS = [
    "录屏模拟实时：生产应为真机 Camera + WebRTC，可测真实端到端延迟（本报告延迟非真机数据）",
    "单目 FOV 公式估距：生产应上 Depth-Anything/MiDaS 深度图或双摄/LiDAR",
    "构图评分用显式启发式规则：SAMPNet/CADB 权重未接入，仅保留可替换接口",
    "语言层已接真实模型（keypool 代理池），但**本报告的延迟指标不含语言层**："
    "语言层在冷路径（按快门后），实测 glm-5.3 单次约 7~15s；"
    "热路径（每帧打分）仍是规则法，P95 约 28~37ms。两者不可混为一谈",
    "语言层依赖本机 keypool 进程：若未启动则自动退回模板兜底"
    "（is_fallback=True，流程不中断），本报告未覆盖该降级路径的端到端耗时",
    "SRCC 标注为「原则驱动」而非多人主观评分：仅证明评分器与构图学原则同向，"
    "不能表述为「符合人类审美」；抽检表已生成待补",
    "headroom 分项与主体大小存在共线（r=-0.83）：总 SRCC 含共线放大，"
    "应以偏相关 +0.82 作为更保守读数",
    "balance 分项有效性弱（与原则 SRCC 仅 +0.15）：该维度待改进",
    "感知层会把部分可见人体扩张至近全幅：导致「主体占比」维度在裁切夹具上被抹平",
    "候选框空间在主体过大时会坍缩（candidate_count=1）：此时构图评估标记 degraded，建议仅供参考",
    "演示素材受合规图源能力限制：可用图源无主题检索能力，"
    "可用素材 34 张（14 真实照片 + 20 AI 人脸），无法交付『咖啡馆/街拍』等主题化素材库",
]


def measure_doc_consistency(py: str) -> CheckResult:
    """AC-DOC：文档口径一致性（死链 + 易漂移事实取值）。

    项目里反复出现「改了源头、漏改副本」的缺陷（延迟最优值、API 路径、
    测试用例数、缺陷文档重命名），单靠人眼核对不可靠，故纳入验收门禁。
    """
    script = PROJECT_ROOT / "scripts" / "check_doc_consistency.py"
    if not script.exists():
        return CheckResult("AC-DOC", "文档口径一致性", "INFO", "校验脚本不存在，跳过")
    code, out = _run([py, str(script), "--quiet"])
    if code == 0:
        return CheckResult("AC-DOC", "文档口径一致性", "PASS",
                           "行内路径无死链；易漂移事实取值全部在合法集合内")
    # 取前几条问题作为证据
    lines = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("✗")]
    detail = "；".join(lines[:3]) if lines else "详见 check_doc_consistency.py 输出"
    if len(lines) > 3:
        detail += f" …（共 {len(lines)} 条）"
    return CheckResult("AC-DOC", "文档口径一致性", "FAIL", detail)


def measure_docker_static(py: str) -> CheckResult:
    """AC-DOCKER：Docker 编排静态校验。

    Dockerfile/compose 从未实测构建（开发机 daemon 未运行），
    因此用静态校验兜底，避免"必然启动失败"的错误潜伏到交付。
    """
    script = PROJECT_ROOT / "scripts" / "check_docker_static.py"
    if not script.exists():
        return CheckResult("AC-DOCKER", "Docker 编排静态校验", "INFO", "校验脚本不存在，跳过")
    code, out = _run([py, str(script)])
    if code == 0:
        return CheckResult("AC-DOCKER", "Docker 编排静态校验", "PASS",
                           "入口/COPY 源/构建参数/命令依赖/挂载/ignore 规则均无必然失败项"
                           "（注：**未实测构建**，仅静态校验）")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("[FAIL]")]
    detail = "；".join(lines[:2]) if lines else "详见 check_docker_static.py 输出"
    if len(lines) > 2:
        detail += f" …（共 {len(lines)} 条）"
    return CheckResult("AC-DOCKER", "Docker 编排静态校验", "FAIL", detail)


def measure_ci_static(py: str) -> CheckResult:
    """AC-CI：CI 工作流静态校验。

    `.github/workflows/*.yml` 在本机无法执行（需要 GitHub runner），
    属于"未实测交付物"。静态校验抓必然失败项：YAML 结构、引用脚本与
    参数是否存在（曾抓出 `run_demo.py --limit` 应为 `--max-frames`）。
    """
    script = PROJECT_ROOT / "scripts" / "check_ci_static.py"
    if not script.exists():
        return CheckResult("AC-CI", "CI 工作流静态校验", "INFO", "校验脚本不存在，跳过")
    code, out = _run([py, str(script)])
    if code == 0:
        return CheckResult("AC-CI", "CI 工作流静态校验", "PASS",
                           "YAML 结构 / 脚本引用 / 参数签名均无必然失败项"
                           "（注：**workflow 未在 runner 上实测运行**）")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("[FAIL]")]
    detail = "；".join(lines[:2]) if lines else "详见 check_ci_static.py 输出"
    if len(lines) > 2:
        detail += f" …（共 {len(lines)} 条）"
    return CheckResult("AC-CI", "CI 工作流静态校验", "FAIL", detail)


def main() -> int:
    ap = argparse.ArgumentParser(description="端到端验收报告（NFR-O3）")
    ap.add_argument("--backend", default="yolo", choices=["yolo", "rule", "auto"],
                    help="感知后端；抖动默认用夹具自带提示（合成→rule，真实→yolo）")
    ap.add_argument("--debounce-source", default="handheld_jitter.mp4",
                    help="抖动对照所用夹具（默认合成夹具，配合 rule 后端）")
    ap.add_argument("--debounce-backend", default=None,
                    help="抖动测量后端（默认自动：合成夹具→rule，真实素材→yolo）")
    ap.add_argument("--quick", action="store_true", help="跳过 SRCC（不需要模型）")
    ap.add_argument("--json", action="store_true", help="额外输出机器可读 JSON")
    args = ap.parse_args()

    py = sys.executable
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 抖动后端：合成夹具必须用 rule（YOLO 检测不到合成人体，已在夹具文档说明）
    deb_backend = args.debounce_backend
    if deb_backend is None:
        deb_backend = "rule" if "jitter" in args.debounce_source or "walk" in args.debounce_source else "yolo"

    print("开始端到端验收 ...", flush=True)
    t0 = time.time()

    results: list[CheckResult] = []
    lat, lat_data = measure_latency(py, args.backend)
    results.append(lat)
    deb, _ = measure_debounce(py, deb_backend, args.debounce_source)
    results.append(deb)
    if not args.quick:
        srcc, _ = measure_srcc(py, args.backend)
        results.append(srcc)
    else:
        results.append(
            CheckResult("AC-03", "构图评分相关性", "INFO", "已按 --quick 跳过")
        )

    # 文档口径一致性：与性能无关，但同属"可交付质量"门禁
    results.append(measure_doc_consistency(py))
    # Docker 编排静态校验：替代"未实测构建"的空白
    results.append(measure_docker_static(py))
    # CI 工作流静态校验：workflow 无法本地运行，静态兜底
    results.append(measure_ci_static(py))

    # 环境快照
    env: dict = {"device": "cpu"}
    try:
        import torch  # type: ignore

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["device"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        env["torch"] = "-"

    generated_at = datetime.now().isoformat(timespec="seconds")
    report = render(results, generated_at=generated_at, env=env)
    print()
    print(report)

    out_path = REPORTS_DIR / f"acceptance_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    payload = {
        "generated_at": generated_at,
        "elapsed_s": round(time.time() - t0, 2),
        "env": env,
        "frame_budget_ms": FRAME_BUDGET_MS,
        "checks": [
            {"ac_id": r.ac_id, "name": r.name, "status": r.status,
             "detail": r.detail, "value": r.value}
            for r in results
        ],
        "simplifications": SIMPLIFICATIONS,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已保存: {out_path}")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))

    # 有任何 FAIL 则返回非零，便于 CI 卡口
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
