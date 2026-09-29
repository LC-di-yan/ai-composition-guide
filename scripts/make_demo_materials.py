"""生成演示录屏与作品集材料（Task #17）。

对应需求：FR-05（实时引导循环）、FR-06（指令防抖）、NFR-P1（延迟）
对应文档：《开发计划.md》M3/M5 出口条件

**为什么需要一个专门的脚本而不是手敲命令**

演示材料最容易出的问题不是"跑不出来"，而是**跑出一个看起来很好的假结果**。
本次实操就撞到两个：

1. ``real_photo_zoom.mp4`` 全程 120 帧都判 ``move_back``、切换 0 次。
   单看数字像是"极其稳定"（0 次切换！），实际是**退化结果**——
   实测该素材主体占高恒在 0.718~0.999（均值 0.833），**从来没有**进入
   保持窗口（0.4675~0.6325），因此"一直该后退"本就是正确答案，
   0 次切换是"没什么可切"，不是"防抖做得好"。**把这张图当稳定性证据是误导。**

2. ``walk_towards.mp4`` 的防抖对照降幅 **0.00%**。
   看着像"防抖没生效"，实际也不对：把原始动作序列打出来只有 3 段
   （``closer×34, up×13, back×43``），**本来就干净得没什么可抑制**。
   真相是：**这个素材不适合证明防抖**，而不是防抖失效。

因此本脚本的核心职责是**按用途分流素材**，并在产物里写清楚
"哪张图能证明什么、哪张图不能"：

===============  ==========================  ==================================
素材             用途                        为什么
===============  ==========================  ==================================
walk_towards     演示视频（展示指令变化）      原始序列有真实弧线：该靠近→该抬
                                             →该后退，共 3 段，观感自然
handheld_jitter  防抖证据（AC-06 唯一基准）    240 帧、原生抖动 84 段，
                                            官方基线 84→19（77.38%）
real_photo_zoom  yolo 链路验证（非演示）      真实照片推拉，检验 yolo 后端；
                                            但动作恒定，**不可作稳定性证据**
===============  ==========================  ==================================

用法::

    python scripts/make_demo_materials.py                 # 生成全套
    python scripts/make_demo_materials.py --skip-video    # 只出报告与清单
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUT_DEMO = PROJECT_ROOT / "outputs" / "demo"

# 素材 → 用途映射。新增素材时必须在此登记，否则脚本会拒绝生成——
# 防止"随手加个视频就当成演示"这种会让作品集失真的做法。
MATERIALS = {
    "walk_towards.mp4": {
        "role": "demo_video",
        "backend": "rule",
        "can_prove": ["指令变化可见（靠近→抬升→后退）", "单帧延迟"],
        "cannot_prove": ["防抖收益（原始序列仅 3 段，无抖动可抑制）"],
        "note": "演示主视频。原始动作序列 closer×34 / up×13 / back×43。",
    },
    "handheld_jitter.mp4": {
        "role": "debounce_evidence",
        "backend": "rule",
        "can_prove": ["AC-06 抖动抑制（官方基线 84→19 次/240帧，降幅 77.38%）", "单帧延迟"],
        "cannot_prove": ["动作方向的语义正确性（合成素材，非真实人像）"],
        "note": "防抖证据唯一基准素材。240 帧原生抖动 84 段。",
    },
    "real_photo_zoom.mp4": {
        "role": "yolo_check",
        "backend": "yolo",
        "can_prove": ["yolo 后端在真实照片上可跑通", "接近真实人像的主体占比"],
        "cannot_prove": [
            "稳定性（全 120 帧同一动作，0 切换是退化结果而非防抖功劳）",
            "指令变化（主体占高恒在 0.718~0.999，从未进入保持窗口）",
        ],
        "note": "用于验证 yolo 链路，**不可作为演示或稳定性证据**。",
    },
}


def _run_demo(src: Path, backend: str, compare: bool) -> dict:
    """调用 run_demo.py 并解析其 SUMMARY_JSON 行。"""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_demo.py"),
        "--source",
        str(src),
        "--backend",
        backend,
    ]
    if compare:
        cmd.append("--compare")

    # 注意：不能写 dict(**os.environ, PYTHONPATH=...)。
    # 本项目运行时 PYTHONPATH 已由外部设过（且本次实操确实设了），
    # 此时 `**os.environ` 已含该键，再显式传会造成
    # "dict() got multiple values for keyword argument 'PYTHONPATH'"。
    # 正确做法是先拷贝再赋值。
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env, timeout=900)
    out = proc.stdout + "\n" + proc.stderr
    summary = None
    for line in out.splitlines():
        if line.startswith("SUMMARY_JSON "):
            try:
                summary = json.loads(line[len("SUMMARY_JSON "):])
            except json.JSONDecodeError:
                pass
    return {"returncode": proc.returncode, "summary": summary, "log_tail": out.strip().splitlines()[-6:]}


def _latency_of(src_name: str) -> dict:
    """读最新一次该素材的延迟报告。"""
    stem = Path(src_name).stem
    cands = sorted(OUT_REPORTS.glob(f"latency_{stem}_*.json"))
    if not cands:
        return {}
    return json.loads(cands[-1].read_text(encoding="utf-8"))


def _debounce_of(src_name: str) -> dict:
    stem = Path(src_name).stem
    cands = sorted(OUT_REPORTS.glob(f"debounce_{stem}_*.json"))
    if not cands:
        return {}
    return json.loads(cands[-1].read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成演示录屏与作品集材料")
    ap.add_argument("--skip-video", action="store_true", help="跳过视频渲染（只出报告）")
    ap.add_argument("--only", default=None, help="只跑指定素材文件名")
    args = ap.parse_args(argv)

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_DEMO.mkdir(parents=True, exist_ok=True)

    names = [args.only] if args.only else list(MATERIALS)
    results: dict[str, dict] = {}

    for name in names:
        spec = MATERIALS.get(name)
        if spec is None:
            print(f"[拒绝] {name} 未在 MATERIALS 登记其用途。"
                  f"新增素材必须显式说明\"能证明什么 / 不能证明什么\"，否则演示材料会失真。",
                  file=sys.stderr)
            continue
        src = PROJECT_ROOT / "tests" / "fixtures" / name
        if not src.exists():
            print(f"[跳过] 素材不存在: {src}", file=sys.stderr)
            continue

        print(f"\n=== {name}  (用途: {spec['role']}, 后端: {spec['backend']}) ===")
        if args.skip_video:
            results[name] = {"summary": None, "skipped": True}
            continue
        r = _run_demo(src, spec["backend"], compare=True)
        results[name] = r
        s = r.get("summary") or {}
        print(f"  帧数={s.get('frames')}  P95={s.get('latency_p95_ms')}ms  "
              f"切换/分={s.get('switches_per_minute')}")
        print(f"  视频={Path(s.get('video','')).name if s.get('video') else '(未产出)'}")
        for line in r.get("log_tail", [])[-3:]:
            print("  |", line)

    # ---- 汇总清单（含诚实的"不可证明"标注）----
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest = {
        "generated_at": stamp,
        "materials": {},
        "honesty_notes": [
            "real_photo_zoom.mp4 全 120 帧同一动作、0 次切换，是**退化结果**："
            "该素材主体占高恒在 0.718~0.999，从未进入保持窗口 0.4675~0.6325，"
            "因此\"一直该后退\"本就是正确答案。**不得当作稳定性证据。**",
            "walk_towards.mp4 的防抖对照降幅为 0.00%，因为其原始动作序列只有 3 段"
            "（closer×34 / up×13 / back×43），**本就没有抖动可抑制**。"
            "该素材适合做演示视频，**不适合证明 FR-06**。",
            "FR-06 的证据应引用 handheld_jitter.mp4（240 帧、原生 84 段抖动），"
            "官方基线 84→19 次（降幅 77.38%）。",
        ],
    }
    for name, spec in MATERIALS.items():
        if name not in results:
            continue
        lat = _latency_of(name)
        deb = _debounce_of(name)
        manifest["materials"][name] = {
            **{k: spec[k] for k in ("role", "backend", "can_prove", "cannot_prove", "note")},
            "summary": results[name].get("summary"),
            "latency_p95_ms": (lat.get("stages", {}) or {}).get("total", {}).get("p95"),
            "switch_reduction_pct": deb.get("switch_reduction_pct"),
        }

    mp = OUT_REPORTS / f"demo_materials_manifest_{stamp}.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[清单] {mp.relative_to(PROJECT_ROOT)}")

    # 更新一份稳定的"当前演示材料"软指针，便于文档引用而不必写时间戳
    latest = OUT_REPORTS / "demo_materials_manifest.json"
    shutil.copyfile(mp, latest)
    print(f"[清单] {latest.relative_to(PROJECT_ROOT)}（稳定指针）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
