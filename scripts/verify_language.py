#!/usr/bin/env python
"""语言层真实模型接入验证：跑通「感知 → 构图 → 真实 LLM 解说」全链路。

对应需求：FR-07 / NFR-E3
对应文档：《技术方案.md》§2.4

**为什么单独做这个脚本**

单元测试里语言层走 mock（不能依赖外部服务），因此"真实模型到底接没接通"
缺少一个可复跑的验证入口。本脚本就是那个入口：
它拿真实照片跑完整链路，把真实模型的解说打印出来，
并**如实标出是否降级、实际用了哪个模型、耗时与 token 消耗**。

用法::

    python scripts/verify_language.py                      # 用默认照片
    python scripts/verify_language.py --image <path>       # 指定照片
    python scripts/verify_language.py --model glm-5.3      # 强制某模型
    python scripts/verify_language.py --count 3            # 多张照片
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.composition.scorer import HeuristicCompositionScorer  # noqa: E402
from aicg.language.model_registry import REGISTRY  # noqa: E402
from aicg.language.vlm_client import VlmClient  # noqa: E402
from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.perception import perception_from_settings  # noqa: E402
from aicg.schemas import (  # noqa: E402
    ActionCommand,
    ActionType,
    CompositionResult,
    FrameSnapshot,
    PerceptionResult,
    StabilizedCommand,
)
from aicg.settings import PROJECT_ROOT, load_settings  # noqa: E402
from aicg.utils.image import resize_keep_aspect  # noqa: E402

log = get_logger("scripts.verify_language")


def read_image(p: Path) -> np.ndarray | None:
    img = cv2.imread(str(p))
    if img is None:
        try:
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            return None
    return img


def build_snapshot(perception, scorer, frame_id: int, img: np.ndarray) -> FrameSnapshot | None:
    img = resize_keep_aspect(img, 480)
    h, w = img.shape[:2]
    perc: PerceptionResult = perception.infer(img, frame_id, 0)
    subj = perc.primary_subject
    if subj is None:
        return None

    sal = perc.extras.get("saliency_map")
    comp: CompositionResult = scorer.score_frame(
        subject_bbox=subj.bbox,
        saliency=sal if isinstance(sal, np.ndarray) else None,
        frame_shape=(h, w),
    )
    raw = ActionCommand(
        action=ActionType.HOLD,
        magnitude_text="保持当前构图",
        magnitude_raw=1.0,
        confidence=1.0,
    )
    cmd = StabilizedCommand(command=raw, raw_command=raw, is_changed=False)
    return FrameSnapshot(
        frame_id=frame_id,
        timestamp_ms=frame_id * 333,
        frame_size=(h, w),
        perception=perc,
        composition=comp,
        command=cmd,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="语言层真实模型接入验证")
    ap.add_argument("--image", default=None, help="单张照片路径")
    ap.add_argument("--dir", default=str(PROJECT_ROOT / "configs" / "eval" / "images_real"))
    ap.add_argument("--count", type=int, default=2, help="默认取几张")
    ap.add_argument("--model", default=None, help="强制模型（跳过链路降级）")
    ap.add_argument("--provider", default=None, help="覆盖 provider（mock/keypool）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup_logging("WARNING")

    overrides = {}
    if args.provider:
        overrides["language.provider"] = args.provider
    cfg = load_settings(overrides=overrides or None)

    print("=" * 72)
    print("  语言层真实模型接入验证")
    print("=" * 72)
    print(f"  provider   : {cfg.language.provider}")
    print(f"  max_tokens : {cfg.language.max_tokens}")
    print(f"  强制模型   : {args.model or '（按角色自动选型）'}")

    vlm = VlmClient(cfg.language)
    print(f"  enabled    : {vlm.enabled}")
    if vlm._kp is not None:
        print(f"  keypool    : {vlm._kp.status()}")
    print()

    print("  已登记模型（实测量）")
    for m in REGISTRY:
        flag = "OK " if m.probe_ok else "NG "
        print(f"    [{flag}] {m.id:<26} roles={','.join(m.roles) or '-'}")
    print()

    # 准备照片
    if args.image:
        paths = [Path(args.image)]
    else:
        d = Path(args.dir)
        paths = sorted(p for p in d.glob("*.jpg"))[: args.count]
    if not paths:
        print(f"[错误] 未找到可用照片：{args.dir}")
        return 2

    perc_cfg = load_settings(overrides={"perception.backend": "yolo"})
    perception = perception_from_settings(perc_cfg)
    perception.warmup((360, 480))
    scorer = HeuristicCompositionScorer(cfg.composition.scoring)

    results = []
    print("-" * 72)
    for i, p in enumerate(paths, 1):
        img = read_image(p)
        if img is None:
            print(f"  [{i}] {p.name}  读取失败，跳过")
            continue
        snap = build_snapshot(perception, scorer, i, img)
        if snap is None:
            print(f"  [{i}] {p.name}  未检出主体，跳过")
            continue

        comp = snap.composition
        print(f"  [{i}] {p.name}")
        print(f"      感知: {snap.perception.backend} / 构图分 {comp.composition_score:.1f} / "
              f"{comp.pattern_label or comp.pattern.value} / {comp.shot_size_label}")

        if args.model:
            # 强制模型：直接调客户端以观察真实降级行为
            from aicg.language.vlm_client import build_facts  # noqa: PLC0415

            import yaml  # noqa: PLC0415

            facts = build_facts(snap)
            res = vlm._kp.chat(  # noqa: SLF001
                [
                    {"role": "system", "content": vlm.assets.system_prompt},
                    {"role": "user", "content": "画面分析结果：\n" + yaml.safe_dump(
                        facts, allow_unicode=True, sort_keys=False) +
                        "\n请基于以上分析结果，生成 1~2 句构图理念解说。"},
                ],
                model=args.model,
                max_tokens=cfg.language.max_tokens,
                temperature=cfg.language.temperature,
            )
            narr = {
                "text": res.text,
                "is_fallback": not res.ok,
                "model": res.model or args.model,
                "ok": res.ok,
                "degraded": res.degraded,
                "latency_ms": round(res.latency_ms, 1),
                "usage": res.usage,
                "reason": res.reason,
            }
        else:
            nr = vlm.narrate(snap)
            narr = {
                "text": nr.text,
                "is_fallback": nr.is_fallback,
                "model": nr.model,
                "ok": not nr.is_fallback,
                "degraded": nr.extras.get("degraded_from_chain", False),
                "latency_ms": nr.extras.get("latency_ms"),
                "usage": nr.token_usage.model_dump() if nr.token_usage else None,
                "reason": nr.extras.get("fallback_note", ""),
            }

        tag = "模板兜底" if narr["is_fallback"] else (
            f"真实模型 {narr['model']}" + ("（降级）" if narr.get("degraded") else "")
        )
        print(f"      语言: {tag}")
        print(f"      解说: {narr['text']}")
        if narr.get("latency_ms"):
            print(f"      耗时: {narr['latency_ms']}ms")
        if narr.get("usage"):
            u = narr["usage"]
            print(f"      token: in={u.get('prompt_tokens')} out={u.get('completion_tokens')}")
        if narr.get("reason"):
            print(f"      原因: {narr['reason']}")
        print()
        results.append({"image": p.name, "composition_score": comp.composition_score, **narr})

    ok = sum(1 for r in results if r["ok"])
    print("-" * 72)
    print(f"  结果: {ok}/{len(results)} 走通真实模型")
    if ok == 0 and results:
        print("  [提示] 全部走模板兜底。排查：")
        print("         1) keypool 是否启动（curl http://127.0.0.1:8790/health）")
        print("         2) provider 是否为 keypool")
        print("         3) 看上方'原因'字段的具体错误")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  报告 -> {outp}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
