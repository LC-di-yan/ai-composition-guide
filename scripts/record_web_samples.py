#!/usr/bin/env python
"""从**真实运行中的 API** 录制逐帧响应，作为 Web 演示页的回放样本。

对应需求：FR-11（演示）/ NFR-O3（可复现）

**为什么必须真的调 API，而不是手写样本**

Web 演示页有两种模式：演示模式（离线回放）与联机模式（真连后端）。
如果演示模式的样本是**手写的**，它就会慢慢和后端真实响应脱节——
字段改名、新增字段、数值范围变化都不会被发现，页面看起来一切正常，
实际上展示的是一个**不存在的系统**。

因此本脚本把真实响应录下来，落到 ``web/samples/*.json``。
回放样本与联机响应**同源同构**，页面渲染代码也只有一条路径。

**为什么用本项目的主题素材**

``assets/topic_photos/``（23 张按场景采集的「女生拍照」素材）
是真实可读的人像照片，主体明确、场景多样，比合成夹具更适合演示。

用法::

    python scripts/record_web_samples.py --port 8010 --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402

from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.settings import PROJECT_ROOT  # noqa: E402

log = get_logger("scripts.record_web_samples")

MAX_EDGE = 1100          # 上传前压到 1100px：够看清新构图，又不至于 base64 过大
JPEG_Q = 88


def imread_any(p: Path) -> np.ndarray | None:
    """读图，兼容非 ASCII 路径。"""
    if not p.exists():
        return None
    img = cv2.imread(str(p))
    if img is not None:
        return img
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except Exception:  # noqa: BLE001
        return None


def encode(img: np.ndarray) -> str:
    h, w = img.shape[:2]
    s = min(1.0, MAX_EDGE / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description="录制 Web 演示页回放样本")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--limit", type=int, default=8, help="最多录多少帧")
    ap.add_argument("--lang", action="store_true",
                    help="同时录制语言层解说（慢，每帧数秒）")
    ap.add_argument("--order", choices=["interleave", "scale"], default="scale",
                    help="帧序组织方式。interleave=按主题交错（场景多样，"
                         "但相邻帧主体大小跳变，会触发防抖的连续帧要求）；"
                         "scale=按实测主体占高排序（**模拟真实拍摄时主体"
                         "连续变大/变小的过程**，默认）。")
    args = ap.parse_args()
    setup_logging("INFO")

    base = f"http://127.0.0.1:{args.port}"
    # **必须绕过系统代理**：本机 127.0.0.1 请求走代理会被拒（实测 502 / 10061）
    cli = httpx.Client(proxy=None, trust_env=False, timeout=120.0)

    try:
        r = cli.get(f"{base}/readyz")
        print(f"  后端就绪: {r.status_code} {r.text.strip()[:110]}")
    except Exception as e:  # noqa: BLE001
        print(f"[错误] 连不上 {base}：{type(e).__name__}: {e}")
        print("       请先启动： PYTHONPATH=src python -m aicg.cli serve --port "
              f"{args.port}")
        return 1

    # 选样本：优先主题素材（真人真场景）
    pool: list[Path] = []
    idx_path = PROJECT_ROOT / "assets" / "topic_photos" / "_index_curated.json"
    if idx_path.exists():
        recs = json.loads(idx_path.read_text(encoding="utf-8"))
        if args.order == "scale":
            # 按实测主体占高排序，模拟"同一被摄对象的连续景别变化"。
            #
            # **为什么需要这一条（D-11 复盘的直接教训）**：原先按主题交错
            # 取样，10 张互相无关的照片被当作连续视频推入，相邻帧的主体
            # 占高在 0.35~0.95 间随机跳变。这在真实拍摄中**不会发生**——
            # 人移动是连续的。随机的 raw 动作序列永远凑不出"连续 N 帧一致"，
            # 于是把防抖层的正常行为误判成了系统缺陷。
            # 测试序列必须尊重被测对象的物理连续性，否则测的是夹具的伪影。
            screen_path = PROJECT_ROOT / "outputs" / "reports" / "topic_photos_screen.json"
            ratios: dict[str, float] = {}
            if screen_path.exists():
                for row in json.loads(screen_path.read_text(encoding="utf-8")):
                    ratios[row["file"]] = float(row.get("subject_area_ratio") or 0.0)
            recs = sorted(recs, key=lambda r: ratios.get(r["file"], 0.0))
            pool = [PROJECT_ROOT / "assets" / "topic_photos" / r["file"]
                    for r in recs[: args.limit]]
        else:
            # 按主题分组后轮转取样，保证场景多样
            groups: dict[str, list[str]] = {}
            for rec in recs:
                groups.setdefault(rec["topic_requested"], []).append(rec["file"])
            cursor = {k: 0 for k in groups}
            while len(pool) < args.limit and any(
                    cursor[k] < len(v) for k, v in groups.items()):
                for k, v in groups.items():
                    if cursor[k] < len(v) and len(pool) < args.limit:
                        pool.append(PROJECT_ROOT / "assets" / "topic_photos" / v[cursor[k]])
                        cursor[k] += 1
    else:
        print(f"[警告] 未找到 {idx_path}，回退到 demo_photos")
        for p in sorted((PROJECT_ROOT / "assets" / "demo_photos").glob("*.jpg"))[:args.limit]:
            pool.append(p)

    if not pool:
        print("[错误] 没有可用素材")
        return 1

    # 每帧一个新 session 会更干净地复现"第 1 帧"状态，但我们要的是
    # **连续引导过程**（含防抖），所以共用一个 session 顺序推帧。
    r = cli.post(f"{base}/v1/session", json={})
    if r.status_code != 200:
        print(f"[错误] 创建会话失败 {r.status_code}: {r.text[:200]}")
        return 1
    sid = r.json()["session_id"]
    print(f"  会话: {sid}")

    out_dir = PROJECT_ROOT / "web" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  素材: {len(pool)} 张（来自 assets/topic_photos）")
    print()

    frames = []
    for i, p in enumerate(pool, 1):
        img = imread_any(p)
        if img is None:
            print(f"  [{i}] 跳过（读不了）{p.name}")
            continue
        h, w = img.shape[:2]
        body = {
            "session_id": sid,
            "frame_id": i,
            "timestamp_ms": (i - 1) * 333,
            "image_ref": encode(img),
            "persist_visual": False,
            "enable_language": bool(args.lang),
        }
        try:
            resp = cli.post(f"{base}/v1/frame", json=body)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] 请求异常 {type(e).__name__}: {e}")
            continue
        if resp.status_code != 200:
            print(f"  [{i}] HTTP {resp.status_code}: {resp.text[:160]}")
            continue
        j = resp.json()
        comp = j.get("composition") or {}
        cmd = j.get("command") or {}
        subj = j.get("subject") or {}
        # 帧图随样本一起存，回放时页面直接画，不需再读文件
        thumb, tbuf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        import base64
        rec = {
            "file": p.name,
            "topic": p.parent.name,
            "image_data_uri": "data:image/jpeg;base64," + base64.b64encode(tbuf.tobytes()).decode(),
            "width": w, "height": h,
            "response": j,
        }
        frames.append(rec)
        print(f"  [{i}] {p.name[:34]:34s} "
              f"score={comp.get('composition_score','—')} "
              f"action={cmd.get('action','—'):12s} "
              f"degraded={j.get('degraded')} "
              f"total={((j.get('latency') or {}).get('total_ms'))}ms")

    if not frames:
        print("[错误] 一帧都没录到")
        return 1

    out = out_dir / "frames.json"
    out.write_text(json.dumps(frames, ensure_ascii=False), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print()
    print(f"  已写入 {out}  ({len(frames)} 帧, {size_kb:.0f} KB)")

    # 顺手记一份"样本说明"，避免后人以为是手工编的
    summary = {
        "source": "真实 API 响应录制（非手写样本）",
        "how": "python scripts/record_web_samples.py",
        "api": f"POST {base}/v1/frame",
        "backend_perception": cli.get(f"{base}/readyz").json().get("perception_backend"),
        "language_recorded": bool(args.lang),
        "frames": len(frames),
        "assets": [f["file"] for f in frames],
        "note": "回放样本与联机响应同源同构；页面渲染只有一条代码路径。",
    }
    (out_dir / "_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  说明 -> {out_dir / '_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
