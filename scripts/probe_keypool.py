#!/usr/bin/env python
"""keypool 端点 / 模型可用性探测（只读探测，不生成素材）。

用途：
1. 确认 keypool 本地代理存活；
2. 逐模型发一条最小请求，确认哪些模型**真实可用**；
3. 确认是否存在**图像生成**能力（本项目需要为评测集/演示准备素材图）。

设计立场：**先验证再使用**。不假设"模型列表里有就等于能用"，
逐个真实调用并记录状态码与错误信息。

用法::

    python scripts/probe_keypool.py
    python scripts/probe_keypool.py --base http://127.0.0.1:8790/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aicg.settings import PROJECT_ROOT  # noqa: E402

DEFAULT_BASE = "http://127.0.0.1:8790/v1"
OUT = PROJECT_ROOT / "outputs" / "reports" / "keypool_probe.json"

# 直调上游的能力探测目标（部分能力不在 /v1/models 中登记，需试）
CHAT_PROBES = [
    "deepseek-v4-flash-0731",
    "glm-5.3",
    "qwen3.8-27b",
    "Atria-Dawn-Preview",
    "glm-5.3-flash",
    "glm-5.3-flashx",
    "qwen3.8-flash",
    "tierflow",
    "tierflow_pro",
    "tiersense",
]

IMAGE_PROBES = [
    "gpt-image-1",
    "dall-e-3",
    "flux-pro",
    "flux-schnell",
    "stable-diffusion-3.5-large",
    "qwen-image",
    "wanx-v1",
    "cogview-4",
    "seedream-3.0",
    "kolors",
    "image-1",
    "imagen-3.0",
]


def _join(base: str, path: str) -> str:
    """拼接 base 与 path，避免 base 已含 /v1 时出现 /v1/v1 重复。"""
    b = base.rstrip("/")
    p = path if path.startswith("/") else "/" + path
    # base 以 /v1 结尾且 path 以 /v1 开头 -> 去掉 path 的前缀
    if b.endswith("/v1") and p.startswith("/v1/"):
        p = p[len("/v1"):]
    return b + p


def post(base: str, path: str, payload: dict, timeout: int = 120) -> tuple[int, str]:
    req = urllib.request.Request(
        _join(base, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def get(base: str, path: str, timeout: int = 30) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(_join(base, path), timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    base = args.base
    report: dict = {"base": base}
    print("=" * 72)
    print("  keypool 端点与模型探测")
    print("=" * 72)

    # 1) 存活
    code, body = get(base.replace("/v1", ""), "/health")
    report["health"] = {"status": code, "body": body[:400]}
    print(f"\n  [1] 存活检查  -> {code}  {body[:120]}")

    # 2) /models
    code, body = get(base, "/v1/models")
    listed: list[str] = []
    if code == 200:
        try:
            listed = [m["id"] for m in json.loads(body).get("data", [])]
        except Exception:  # noqa: BLE001
            pass
    report["models_listed"] = listed
    print(f"  [2] /v1/models -> {code}，登记模型 {len(listed)} 个")

    # 3) 逐模型 chat 探测
    print("\n  [3] 对话能力探测（真实发一条最小请求）")
    chat: dict = {}
    for m in CHAT_PROBES:
        c, b = post(base, "/v1/chat/completions",
                    {"model": m, "messages": [{"role": "user", "content": "hi"}],
                     "max_tokens": 4})
        ok = c == 200
        note = ""
        if not ok:
            try:
                note = json.loads(b).get("error", {}).get("message", b[:120])
            except Exception:  # noqa: BLE001
                note = b[:120]
        chat[m] = {"status": c, "ok": ok, "note": note}
        print(f"      {m:<26} {c}  {'OK' if ok else note[:70]}")
    report["chat_probe"] = chat

    # 4) 图像生成能力探测
    print("\n  [4] 图像生成能力探测（/v1/images/generations）")
    img: dict = {}
    for m in IMAGE_PROBES:
        c, b = post(base, "/v1/images/generations",
                    {"model": m, "prompt": "a red apple on a table", "n": 1, "size": "512x512"},
                    timeout=90)
        ok = c == 200
        note = ""
        if not ok:
            try:
                note = json.loads(b).get("error", {}).get("message", b[:140])
            except Exception:  # noqa: BLE001
                note = b[:140]
        img[m] = {"status": c, "ok": ok, "note": note}
        print(f"      {m:<26} {c}  {'OK' if ok else note[:70]}")
    report["image_probe"] = img

    n_img = sum(1 for v in img.values() if v["ok"])
    report["summary"] = {
        "chat_available": [k for k, v in chat.items() if v["ok"]],
        "image_available": [k for k, v in img.items() if v["ok"]],
        "has_image_generation": n_img > 0,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"  可用对话模型: {report['summary']['chat_available']}")
    print(f"  可用图像模型: {report['summary']['image_available'] or '（无）'}")
    print(f"  是否具备图像生成能力: {report['summary']['has_image_generation']}")
    print(f"  报告 -> {OUT}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
