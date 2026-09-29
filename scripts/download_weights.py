#!/usr/bin/env python
"""下载感知层模型权重。

对应文档：《技术方案.md》§2.2、《开发计划.md》M0 待办项

**为什么需要这个脚本**：权重文件（约 7MB / 更大）不入版本库，
但环境搭建必须是**一条命令可复现**的。本脚本提供一个幂等、可校验、
可离线降级的下载入口。

设计要点：
- **幂等**：已存在且大小合理则跳过，不重复下载；
- **可校验**：下载后校验文件大小与 magic bytes，避免半截文件被误用；
- **可离线**：失败只警告不报错，因为规则后端可以兜底（NFR-R1）。

用法::

    python scripts/download_weights.py              # 下载默认权重
    python scripts/download_weights.py --check      # 只检查不下载
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# (文件名, 下载地址, 最小合理字节数)
# 最小字节数用于识别"下载到一半的截断文件"
WEIGHTS: dict[str, tuple[str, int]] = {
    "yolov8n-seg.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt",
        3_000_000,
    ),
    "yolov8n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        3_000_000,
    ),
}


def _fmt(n: int) -> str:
    return f"{n / 1024 / 1024:.1f}MB"


def check_one(name: str, min_bytes: int) -> tuple[bool, str]:
    """检查权重是否就绪，返回 ``(是否可用, 说明)``。"""
    p = MODELS_DIR / name
    if not p.exists():
        return False, "不存在"
    size = p.stat().st_size
    if size < min_bytes:
        return False, f"文件过小（{_fmt(size)}），疑似截断"
    # .pt 是 torch 的 zip 容器，magic bytes 为 PK\x03\x04
    with p.open("rb") as f:
        magic = f.read(4)
    if not magic.startswith(b"PK"):
        return False, f"magic bytes 异常（{magic!r}），不是有效的 torch 权重"
    return True, f"就绪（{_fmt(size)}）"


def download_one(name: str, url: str, min_bytes: int) -> bool:
    """下载单个权重。返回是否成功。"""
    target = MODELS_DIR / name
    tmp = target.with_suffix(target.suffix + ".part")

    print(f"  下载 {name} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "aicg-weight-fetcher/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with tmp.open("wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        print(f"\r    {pct:5.1f}%  {_fmt(done)} / {_fmt(total)}", end="")
        print()

        if tmp.stat().st_size < min_bytes:
            print(f"  失败：下载不完整（{_fmt(tmp.stat().st_size)}）")
            tmp.unlink(missing_ok=True)
            return False

        tmp.replace(target)
        print(f"  完成：{target}")
        return True

    except Exception as e:  # noqa: BLE001 - 下载失败不应让脚本崩，规则后端可兜底
        print(f"  失败：{type(e).__name__}: {e}")
        tmp.unlink(missing_ok=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="下载感知层模型权重")
    ap.add_argument("--check", action="store_true", help="只检查，不下载")
    ap.add_argument("--only", default=None, help="只处理指定文件")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"权重目录: {MODELS_DIR}\n")

    items = {k: v for k, v in WEIGHTS.items() if args.only is None or k == args.only}
    if not items:
        print(f"未找到匹配的权重名：{args.only}")
        return 1

    missing: list[str] = []
    for name, (url, min_bytes) in items.items():
        ok, note = check_one(name, min_bytes)
        status = "OK  " if ok else "缺失"
        print(f"[{status}] {name}: {note}")
        if not ok:
            missing.append(name)

    if args.check:
        print(f"\n检查完成：{len(items) - len(missing)}/{len(items)} 就绪")
        return 0 if not missing else 2

    if not missing:
        print("\n全部就绪，无需下载。")
        return 0

    print(f"\n开始下载 {len(missing)} 个权重 ...")
    failed: list[str] = []
    for name in missing:
        url, min_bytes = WEIGHTS[name]
        if not download_one(name, url, min_bytes):
            failed.append(name)

    print("\n" + "=" * 56)
    if failed:
        print(f"未成功下载：{', '.join(failed)}")
        print("提示：感知层会自动降级为规则后端，链路仍可运行（NFR-R1）。")
        print("      但构图建议的准确度会下降，建议联网后重试。")
        return 1
    print("全部权重就绪。感知层将使用 YOLO 后端。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
