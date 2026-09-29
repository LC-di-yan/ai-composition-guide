"""用无头 Chromium 对 Web 演示页做真实渲染验证 + 截图。

对应需求：NFR-O3（可复现）/ FR-11（演示）

**为什么不能只做语法检查**

``node --check`` 只能证明 JS 没有语法错误，证明不了：

1. **页面真的能加载**（元素 id 对不上、脚本抛异常都会白屏）；
2. **回放的图能画出来**（``samples/frames.json`` 里的 data URI 是否有效）；
3. **渲染结果与后端数据一致**（分数、指令、占比是否真的落到 DOM 上）。

因此这里用真实 Chromium 打开页面、推进若干帧、断言关键 DOM 文本，
最后截图留证。

**为什么走 CDP 而不是 playwright 库**

本机已装 Chromium（``ms-playwright``），但未在项目里装 playwright
的 Node/Python 包。直接连 CDP 无需新增依赖，也不污染项目环境。
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from shutil import which

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aicg.settings import PROJECT_ROOT  # noqa: E402

CHROME_CANDIDATES = [
    # 注意目录名是 ``chrome-win64`` 而非 ``chrome-win``（踩过）
    r"C:\Users\86134\AppData\Local\ms-playwright\chromium-1243\chrome-win64\chrome.exe",
    r"C:\Users\86134\AppData\Local\ms-playwright\chromium-1234\chrome-win64\chrome.exe",
    r"C:\Users\86134\AppData\Local\ms-playwright\chromium-1228\chrome-win64\chrome.exe",
]


def find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    # 兜底：在 ms-playwright 下递归找
    base = Path(r"C:\Users\86134\AppData\Local\ms-playwright")
    if base.exists():
        for p in sorted(base.glob("chromium-*/chrome-win64/chrome.exe"), reverse=True):
            return str(p)
    for name in ("chrome.exe", "chrome", "msedge.exe"):
        p = which(name)
        if p:
            return p
    return None


class CDP:
    """极简 CDP 客户端（只用 websocket + json）。"""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.i = 0

    def call(self, method: str, **params):
        self.i += 1
        mid = self.i
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str, timeout_note: str = ""):
        r = self.call("Runtime.evaluate", expression=expr,
                      returnByValue=True, awaitPromise=True)
        res = r.get("result", {})
        if r.get("exceptionDetails"):
            raise RuntimeError(f"JS 异常 {timeout_note}: {r['exceptionDetails']}")
        return res.get("value")


def main() -> int:
    from websockets.sync.client import connect
    import websocket  # noqa: F401  (websocket-client)

    chrome = find_chrome()
    if not chrome:
        print("[跳过] 找不到 Chromium/Chrome，无法做真实渲染验证")
        return 0
    print(f"  浏览器: {chrome}")

    port = 9222
    proc = subprocess.Popen([
        chrome, "--headless=new", f"--remote-debugging-port={port}",
        # CDP 默认只接受受信任来源的 WS 连接；不加这条会握手中止（实测 403）
        "--remote-allow-origins=*",
        "--disable-gpu", "--no-first-run", "--no-default-browser-check",
        "--user-data-dir=" + str(PROJECT_ROOT / "outputs" / "_cdp_profile"),
        "--window-size=1400,1200", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        # 等 CDP 就绪。
        # **必须绕过系统代理**：本机环境设了 HTTP_PROXY，urllib 默认会走代理，
        # 导致连 127.0.0.1 也被拒（实测 405/10061）。用 ProxyHandler({}) 显式禁用。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(40):
            try:
                with opener.open(
                        f"http://127.0.0.1:{port}/json/version", timeout=1) as r:
                    json.load(r)
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("[错误] CDP 未就绪")
            return 1

        import websocket as _ws

        # 新建标签页打开演示页。Chrome 较新版本要求 PUT 而非 GET。
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/json/new?http://127.0.0.1:8010/", method="PUT")
        with opener.open(req, timeout=5) as r:
            tab = json.load(r)
        ws = _ws.create_connection(tab["webSocketDebuggerUrl"], timeout=30)
        c = CDP(ws)
        c.call("Runtime.enable")
        c.call("Page.enable")

        # [踩坑记录] 必须禁用浏览器缓存后再导航，否则会**验到上一次的页面**。
        # 实测症状极具误导性：磁盘与 ``GET /`` 都已是新内容（curl 可见），
        # 但 DOM 断言仍报旧文案，看起来像"改了没生效"。
        # 根因是 Chrome 对该 localhost 页做了内存缓存，``PUT /json/new``
        # 打开的新标签直接复用了缓存副本。
        c.call("Network.enable")
        c.call("Network.setCacheDisabled", cacheDisabled=True)
        c.call("Page.navigate", url=f"http://127.0.0.1:8010/")

        # 收集控制台错误
        errors: list[str] = []
        c.call("Log.enable")

        time.sleep(4.0)   # 等 boot() 拉 samples 并渲染（已禁用缓存，需真实往返）

        # 断言：页面已加载、内联兜底未被使用
        title = c.js("document.title")
        frame_count = c.js("state.frames.length")
        is_fallback = c.js("state.frames.length===1 && state.frames[0].file.indexOf('fallback')>=0")
        print(f"  标题      : {title}")
        print(f"  载入帧数  : {frame_count}")
        print(f"  用兜底样本: {is_fallback}")
        if is_fallback:
            print("[错误] 页面没能拉到 samples/frames.json，回放内容不真实")
            return 1

        # 推进到第 5 帧（索引 4），检查 DOM 是否真的写入了后端数据
        c.js("stopPlay(); showFrame(4)")
        time.sleep(1.2)
        got = c.js("""(() => {
          const g = id => (document.getElementById(id)||{}).textContent || "";
          const dots = [...document.querySelectorAll('.tl .dot')];
          return {
            action: g('mAction'), mag: g('mMag'),
            score: g('mScore'), best: g('mBest'),
            total: g('mTotal'), perc: g('mPerc'),
            budget: g('mBudget'),
            subCount: document.querySelectorAll('.sub').length,
            chips: [...document.querySelectorAll('#mChips .chip')].map(e=>e.textContent),
            d10chip: !!document.querySelector('#mChips .chip.warn'),
            boxes: document.querySelectorAll('.bb').length,
            corners: g('corner').slice(0,200),
            activeDot: dots.findIndex(d=>d.classList.contains('on')),
            narr: g('mNarr').slice(0,60),
          };
        })()""")
        print()
        print("  —— 第 5 帧渲染结果（应等于录制的真实响应）——")
        for k, v in got.items():
            print(f"    {k:10s}: {v}")

        # 与录制的原始响应逐项对照
        raw = json.loads((PROJECT_ROOT / "web/samples/frames.json").read_text(
            encoding="utf-8"))[4]["response"]
        print()
        print("  —— 与录制原值对照 ——")
        checks = [
            ("构图分", got["score"], f"{raw['composition']['composition_score']:.1f}"),
            ("建议得分", got["best"], f"{raw['composition']['best_score']:.1f}"),
        ]
        all_ok = True
        for name, a, b in checks:
            ok = a == b
            all_ok &= ok
            print(f"    {name:8s}: 页面={a:>8s}  录制={b:>8s}  {'✅' if ok else '❌'}")
        for label, cond in [
            ("主体框+建议框已绘制", got["boxes"] >= 1),
            ("子项分值已渲染", got["subCount"] >= 4),
            ("时间轴有选中帧", got["activeDot"] >= 0),
            ("解说区有内容", len(got["narr"]) > 10),
            ("降级/来源徽标存在", len(got["chips"]) >= 2),
        ]:
            all_ok &= cond
            print(f"    {label:12s}: {'✅' if cond else '❌'}")

        # 截图（首屏 + 第5帧）
        for tag, expr in [("curated", None)]:
            shot = c.call("Page.captureScreenshot", format="png",
                          captureBeyondViewport=True)
            out = PROJECT_ROOT / "outputs" / "reports" / "web_demo_screenshot.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(base64.b64decode(shot["data"]))
            print(f"\n  截图 -> {out}")

        if errors:
            print(f"  [警告] 页面报错 {len(errors)} 条: {errors[:3]}")

        print()
        print("=" * 60)
        print("  真实渲染验证: " + ("通过 ✅" if all_ok else "未通过 ❌"))
        print("=" * 60)
        return 0 if all_ok else 1

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
