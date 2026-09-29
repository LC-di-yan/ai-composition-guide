"""CLI 测试。

对应文档：《目录结构.md》§CLI
对应需求：全流程可运行（NFR-M4）

**测试范围**：CLI 是胶水层，重点验证"参数解析正确、错误有友好提示、
子命令能真正跑通"，而不重复验证业务逻辑（那是各层的职责）。

为保持 CI 快速且无副作用，这里：
- 只调用**不加载模型**的子命令（``doctor`` / ``make-fixtures --list``）；
- 用 ``capsys`` 检查输出，用返回值检查退出码；
- 不实际启动 uvicorn（只用 ``--help`` 验证参数）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aicg.cli.__main__ import build_parser, main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestParser:
    """参数解析契约。"""

    def test_all_subcommands_registered(self):
        """每个子命令都能被解析（带必需参数）。"""
        parser = build_parser()
        cases = {
            "demo": ["demo"],
            "score": ["score", "x.jpg"],
            "serve": ["serve"],
            "make-fixtures": ["make-fixtures"],
            "bench": ["bench"],
            "eval": ["eval"],
            "accept": ["accept"],
            "doctor": ["doctor"],
        }
        for cmd, argv in cases.items():
            ns = parser.parse_args(argv)
            assert ns.command == cmd, f"{cmd} 解析结果不符"

    def test_demo_defaults(self):
        ns = build_parser().parse_args(["demo"])
        assert ns.source.endswith(".mp4")
        assert ns.backend is None
        assert ns.compare is False

    def test_demo_backend_choices_validated(self):
        parser = build_parser()
        ns = parser.parse_args(["demo", "--backend", "rule"])
        assert ns.backend == "rule"
        with pytest.raises(SystemExit):
            parser.parse_args(["demo", "--backend", "bogus"])

    def test_bench_defaults_match_script(self):
        """bench 的参数名必须与 scripts/benchmark_latency.py 一致。

        这里曾有过一次真实不一致：CLI 用 ``--runs``，脚本用 ``--repeat``，
        导致 ``aicg bench`` 直接崩。用参数名断言把它锁住。
        """
        ns = build_parser().parse_args(["bench"])
        assert hasattr(ns, "repeat")
        assert hasattr(ns, "frames")
        assert not hasattr(ns, "runs")

    def test_score_requires_image(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["score"])

    def test_accept_defaults_match_script(self):
        """``accept`` 的参数名必须与 scripts/acceptance_report.py 一致。

        同 ``bench`` 的历史教训：CLI 与脚本参数名不一致会让子命令直接崩。
        """
        ns = build_parser().parse_args(["accept"])
        assert ns.backend == "yolo"
        assert hasattr(ns, "debounce_source")
        assert hasattr(ns, "quick")
        assert hasattr(ns, "json")
        assert ns.debounce_backend is None
        # 脚本确实支持这些参数
        script = (PROJECT_ROOT / "scripts" / "acceptance_report.py").read_text(encoding="utf-8")
        for flag in ("--debounce-source", "--debounce-backend", "--quick", "--backend"):
            assert flag in script, f"验收脚本缺少参数 {flag}"

    def test_missing_subcommand_exits(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestDoctor:
    """``doctor`` 环境自检。"""

    def test_runs_and_reports(self, capsys):
        rc = main(["doctor"])
        out = capsys.readouterr().out
        assert "环境自检" in out
        assert "Python" in out
        assert "感知后端" in out
        # 本机环境健全（依赖齐全），应返回 0
        assert rc == 0

    def test_reports_optional_missing_gracefully(self, capsys):
        """可选依赖缺失应给出提示而不是崩溃。"""
        rc = main(["doctor"])
        out = capsys.readouterr().out
        assert rc == 0
        # YOLO 权重与夹具要么 OK 要么有明确提示
        assert ("权重" in out) and ("夹具" in out)


class TestMakeFixturesList:
    def test_list_mode(self, capsys):
        rc = main(["make-fixtures", "--list"])
        out = capsys.readouterr().out
        assert "夹具目录" in out
        assert rc == 0


class TestScoreErrors:
    """``score`` 的错误处理必须是友好提示而非 traceback。"""

    def test_missing_file(self, capsys):
        rc = main(["score", "no/such/image.jpg"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "不存在" in err

    def test_video_input_rejected_with_hint(self, capsys):
        """给视频文件应提示改用 demo，而不是尝试解码后报错。"""
        rc = main(["score", "tests/fixtures/handheld_jitter.mp4"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "视频" in err
        assert "demo" in err


class TestEval:
    def test_eval_missing_script_gives_clear_message(self, capsys):
        """eval 依赖尚未产出的标注数据集，应明确说明而非静默失败。"""
        script = PROJECT_ROOT / "scripts" / "evaluate_composition.py"
        if script.exists():
            pytest.skip("评估脚本已存在，跳过该检查")
        rc = main(["eval"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "evaluate_composition" in err


class TestServeArgs:
    """serve 参数（不真正启动服务）。"""

    def test_serve_defaults(self):
        ns = build_parser().parse_args(["serve"])
        assert ns.host == "127.0.0.1"
        assert ns.port == 8000
        assert ns.reload is False

    def test_serve_custom_port(self):
        ns = build_parser().parse_args(["serve", "--port", "9001"])
        assert ns.port == 9001


class TestModuleEntry:
    """``python -m aicg.cli`` 必须可用（作为包的入口）。"""

    def test_dunder_main_importable(self):
        import aicg.cli.__main__ as m

        assert callable(m.main)
        assert callable(m.build_parser)

    def test_package_reexports(self):
        import aicg.cli as c

        assert callable(c.main)
        assert callable(c.build_parser)
