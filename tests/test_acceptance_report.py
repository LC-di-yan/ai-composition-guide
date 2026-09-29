"""端到端验收报告的纯函数测试（NFR-O3）。

对应文档：《测试与验收.md》§3 验收标准、§4 效果评测方案

**测试意图**：验收报告本身是"结论的载体"，如果它的汇总/判定逻辑有
偏差，会直接把错误结论写进作品集。因此这里只测**不依赖推理**的纯逻辑
（解析、判定、折行），不跑真实测量（那由 ``scripts/acceptance_report.py``
的集成运行覆盖）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import acceptance_report as ar  # noqa: E402


class TestSummaryJsonExtraction:
    def test_extracts_last_summary(self):
        text = "\n".join([
            "some log",
            'SUMMARY_JSON {"a": 1, "b": 2}',
            "trailing noise",
        ])
        got = ar._extract_summary(text)
        assert got == {"a": 1, "b": 2}

    def test_takes_the_last_when_multiple(self):
        """中间过程可能打印多个 SUMMARY，必须取最后一次（最终结论）。"""
        text = '\n'.join([
            'SUMMARY_JSON {"n": 1}',
            'SUMMARY_JSON {"n": 2}',
        ])
        assert ar._extract_summary(text) == {"n": 2}

    def test_returns_none_when_absent(self):
        assert ar._extract_summary("no summary here") is None

    def test_malformed_json_returns_none(self):
        assert ar._extract_summary('SUMMARY_JSON {not json}') is None


class TestWrap:
    def test_cjk_counts_as_double_width(self):
        """中文字符按 2 列宽计算，避免报告在终端里错行。"""
        # 宽度 6 → 每行最多 3 个汉字；6 字恰好折成两行
        assert ar._wrap("中文中文中文", width=6) == ["中文中", "文中文"]
        # 若按 1 列宽算，6 个字会挤成一行 — 断言确实折了行
        assert len(ar._wrap("中文中文中文", width=6)) == 2

    def test_ascii_wrap(self):
        assert ar._wrap("abcdefgh", width=4) == ["abcd", "efgh"]

    def test_never_returns_empty(self):
        assert ar._wrap("", width=10) == [""]

    def test_mixed_text_no_loss(self):
        text = "P95=28.2ms 预算 333ms 占 8.4%"
        joined = "".join(ar._wrap(text, width=20))
        assert joined == text


class TestRenderStatusSummary:
    def _results(self, *statuses):
        return [
            ar.CheckResult(f"AC-{i}", f"项{i}", s, "细节", {})
            for i, s in enumerate(statuses)
        ]

    def test_counts_each_status(self):
        out = ar.render(self._results("PASS", "PASS", "WARN", "FAIL"),
                        generated_at="t", env={})
        assert "PASS 2" in out
        assert "WARN 1" in out
        assert "FAIL 1" in out

    def test_simplifications_always_printed(self):
        """简化处清单必须始终出现——这是项目的诚实底线。"""
        out = ar.render(self._results("PASS"), generated_at="t", env={})
        assert "简化处" in out
        for s in ar.SIMPLIFICATIONS:
            assert s.split("：")[0][:6] in out

    def test_source_files_listed(self):
        results = [ar.CheckResult("AC-N1", "延迟", "PASS", "x", {"source": "bench_x.json"})]
        out = ar.render(results, generated_at="t", env={})
        assert "bench_x.json" in out


class TestFrameBudget:
    def test_budget_is_333ms_for_3fps(self):
        assert abs(ar.FRAME_BUDGET_MS - 333.333) < 0.01


class TestLatestAndLoad:
    def test_latest_picks_newest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ar, "REPORTS_DIR", tmp_path)
        (tmp_path / "a_1.json").write_text("{}", encoding="utf-8")
        (tmp_path / "a_2.json").write_text("{}", encoding="utf-8")  # 后写 → 更新
        assert ar._latest("a_*.json").name == "a_2.json"

    def test_latest_none_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ar, "REPORTS_DIR", tmp_path)
        assert ar._latest("nope_*.json") is None

    def test_load_handles_bad_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{oops", encoding="utf-8")
        assert ar._load(bad) is None

    def test_load_handles_missing(self, tmp_path):
        assert ar._load(tmp_path / "gone.json") is None

    def test_load_reads_valid(self, tmp_path):
        p = tmp_path / "ok.json"
        p.write_text(json.dumps({"x": 1}), encoding="utf-8")
        assert ar._load(p) == {"x": 1}


class TestSrccGate:
    """SRCC 判定逻辑：样本不足必须 WARN，不得冒充 PASS。"""

    def test_small_n_is_warn_not_pass(self, monkeypatch):
        monkeypatch.setattr(
            ar, "_run",
            lambda *a, **k: (0, "ok"),
        )
        monkeypatch.setattr(
            ar, "_load",
            lambda p: {"srcc": 0.8, "n_samples": 5, "statistically_meaningful": False,
                       "min_meaningful_n": 30},
        )
        monkeypatch.setattr(ar, "REPORTS_DIR", Path("."))
        result, _ = ar.measure_srcc("python", "yolo")
        assert result.status == "WARN", "样本量不足时不得判 PASS"
        assert "统计效力" in result.detail

    def test_large_n_high_srcc_passes(self, monkeypatch):
        monkeypatch.setattr(ar, "_run", lambda *a, **k: (0, "ok"))
        monkeypatch.setattr(
            ar, "_load",
            lambda p: {"srcc": 0.72, "n_samples": 60, "statistically_meaningful": True},
        )
        result, _ = ar.measure_srcc("python", "yolo")
        assert result.status == "PASS"

    def test_large_n_low_srcc_fails(self, monkeypatch):
        monkeypatch.setattr(ar, "_run", lambda *a, **k: (0, "ok"))
        monkeypatch.setattr(
            ar, "_load",
            lambda p: {"srcc": 0.12, "n_samples": 60, "statistically_meaningful": True},
        )
        result, _ = ar.measure_srcc("python", "yolo")
        assert result.status == "FAIL"


class TestDebounceGate:
    def test_strong_reduction_passes(self, monkeypatch):
        monkeypatch.setattr(ar, "_run", lambda *a, **k: (0, "ok"))
        monkeypatch.setattr(
            ar, "_load",
            lambda p: {
                "frames": 240, "switch_reduction_pct": 77.38,
                "mean_run_frames_before": 2.82, "mean_run_frames_after": 12.0,
                "off": {"switch_stats": {"switch_count": 84, "switches_per_minute": 254.07}},
                "on": {"switch_stats": {"switch_count": 19, "switches_per_minute": 57.47}},
            },
        )
        monkeypatch.setattr(ar, "FIXTURES", PROJECT_ROOT / "tests" / "fixtures")
        result, _ = ar.measure_debounce("python", "rule", "handheld_jitter.mp4")
        assert result.status == "PASS"
        assert result.value["reduction_pct"] == 77.38
        assert abs(result.value["mean_run_gain_x"] - 4.26) < 0.1

    def test_weak_reduction_warns(self, monkeypatch):
        monkeypatch.setattr(ar, "_run", lambda *a, **k: (0, "ok"))
        monkeypatch.setattr(
            ar, "_load",
            lambda p: {
                "frames": 240, "switch_reduction_pct": 10.0,
                "mean_run_frames_before": 3.0, "mean_run_frames_after": 3.3,
                "off": {"switch_stats": {"switch_count": 30, "switches_per_minute": 90}},
                "on": {"switch_stats": {"switch_count": 27, "switches_per_minute": 81}},
            },
        )
        monkeypatch.setattr(ar, "FIXTURES", PROJECT_ROOT / "tests" / "fixtures")
        result, _ = ar.measure_debounce("python", "rule", "handheld_jitter.mp4")
        assert result.status == "WARN"

    def test_missing_fixture_fails_fast(self, monkeypatch):
        monkeypatch.setattr(ar, "FIXTURES", Path("/nonexistent/dir"))
        result, _ = ar.measure_debounce("python", "rule", "nope.mp4")
        assert result.status == "FAIL"
        assert "夹具缺失" in result.detail


class TestLatencyGate:
    def _patch(self, monkeypatch, p95):
        """模拟：脚本执行成功，且产生了一份新的 benchmark 报告。"""
        calls = {"n": 0}

        def fake_run(*a, **k):
            # 每次调用"产生"一份新报告：让 _latest 前后返回不同值
            calls["n"] += 1
            return 0, "ok"

        monkeypatch.setattr(ar, "_run", fake_run)
        monkeypatch.setattr(ar, "_load", lambda p: {
            "stages": {
                "perception": {"mean": p95 * 0.99},
                "total": {"mean": p95 * 0.9, "p50": p95 * 0.9, "p95": p95,
                          "p99": p95 * 1.05},
            },
            "first_frame_ms": 30.0,
        })
        # 关键：before 与 after 必须不同，否则被判定为"脚本失败"
        seq = iter([Path("bench_old.json"), Path("bench_new.json")])
        monkeypatch.setattr(ar, "_latest", lambda pat: next(seq, Path("bench_new.json")))

    def test_within_budget_passes(self, monkeypatch):
        self._patch(monkeypatch, 100.0)
        result, _ = ar.measure_latency("python", "yolo")
        assert result.status == "PASS"
        assert result.value["bottleneck"] == "perception"

    def test_over_budget_fails(self, monkeypatch):
        self._patch(monkeypatch, 500.0)
        result, _ = ar.measure_latency("python", "yolo")
        assert result.status == "FAIL"
