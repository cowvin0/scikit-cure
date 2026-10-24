"""Process isolation, native memory reporting, and retained benchmark failures."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "performance.py"
spec = importlib.util.spec_from_file_location("skcure_performance_benchmark", PATH)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_fresh_processes_report_native_memory_timings_and_bounded_fit_diagnostics():
    result = benchmark.run_benchmark(
        sample_sizes=(24, 32),
        repetitions=1,
        model_names=("mixture",),
        max_em_iter=2,
        prediction_samples=5,
        timeout=60,
    )
    assert len(result["attempts"]) == 2
    pids = []
    for record in result["attempts"]:
        assert not record["failed"], record["error"]
        assert record["phase"] == "complete"
        assert record["worker_pid"] != os.getpid()
        pids.append(record["worker_pid"])
        assert record["fit_seconds"] >= 0
        assert record["prediction_seconds"] >= 0
        assert record["peak_rss_bytes"] >= record["baseline_peak_rss_bytes"] > 0
        assert record["peak_rss_bytes"] >= record["fit_peak_rss_bytes"]
        assert record["memory_method"]
        assert record["memory_error"] is None
        assert record["retained_kernel_bytes"] == 0
        assert record["estimator_parameters"]["max_em_iter"] == 2
        assert record["termination_reason"] in ("converged", "max_em_iter")
        if not record["converged"]:
            assert any("ConvergenceWarning" in value for value in record["warnings"])
    assert len(set(pids)) == len(pids)
    assert result["environment"]["packages"]["numpy"]
    assert result["configuration"]["thread_limit"] == 1
    assert all(summary["fit_seconds"]["count"] == 1 for summary in result["summaries"])
    assert all(
        summary["fit_seconds"]["standard_deviation"] is None
        for summary in result["summaries"]
    )
    json.dumps(result, allow_nan=False)


def test_ggkm_reports_actual_dense_kernel_storage():
    result = benchmark.run_benchmark(
        sample_sizes=(24,),
        repetitions=1,
        model_names=("ggkm",),
        max_em_iter=1,
        prediction_samples=5,
        timeout=60,
    )
    record = result["attempts"][0]
    assert not record["failed"], record["error"]
    assert record["dense_gram_estimate_bytes"] == 24 * 24 * 8
    assert record["retained_kernel_bytes"] == record["dense_gram_estimate_bytes"]


@pytest.mark.parametrize("failure", ["timeout", "exit"])
def test_worker_failure_is_retained_in_attempts_and_summaries(failure, monkeypatch):
    def failed(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 0.01)
        return subprocess.CompletedProcess(
            args[0], 42, stdout="", stderr="controlled failure"
        )

    monkeypatch.setattr(benchmark.subprocess, "run", failed)
    result = benchmark.run_benchmark(
        sample_sizes=(24,),
        repetitions=2,
        model_names=("mixture",),
        timeout=0.01,
    )
    assert len(result["attempts"]) == 2
    assert all(
        record["failed"] and record["phase"] == "worker"
        for record in result["attempts"]
    )
    assert all(record["peak_rss_bytes"] is None for record in result["attempts"])
    if failure == "timeout":
        assert all("TimeoutExpired" in record["error"] for record in result["attempts"])
    else:
        assert all(
            "controlled failure" in record["error"] for record in result["attempts"]
        )
    summary = result["summaries"][0]
    assert summary["failure_rate"] == 1
    assert summary["completed"] == 0
    assert summary["peak_rss_bytes"]["mean"] is None
    assert summary["fit_seconds"]["count"] == 0
    json.dumps(result, allow_nan=False)


def test_quick_cli_writes_a_reproducible_configuration(tmp_path):
    output = tmp_path / "reports" / "performance.json"
    subprocess.run(
        [
            sys.executable,
            str(PATH),
            "--quick",
            "--models",
            "mixture",
            "--sample-sizes",
            "24",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["configuration"]["sample_sizes"] == [24]
    assert result["configuration"]["repetitions"] == 1
    assert result["configuration"]["max_em_iter"] == 2
    assert result["configuration"]["max_total_estimators"] == 10
    assert len(result["attempts"]) == 1
    assert not result["attempts"][0]["failed"]


@pytest.mark.parametrize(
    "settings",
    [
        {"sample_sizes": ()},
        {"sample_sizes": (1,)},
        {"sample_sizes": (24, 24)},
        {"model_names": ()},
        {"model_names": ("unknown",)},
        {"repetitions": 0},
        {"max_em_iter": False},
        {"thread_limit": 0},
        {"timeout": float("nan")},
        {"random_state": -1},
    ],
)
def test_invalid_configuration_fails_before_starting_workers(settings):
    with pytest.raises(ValueError):
        benchmark.run_benchmark(**settings)
