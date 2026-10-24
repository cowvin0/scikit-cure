"""Measure fitting and prediction cost in fresh processes.

Run ``python benchmarks/performance.py --quick --output performance.json``
for a small smoke run. A scaling study can select ``--sample-sizes 100 250
500 1000 --repetitions 3 --models mixture promotion_time ggkm gradient_boosting``.

Peak RSS is the operating system's process high-water mark, including Python,
imported libraries, training/evaluation data, and native NumPy allocations.
It is not memory allocated exclusively by the estimator. Baseline and post-fit
high-water marks provide context; their difference is not an allocation total.
This measures a bounded training configuration, not comparative model accuracy.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from itertools import product
import json
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence
import warnings


MODEL_NAMES = ("mixture", "promotion_time", "ggkm", "gradient_boosting")
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("skcure", "numpy", "scipy", "scikit-learn", "scikit-survival"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "packages": packages,
        "package_versions_scope": "installed distribution metadata; loaded scikit-cure source versions are recorded separately",
    }


def _peak_process_rss() -> tuple[int, str]:
    """Read peak resident memory including native numerical allocations."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_memory = psapi.GetProcessMemoryInfo
        get_memory.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
        ]
        get_memory.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not get_memory(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize), "GetProcessMemoryInfo.PeakWorkingSetSize"
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        multiplier = 1 if sys.platform == "darwin" else 1024
        return int(value * multiplier), "getrusage(RUSAGE_SELF).ru_maxrss"
    raise OSError(f"Peak RSS measurement is unsupported on {sys.platform!r}.")


def _make_model(payload: dict[str, Any]) -> Any:
    from skcure.ensemble import GradientBoostingCure
    from skcure.ggkm import GGKM
    from skcure.linear_model import MixtureCure, PromotionTimeCure

    common = {"max_em_iter": payload["max_em_iter"], "tol": 1e-6}
    name = payload["model"]
    if name == "mixture":
        return MixtureCure(**common)
    if name == "promotion_time":
        return PromotionTimeCure(**common)
    if name == "ggkm":
        return GGKM(gamma=1.0 / payload["n_features"], lambda_reg=.1, **common)
    if name == "gradient_boosting":
        return GradientBoostingCure(
            n_estimators=5, max_total_estimators=payload["max_total_estimators"],
            max_depth=2, random_state=payload["seed"], **common,
        )
    raise ValueError(f"Unknown model {name!r}.")


def _empty_record(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload, "failed": False, "error": None, "phase": "initializing",
        "converged": False, "termination_reason": None, "n_iter": None,
        "n_trees": None, "fit_seconds": None, "prediction_seconds": None,
        "baseline_peak_rss_bytes": None, "fit_peak_rss_bytes": None,
        "peak_rss_bytes": None, "memory_method": None, "memory_error": None,
        "retained_kernel_bytes": 0, "warnings": [], "estimator_parameters": None,
        "dense_gram_estimate_bytes": (
            8 * payload["n_samples"] ** 2 if payload["model"] == "ggkm" else 0
        ),
    }


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Fit one scenario; only this fresh process imports numerical libraries."""
    import numpy as np
    import skcure

    from skcure.datasets import simulate_cure_survival

    record = _empty_record(payload)
    record["worker_pid"] = os.getpid()
    record["skcure_source_version"] = skcure.__version__
    captured: list[Any] = []

    def measure_memory(key: str) -> None:
        try:
            record[key], record["memory_method"] = _peak_process_rss()
        except (OSError, AttributeError) as exc:
            record["memory_error"] = f"{type(exc).__name__}: {exc}"

    model = None
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            record["phase"] = "simulation"
            cohort = simulate_cure_survival(
                payload["n_samples"], n_features=payload["n_features"],
                random_state=payload["seed"], followup=5., censoring_rate=.1,
            )
            evaluation_X = np.random.default_rng(payload["seed"] + 1).normal(
                size=(payload["prediction_samples"], payload["n_features"]),
            )
            times = np.linspace(0., 5., 21)
            model = _make_model(payload)
            record["estimator_parameters"] = model.get_params(deep=False)
            record["event_proportion"] = float(cohort.y["event"].mean())
            measure_memory("baseline_peak_rss_bytes")
            record["phase"] = "fit"
            start = time.perf_counter()
            try:
                model.fit(cohort.X, cohort.y)
            finally:
                record["fit_seconds"] = time.perf_counter() - start
                measure_memory("fit_peak_rss_bytes")
            record["phase"] = "prediction"
            start = time.perf_counter()
            try:
                risk = np.asarray(model.predict(evaluation_X))
                cure = np.asarray(model.predict_cure_probability(evaluation_X))
                survival = np.asarray(model.predict_survival(evaluation_X, times))
                if (
                    risk.shape != (payload["prediction_samples"],)
                    or cure.shape != risk.shape
                    or survival.shape != (payload["prediction_samples"], len(times))
                    or not all(np.isfinite(values).all() for values in (risk, cure, survival))
                    or np.any((cure < 0) | (cure > 1))
                    or np.any((survival < 0) | (survival > 1))
                    or np.any(np.diff(survival, axis=1) > 1e-12)
                ):
                    raise ValueError("The fitted model returned invalid predictions.")
            finally:
                record["prediction_seconds"] = time.perf_counter() - start
            record["phase"] = "complete"
    except Exception as exc:
        record.update(failed=True, error=f"{type(exc).__name__}: {exc}")
    finally:
        measure_memory("peak_rss_bytes")
        if model is not None:
            record["converged"] = bool(getattr(model, "converged_", False))
            record["termination_reason"] = getattr(model, "termination_reason_", None)
            record["n_iter"] = getattr(model, "n_iter_", None)
            trees = getattr(model, "models_", None)
            record["n_trees"] = len(trees) if trees is not None else None
            kernel = getattr(model, "K_", None)
            record["retained_kernel_bytes"] = int(kernel.nbytes) if kernel is not None else 0
        record["warnings"] = [f"{item.category.__name__}: {item.message}" for item in captured]
    return record


def _run_worker(payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    record = _empty_record(payload)
    record["phase"] = "worker"
    environment = dict(os.environ)
    environment.update({name: str(payload["thread_limit"]) for name in _THREAD_VARIABLES})
    with tempfile.TemporaryDirectory(prefix="skcure-performance-") as directory:
        output = Path(directory) / "result.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(output)]
        try:
            result = subprocess.run(
                command, input=json.dumps(payload), capture_output=True, text=True,
                env=environment, timeout=timeout, check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Worker exited with status {result.returncode}: {result.stderr[-4000:]}")
            if not output.is_file():
                raise RuntimeError("Worker produced no result file.")
            record = json.loads(output.read_text(encoding="utf-8"))
            record["worker_stderr"] = result.stderr[-4000:]
        except (subprocess.TimeoutExpired, OSError, RuntimeError, ValueError) as exc:
            record.update(failed=True, error=f"{type(exc).__name__}: {exc}")
    return record


def _statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for name, n_samples in sorted({(item["model"], item["n_samples"]) for item in records}):
        group = [item for item in records if (item["model"], item["n_samples"]) == (name, n_samples)]
        completed = [item for item in group if not item["failed"]]
        summary: dict[str, Any] = {
            "model": name, "n_samples": n_samples, "attempts": len(group),
            "completed": len(completed), "failed": len(group) - len(completed),
            "failure_rate": (len(group) - len(completed)) / len(group),
            "converged": sum(item["converged"] for item in completed),
            "nonconverged": sum(not item["converged"] for item in completed),
            "performance_population": "all completed fits, including nonconverged fits",
        }
        for metric in ("fit_seconds", "prediction_seconds", "peak_rss_bytes", "n_iter", "n_trees"):
            summary[metric] = _statistics([item[metric] for item in completed if item[metric] is not None])
        summaries.append(summary)
    return summaries


def run_benchmark(
    *, sample_sizes: Sequence[int] = (100, 250, 500, 1000), repetitions: int = 3,
    model_names: Sequence[str] = MODEL_NAMES, n_features: int = 5,
    prediction_samples: int = 200, max_em_iter: int = 20,
    max_total_estimators: int = 100, random_state: int = 73,
    thread_limit: int = 1, timeout: float = 300.,
) -> dict[str, Any]:
    """Return observed performance, settings, failures, and process RSS peaks."""
    import random

    for name, value in (
        ("repetitions", repetitions), ("n_features", n_features),
        ("prediction_samples", prediction_samples), ("max_em_iter", max_em_iter),
        ("max_total_estimators", max_total_estimators), ("thread_limit", thread_limit),
    ):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not sample_sizes or any(not isinstance(n, Integral) or isinstance(n, bool) or n < 2 for n in sample_sizes):
        raise ValueError("sample_sizes must contain integers of at least two.")
    if not model_names or any(name not in MODEL_NAMES for name in model_names):
        raise ValueError(f"model_names must select one or more of {MODEL_NAMES}.")
    if len(set(model_names)) != len(model_names) or len(set(sample_sizes)) != len(sample_sizes):
        raise ValueError("model_names and sample_sizes must not contain duplicates.")
    if not isinstance(random_state, Integral) or isinstance(random_state, bool) or random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")
    if not isinstance(timeout, Real) or isinstance(timeout, bool) or not 0 < timeout < float("inf"):
        raise ValueError("timeout must be positive and finite.")
    generator = random.Random(int(random_state))
    records = []
    for n_samples, repetition in product(sample_sizes, range(repetitions)):
        seed = generator.randrange(2**31 - 1)
        for name in model_names:
            payload = {
                "model": name, "n_samples": int(n_samples), "repetition": repetition,
                "seed": seed, "n_features": int(n_features),
                "prediction_samples": int(prediction_samples), "max_em_iter": int(max_em_iter),
                "max_total_estimators": int(max_total_estimators), "thread_limit": int(thread_limit),
            }
            records.append(_run_worker(payload, float(timeout)))
    environment = _environment()
    environment["skcure_source_versions"] = sorted({
        item["skcure_source_version"] for item in records if "skcure_source_version" in item
    })
    return {
        "schema_version": 1, "benchmark": "fit_prediction_scaling",
        "environment": environment,
        "configuration": {
            "sample_sizes": list(sample_sizes), "repetitions": int(repetitions),
            "models": list(model_names), "n_features": int(n_features),
            "prediction_samples": int(prediction_samples), "max_em_iter": int(max_em_iter),
            "max_total_estimators": int(max_total_estimators), "random_state": int(random_state),
            "thread_limit": int(thread_limit), "worker_timeout_seconds": float(timeout),
            "prediction_task": "risk, cure probability, and population survival on 21 times from 0 to 5",
            "data": "independently censored logistic mixture cure; unit-scale Weibull latency",
        },
        "memory_measurement": {
            "metric": "operating-system peak process resident set / working set in bytes",
            "includes": "interpreter, imports, data, estimator, prediction, and native numerical allocations",
            "baseline": "process high-water mark after imports and data preparation, before fitting",
            "limitation": "peak minus baseline is a high-water increment, not estimator-exclusive allocated memory",
            "dense_gram_estimate": "one n_samples by n_samples float64 matrix for GGKM; actual peak includes other arrays",
        },
        "summaries": _summaries(records), "attempts": records,
    }


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        record = _worker(json.load(sys.stdin))
        Path(sys.argv[2]).write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--sample-sizes", nargs="+", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--prediction-samples", type=int)
    parser.add_argument("--max-em-iter", type=int)
    parser.add_argument("--max-total-estimators", type=int)
    parser.add_argument("--random-state", type=int, default=73)
    parser.add_argument("--thread-limit", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.)
    parser.add_argument("--output", type=Path, default=Path("performance-results.json"))
    options = parser.parse_args()
    result = run_benchmark(
        sample_sizes=options.sample_sizes if options.sample_sizes is not None else ((48,) if options.quick else (100, 250, 500, 1000)),
        repetitions=options.repetitions if options.repetitions is not None else (1 if options.quick else 3),
        model_names=options.models, n_features=options.n_features,
        prediction_samples=options.prediction_samples if options.prediction_samples is not None else (20 if options.quick else 200),
        max_em_iter=options.max_em_iter if options.max_em_iter is not None else (2 if options.quick else 20),
        max_total_estimators=options.max_total_estimators if options.max_total_estimators is not None else (10 if options.quick else 100),
        random_state=options.random_state, thread_limit=options.thread_limit, timeout=options.timeout,
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summaries"], indent=2))
    print(f"Saved every attempted fit to {options.output}.")


if __name__ == "__main__":
    main()
