"""Simulation study of cure recovery, survival calibration, and fit failures.

Run ``python benchmarks/simulation.py --repetitions 10 --output results.json``.
For a quick smoke check use ``--quick``. The default study varies sample size,
censoring, follow-up, and susceptible latency (Weibull versus misspecified
lognormal), fitting mixture and promotion-time models to the same cohorts.
Results include all attempts, convergence warnings, failure rates, signed cure
bias, cure RMSE, survival RMSE, and empirical interval-free calibration error.
Use ``--models mixture promotion_time ggkm gradient_boosting`` to include the
kernel and tree estimators; their bounded budgets make nonconvergence visible.
This benchmark is intentionally separate from the short CI test suite.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Any, Sequence
import time
import warnings

import numpy as np

from skcure.datasets import simulate_cure_survival
from skcure.ensemble import GradientBoostingCure
from skcure.ggkm import GGKM
from skcure.linear_model import MixtureCure, PromotionTimeCure


MODEL_NAMES = ("mixture", "promotion_time", "ggkm", "gradient_boosting")


def _make_model(name: str) -> Any:
    if name == "mixture":
        return MixtureCure(max_em_iter=300, tol=1e-6)
    if name == "promotion_time":
        return PromotionTimeCure(max_em_iter=300, tol=1e-6)
    if name == "ggkm":
        return GGKM(gamma=.3, lambda_reg=.1, max_em_iter=100, tol=1e-6)
    if name == "gradient_boosting":
        return GradientBoostingCure(
            n_estimators=5, max_total_estimators=100, max_em_iter=100,
            max_depth=2, tol=1e-6, random_state=73,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}.")


def run_benchmark(
    *, repetitions: int = 10, sample_sizes: Sequence[int] = (200, 1000),
    censoring_rates: Sequence[float] = (.05, .5), followups: Sequence[float] = (1., 5.),
    latencies: Sequence[str] = ("weibull", "lognormal"), random_state: int = 73,
    model_names: Sequence[str] = ("mixture", "promotion_time"),
) -> dict[str, Any]:
    """Return per-attempt records and summaries, keeping every unsuccessful fit."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    if not model_names or any(name not in MODEL_NAMES for name in model_names):
        raise ValueError(f"model_names must select one or more of {MODEL_NAMES}.")
    rng = np.random.default_rng(random_state)
    records: list[dict[str, Any]] = []
    for n, rate, followup, latency, repetition in product(sample_sizes, censoring_rates, followups, latencies, range(repetitions)):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        train = simulate_cure_survival(n, censoring_rate=rate, followup=followup, latency=latency, random_state=seed)
        test = simulate_cure_survival(1000, censoring_rate=rate, followup=followup, latency=latency, random_state=seed + 1)
        times = np.linspace(.05, followup, 20)
        truth = test.survival_probability(times)
        for name in model_names:
            estimator = _make_model(name)
            record: dict[str, Any] = dict(n_samples=n, censoring_rate=rate, followup=followup, latency=latency,
                          repetition=repetition, seed=seed, model=name,
                          censoring_proportion=float(1 - train.y["event"].mean()),
                          converged=False, failed=False, error=None, warnings=[],
                          estimator_parameters=estimator.get_params(deep=False))
            start = time.perf_counter()
            captured = []
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    model = estimator.fit(train.X, train.y)
                    cure = model.predict_cure_probability(test.X)
                    survival = model.predict_survival(test.X, times)
                record.update(
                    converged=bool(model.converged_), n_iter=int(model.n_iter_),
                    cure_bias=float(np.mean(cure - test.cure_probability)),
                    cure_rmse=float(np.sqrt(np.mean((cure - test.cure_probability) ** 2))),
                    survival_rmse=float(np.sqrt(np.mean((survival - truth) ** 2))),
                    survival_calibration_error=float(np.mean(np.abs(survival.mean(axis=0) - truth.mean(axis=0)))),
                )
            except Exception as exc:
                record.update(failed=True, error=f"{type(exc).__name__}: {exc}")
            record["warnings"] = [f"{item.category.__name__}: {item.message}" for item in captured]
            record["seconds"] = time.perf_counter() - start
            records.append(record)
    summaries = []
    grouping = ("n_samples", "censoring_rate", "followup", "latency", "model")
    keys = {tuple(record[key] for key in grouping) for record in records}
    for key in sorted(keys):
        group = [record for record in records if tuple(record[name] for name in grouping) == key]
        usable = [record for record in group if not record["failed"] and record["converged"]]
        summary: dict[str, Any] = dict(zip(grouping, key))
        summary.update(attempts=len(group), failure_rate=float(np.mean([record["failed"] for record in group])),
                       nonconvergence_rate=float(np.mean([not record["failed"] and not record["converged"] for record in group])),
                       usable=len(usable))
        for name in ("cure_bias", "cure_rmse", "survival_rmse", "survival_calibration_error"):
            summary[name] = float(np.mean([record[name] for record in usable])) if usable else None
        summaries.append(summary)
    return {"configuration": {"random_state": random_state, "repetitions": repetitions,
                               "models": list(model_names),
                               "truth": "logistic mixture cure with covariate-independent latency"},
            "summaries": summaries, "attempts": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("simulation-results.json"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["mixture", "promotion_time"])
    options = parser.parse_args()
    configuration: dict[str, Any] = dict(repetitions=options.repetitions, model_names=options.models)
    if options.quick:
        configuration.update(repetitions=1, sample_sizes=(100,), censoring_rates=(.1,), followups=(4.,), latencies=("weibull",))
    result = run_benchmark(**configuration)
    options.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summaries"], indent=2))
    print(f"Saved every attempted fit to {options.output}.")


if __name__ == "__main__":
    main()
