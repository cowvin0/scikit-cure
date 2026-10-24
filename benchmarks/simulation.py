"""Repeated scientific validation of latent-count cure estimators.

Run python benchmarks/simulation.py --quick for a bounded smoke study.
The full default varies count family, latency specification, sample size,
censoring and follow-up. Every attempted fit and metric failure is retained.
Reported Monte Carlo standard errors quantify variation across independent
simulated cohorts; they are not bootstrap confidence intervals for predictions.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from itertools import product
import json
from numbers import Integral
from pathlib import Path
import platform
from typing import Any, Sequence
import time
import warnings

import numpy as np

from skcure.datasets import LATENT_DISTRIBUTIONS, simulate_cure_survival
from skcure.ensemble import GradientBoostingCure
from skcure.ggkm import GGKM
from skcure.linear_model import MixtureCure, PromotionTimeCure
from skcure.metrics import evaluate_survival


MODEL_NAMES = ("mixture", "promotion_time", "ggkm", "gradient_boosting")
TRUTH_METRICS = (
    "cure_bias", "cure_rmse", "survival_rmse", "survival_calibration_error",
)
OBSERVED_METRICS = ("concordance_ipcw", "mean_auc", "integrated_brier_score")


def _make_model(
    name: str, *, latent_distribution: str = "poisson", K_bin: int = 1,
    phi: float = .5, random_state: int = 73,
) -> Any:
    if name == "mixture":
        return MixtureCure(max_em_iter=300, tol=1e-6)
    if name == "promotion_time":
        return PromotionTimeCure(max_em_iter=300, tol=1e-6)
    effective_family = (
        "poisson" if latent_distribution == "negative_binomial" and phi == 0
        else latent_distribution
    )
    parameters: dict[str, Any] = dict(latent_distribution=effective_family, K_bin=K_bin)
    if name == "ggkm":
        return GGKM(gamma=.3, lambda_reg=.1, max_em_iter=100, tol=1e-6, **parameters)
    if name == "gradient_boosting":
        return GradientBoostingCure(
            n_estimators=5, max_total_estimators=100, max_em_iter=100,
            max_depth=2, tol=1e-6, random_state=random_state, **parameters,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}.")


def _count_family_matches(name, family, trials, phi):
    if name in ("ggkm", "gradient_boosting"):
        return True
    if name == "mixture":
        return family == "binomial" and trials == 1
    return family == "poisson" or (family == "negative_binomial" and phi == 0)


def _versions():
    import skcure

    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "skcure_source_version": skcure.__version__,
        "package_versions_scope": "installed distribution metadata; loaded source version recorded separately",
    }
    for package in ("skcure", "numpy", "scipy", "scikit-learn", "scikit-survival"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def _metric_summary(records, metric_names):
    """Summarize each metric using its actual number of available replicates."""
    result = {}
    for name in metric_names:
        values = np.asarray([item[name] for item in records if name in item], dtype=float)
        result[name] = float(values.mean()) if len(values) else None
        result[f"{name}_mcse"] = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
        )
        result[f"{name}_n"] = len(values)
    return result


def _observed_metrics(model, X, y, y_train, times):
    result = evaluate_survival(model, X, y, y_train=y_train, times=times)
    metrics = {name: float(getattr(result, name)) for name in OBSERVED_METRICS}
    if not np.all(np.isfinite(list(metrics.values()))):
        raise ValueError("Observed-data evaluation returned non-finite metrics.")
    return metrics | {"auc": result.auc.tolist(), "times": result.times.tolist()}


def run_benchmark(
    *, repetitions: int = 10, sample_sizes: Sequence[int] = (200, 1000),
    censoring_rates: Sequence[float] = (.05, .5), followups: Sequence[float] = (1., 5.),
    latencies: Sequence[str] = ("weibull", "lognormal"), random_state: int = 73,
    model_names: Sequence[str] = ("mixture", "promotion_time"),
    latent_distributions: Sequence[str] = LATENT_DISTRIBUTIONS,
    cure_fractions: Sequence[float] = (.3,), K_bin: int = 1, phi: float = .5,
    test_size: int = 1000,
) -> dict[str, Any]:
    """Return all attempts plus converged-fit and all-fitted accuracy summaries.

    GGKM and boosting match the generating count family. Linear mixture and
    promotion-time fits provide matched and deliberately misspecified cases.
    Negative-binomial dispersion is estimated when fitting flexible models.
    Each scenario uses independent train and test cohorts with the same truth,
    and all estimators in a repetition receive those same cohorts.

    Calibration error here is the difference between mean predicted survival
    and mean known survival over a grid. Observed-data IPCW/AUC/Brier errors are
    recorded separately, without dropping unsupported test observations.
    """
    if not isinstance(repetitions, Integral) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer.")
    if not isinstance(test_size, Integral) or isinstance(test_size, bool) or test_size < 2:
        raise ValueError("test_size must be an integer of at least two.")
    if not model_names or any(name not in MODEL_NAMES for name in model_names):
        raise ValueError(f"model_names must select one or more of {MODEL_NAMES}.")
    if not latent_distributions or any(name not in LATENT_DISTRIBUTIONS for name in latent_distributions):
        raise ValueError(f"latent_distributions must select from {LATENT_DISTRIBUTIONS}.")
    if any(not values for values in (sample_sizes, censoring_rates, followups, latencies, cure_fractions)):
        raise ValueError("Every scenario dimension must contain at least one value.")
    rng = np.random.default_rng(random_state)
    records: list[dict[str, Any]] = []
    scenarios = product(
        sample_sizes, censoring_rates, followups, latencies,
        latent_distributions, cure_fractions, range(repetitions),
    )
    for n, rate, followup, latency, family, fraction, repetition in scenarios:
        train_seed, test_seed, fit_seed = map(
            int, rng.integers(0, np.iinfo(np.int32).max, size=3)
        )
        generator: dict[str, Any] = dict(
            censoring_rate=rate, followup=followup, latency=latency,
            latent_distribution=family, cure_fraction=fraction, K_bin=K_bin, phi=phi,
        )
        train = simulate_cure_survival(n, random_state=train_seed, **generator)
        test = simulate_cure_survival(test_size, random_state=test_seed, **generator)
        if train.incidence_coefficients is None:
            raise RuntimeError("The simulator did not provide incidence coefficient truth.")
        times = np.linspace(.05 * followup, followup, 20)
        observed_times = np.linspace(.05 * followup, .95 * followup, 20)
        truth = test.survival_probability(times)
        for name in model_names:
            estimator = _make_model(
                name, latent_distribution=family, K_bin=K_bin, phi=phi,
                random_state=fit_seed,
            )
            count_matches = _count_family_matches(name, family, K_bin, phi)
            record: dict[str, Any] = dict(
                n_samples=int(n), censoring_rate=float(rate), followup=float(followup),
                latency=latency, latent_distribution=family, cure_fraction=float(fraction),
                K_bin=int(K_bin), phi=float(phi), repetition=repetition, seed=train_seed,
                train_seed=train_seed, test_seed=test_seed, fit_seed=fit_seed, model=name,
                censoring_proportion=float(1 - train.y["event"].mean()),
                count_family_correct=count_matches,
                latency_family_correct=latency == "weibull",
                distribution_families_match=count_matches and latency == "weibull",
                # Finite tree/kernel fits approximate the linear truth; a
                # matching count family alone does not establish exact model
                # specification for those incidence function classes.
                correctly_specified=(
                    count_matches and latency == "weibull"
                    if name in ("mixture", "promotion_time") else None
                ),
                converged=False, failed=False, error=None, warnings=[],
                observed_metric_error=None,
                estimator_parameters=estimator.get_params(deep=False),
                generating_parameters={
                    "incidence_intercept": train.incidence_intercept,
                    "incidence_coefficients": train.incidence_coefficients.tolist(),
                    "latent_time_shape": train.latency_shape,
                    "latent_time_scale": 1.0,
                    "latent_time_log_location": 0.0 if latency == "lognormal" else None,
                },
            )
            start = time.perf_counter()
            captured = []
            model = None
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    model = estimator.fit(train.X, train.y)
                    record.update(
                        converged=bool(model.converged_), n_iter=int(model.n_iter_),
                        termination_reason=getattr(model, "termination_reason_", None),
                    )
                    cure = np.asarray(model.predict_cure_probability(test.X), dtype=float)
                    survival = np.asarray(model.predict_survival(test.X, times), dtype=float)
                    if (
                        cure.shape != test.cure_probability.shape
                        or survival.shape != truth.shape
                        or not np.all(np.isfinite(cure)) or not np.all(np.isfinite(survival))
                        or np.any((cure < 0) | (cure > 1))
                        or np.any((survival < 0) | (survival > 1))
                    ):
                        raise ValueError("The fitted model returned invalid probabilities.")
                    record.update(
                        cure_bias=float(np.mean(cure - test.cure_probability)),
                        cure_rmse=float(np.sqrt(np.mean((cure - test.cure_probability) ** 2))),
                        survival_rmse=float(np.sqrt(np.mean((survival - truth) ** 2))),
                        survival_calibration_error=float(np.mean(np.abs(survival.mean(axis=0) - truth.mean(axis=0)))),
                    )
                    try:
                        observed = _observed_metrics(
                            model, test.X, test.y, train.y, observed_times
                        )
                        record["observed_metrics"] = observed
                        record.update({metric: observed[metric] for metric in OBSERVED_METRICS})
                    except Exception as exc:
                        record["observed_metric_error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                record.update(failed=True, error=f"{type(exc).__name__}: {exc}")
                if model is None:
                    record["termination_reason"] = getattr(estimator, "termination_reason_", None)
            record["warnings"] = [f"{item.category.__name__}: {item.message}" for item in captured]
            record["seconds"] = time.perf_counter() - start
            records.append(record)
    summaries = []
    grouping = (
        "n_samples", "censoring_rate", "followup", "latency",
        "latent_distribution", "cure_fraction", "K_bin", "phi", "model",
    )
    keys = {tuple(record[key] for key in grouping) for record in records}
    for key in sorted(keys):
        group = [record for record in records if tuple(record[name] for name in grouping) == key]
        fitted = [record for record in group if not record["failed"]]
        usable = [record for record in fitted if record["converged"]]
        summary: dict[str, Any] = dict(zip(grouping, key))
        summary.update(
            attempts=len(group), failure_rate=float(np.mean([record["failed"] for record in group])),
            nonconvergence_rate=float(np.mean([not record["failed"] and not record["converged"] for record in group])),
            observed_metric_failure_rate=float(np.mean([record["observed_metric_error"] is not None for record in group])),
            usable=len(usable), fitted=len(fitted),
            accuracy_population="converged fits",
            correctly_specified=group[0]["correctly_specified"],
            distribution_families_match=group[0]["distribution_families_match"],
        )
        summary.update(_metric_summary(usable, TRUTH_METRICS + OBSERVED_METRICS))
        summary["all_fitted"] = _metric_summary(fitted, TRUTH_METRICS + OBSERVED_METRICS)
        summaries.append(summary)
    return {
        "configuration": {
            "random_state": random_state, "repetitions": repetitions,
            "sample_sizes": list(sample_sizes), "censoring_rates": list(censoring_rates),
            "followups": list(followups), "latencies": list(latencies),
            "latent_distributions": list(latent_distributions), "cure_fractions": list(cure_fractions),
            "K_bin": K_bin, "phi": phi, "test_size": test_size, "models": list(model_names),
            "truth": "linear incidence with latent-count PGF and independent censoring",
            "mcse": "sample standard deviation / sqrt(number of available independent repetitions)",
        },
        "environment": _versions(), "summaries": summaries, "attempts": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("simulation-results.json"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["mixture", "promotion_time"])
    parser.add_argument("--latent-distributions", nargs="+", choices=LATENT_DISTRIBUTIONS, default=list(LATENT_DISTRIBUTIONS))
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[200, 1000])
    parser.add_argument("--censoring-rates", nargs="+", type=float, default=[.05, .5])
    parser.add_argument("--followups", nargs="+", type=float, default=[1., 5.])
    parser.add_argument("--latencies", nargs="+", choices=["weibull", "lognormal"], default=["weibull", "lognormal"])
    parser.add_argument("--cure-fractions", nargs="+", type=float, default=[.3])
    parser.add_argument("--binomial-trials", dest="K_bin", type=int, default=1)
    parser.add_argument("--phi", type=float, default=.5)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=73)
    options = parser.parse_args()
    configuration = vars(options).copy()
    output = configuration.pop("output")
    quick = configuration.pop("quick")
    configuration["model_names"] = configuration.pop("models")
    if quick:
        configuration.update(
            repetitions=1, sample_sizes=(100,), censoring_rates=(.1,),
            followups=(4.,), latencies=("weibull",),
        )
    result = run_benchmark(**configuration)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summaries"], indent=2))
    print(f"Saved every attempted fit to {output}.")


if __name__ == "__main__":
    main()
