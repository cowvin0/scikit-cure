"""Repeated-study bookkeeping, specification labels, and failure visibility."""

import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning


@pytest.fixture
def benchmark():
    path = Path(__file__).parents[1] / "benchmarks" / "simulation.py"
    spec = importlib.util.spec_from_file_location("skcure_scientific_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ControlledEstimator:
    def __init__(self, outcome="converged"):
        self.outcome = outcome

    def get_params(self, deep=False):
        return {"outcome": self.outcome}

    def fit(self, X, y):
        self.termination_reason_ = self.outcome
        if self.outcome == "failed":
            raise RuntimeError("deliberate fit failure")
        self.n_iter_ = 2
        self.converged_ = self.outcome == "converged"
        if not self.converged_:
            warnings.warn("deliberate nonconvergence", ConvergenceWarning)
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_cure_probability(self, X):
        return np.full(len(X), .3)

    def predict_survival(self, X, times):
        return np.tile(.3 + .7 * np.exp(-np.asarray(times)), (len(X), 1))


def test_study_retains_all_attempts_and_computes_repetition_mcse(benchmark, monkeypatch):
    outcomes = {"mixture": "converged", "promotion_time": "max_em_iter", "ggkm": "failed"}
    monkeypatch.setattr(
        benchmark, "_make_model",
        lambda name, **kwargs: _ControlledEstimator(outcomes[name]),
    )
    result = benchmark.run_benchmark(
        repetitions=3, sample_sizes=(80,), censoring_rates=(.1,),
        followups=(5,), latencies=("weibull",), latent_distributions=("binomial",),
        model_names=tuple(outcomes), test_size=100,
    )
    assert len(result["attempts"]) == 9
    summaries = {item["model"]: item for item in result["summaries"]}
    assert summaries["mixture"]["usable"] == 3
    assert summaries["mixture"]["correctly_specified"]
    assert summaries["promotion_time"]["nonconvergence_rate"] == 1
    assert summaries["promotion_time"]["usable"] == 0
    assert summaries["promotion_time"]["all_fitted"]["cure_rmse_n"] == 3
    assert not summaries["promotion_time"]["correctly_specified"]
    assert summaries["ggkm"]["failure_rate"] == 1
    assert summaries["ggkm"]["cure_rmse"] is None
    assert summaries["ggkm"]["correctly_specified"] is None
    values = [item["cure_rmse"] for item in result["attempts"] if item["model"] == "mixture"]
    assert summaries["mixture"]["cure_rmse"] == pytest.approx(np.mean(values))
    assert summaries["mixture"]["cure_rmse_mcse"] == pytest.approx(np.std(values, ddof=1) / np.sqrt(3))
    for item in result["attempts"]:
        assert {"train_seed", "test_seed", "fit_seed", "estimator_parameters", "warnings", "seconds"} <= item.keys()
        if item["model"] == "ggkm":
            assert "deliberate fit failure" in item["error"]
        if item["model"] == "promotion_time":
            assert "deliberate nonconvergence" in item["warnings"][0]
    assert "python" in result["environment"]
    json.dumps(result, allow_nan=False)


def test_unsupported_observed_metrics_do_not_discard_fit_or_truth_errors(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark, "_make_model", lambda *args, **kwargs: _ControlledEstimator())

    def unsupported(*args, **kwargs):
        raise ValueError("unsupported censoring support")

    monkeypatch.setattr(benchmark, "_observed_metrics", unsupported)
    result = benchmark.run_benchmark(
        repetitions=1, sample_sizes=(40,), censoring_rates=(.1,), followups=(5,),
        latencies=("weibull",), latent_distributions=("poisson",),
        model_names=("promotion_time",), test_size=30,
    )
    attempt = result["attempts"][0]
    assert not attempt["failed"]
    assert attempt["converged"]
    assert "cure_rmse" in attempt
    assert "unsupported censoring support" in attempt["observed_metric_error"]
    summary = result["summaries"][0]
    assert summary["observed_metric_failure_rate"] == 1
    assert summary["integrated_brier_score"] is None
    assert summary["cure_rmse_mcse"] is None  # one repetition cannot estimate MCSE


@pytest.mark.parametrize("family", ["binomial", "poisson", "negative_binomial"])
def test_flexible_model_factory_matches_truth_count_parameters(benchmark, family):
    for name in ("ggkm", "gradient_boosting"):
        model = benchmark._make_model(name, latent_distribution=family, K_bin=3, random_state=41)
        assert model.latent_distribution == family
        assert model.K_bin == 3
        if name == "gradient_boosting":
            assert model.random_state == 41


@pytest.mark.parametrize("kwargs", [
    {"repetitions": 0}, {"repetitions": True}, {"model_names": ()},
    {"latent_distributions": ("invalid",)}, {"test_size": 1}, {"sample_sizes": ()},
])
def test_invalid_benchmark_configuration_is_rejected(benchmark, kwargs):
    with pytest.raises(ValueError):
        benchmark.run_benchmark(**kwargs)
