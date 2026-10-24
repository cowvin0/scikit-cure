"""Real-cohort evaluation support, split isolation, and failure reporting."""

from collections import Counter
import json
import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sksurv.util import Surv

from examples import gbsg2_analysis as example
from skcure.linear_model import MixtureCure


def test_administrative_horizon_preserves_subjects_events_at_boundary_and_input():
    source = Surv.from_arrays(
        [True, False, True, True, False],
        np.array([1.0, 5.0, 6.0, 6.5, 7.0]) * example.DAYS_PER_YEAR,
        name_event="cens",
    )
    original = source.copy()
    target = example.administrative_targets(source)
    assert len(target) == len(source)
    np.testing.assert_array_equal(target["event"], [True, False, True, False, False])
    np.testing.assert_allclose(target["time"], [1.0, 5.0, 6.0, 6.0, 6.0])
    np.testing.assert_array_equal(source, original)


def test_quick_real_cohort_analysis_fits_only_training_folds_and_evaluates_once(
    monkeypatch,
):
    preprocessing_rows = []
    fit_calls = []
    evaluations = []
    original_preprocess = ColumnTransformer.fit_transform
    original_fit = example._fit
    original_evaluate = example.evaluate_survival

    def record_preprocessing(self, X, y=None, **kwargs):
        preprocessing_rows.append(frozenset(X.index))
        return original_preprocess(self, X, y, **kwargs)

    def record_fit(candidate, X, y, **kwargs):
        validation = kwargs.get("X_validation")
        fit_calls.append(
            (
                frozenset(X.index),
                None if validation is None else frozenset(validation.index),
            )
        )
        return original_fit(candidate, X, y, **kwargs)

    def record_evaluation(model, X, y, **kwargs):
        evaluations.append((frozenset(X.index), y.copy(), kwargs["y_train"].copy()))
        return original_evaluate(model, X, y, **kwargs)

    monkeypatch.setattr(ColumnTransformer, "fit_transform", record_preprocessing)
    monkeypatch.setattr(example, "_fit", record_fit)
    monkeypatch.setattr(example, "evaluate_survival", record_evaluation)
    report = example.run_analysis(quick=True, random_state=73)

    assert report["status"] == "ok"
    assert report["dataset"]["n_subjects"] == 686
    assert report["dataset"]["subjects_dropped"] == 0
    assert (
        report["dataset"]["events_within_horizon"]
        <= report["dataset"]["original_events"]
    )
    train = frozenset(report["split"]["train_subject_indices"])
    test = frozenset(report["split"]["test_subject_indices"])
    assert len(train) == 514
    assert len(test) == 172
    assert train.isdisjoint(test)
    assert train | test == frozenset(range(686))

    expected_preprocessing = [train]
    expected_fold_pairs = []
    for fold in report["split"]["folds"]:
        fit = frozenset(fold["train_subject_indices"])
        validation = frozenset(fold["validation_subject_indices"])
        assert fit.isdisjoint(validation)
        assert fit | validation == train
        expected_fold_pairs.append((fit, validation))
        expected_preprocessing.extend([fit] * len(report["candidates"]))
    assert Counter(preprocessing_rows) == Counter(expected_preprocessing)
    assert Counter(fit_calls) == Counter(
        expected_fold_pairs * len(report["candidates"]) + [(train, None)]
    )
    assert all(rows.isdisjoint(test) for rows in preprocessing_rows)
    assert len(evaluations) == 1
    assert evaluations[0][0] == test

    for candidate in report["candidates"]:
        assert len(candidate["folds"]) == report["configuration"]["cv_folds"]
        assert candidate["parameters"]["max_em_iter"] == 300
    eligible = [item for item in report["candidates"] if item["eligible"]]
    selected = next(
        item for item in eligible if item["model"] == report["selected_model"]
    )
    assert selected["mean_cv_concordance"] == max(
        item["mean_cv_concordance"] for item in eligible
    )
    assert report["refit"]["usable"]
    metrics = report["evaluation"]
    assert 0 <= metrics["concordance_ipcw"] <= 1
    assert 0 <= metrics["mean_auc"] <= 1
    assert 0 <= metrics["integrated_brier_score"] <= 1
    times = np.asarray(metrics["times"])
    assert np.all(np.diff(times) > 0)
    assert times[-1] < report["split"]["train_max_followup_years"]
    assert times[-1] < report["split"]["test_max_followup_years"]
    cure = np.asarray(report["predictions"]["cure_probability"])
    survival = np.asarray(report["predictions"]["survival_probability"])
    assert np.all((cure >= 0) & (cure <= 1))
    assert np.all((survival >= 0) & (survival <= 1))
    assert np.all(np.diff(survival, axis=1) <= 1e-12)
    assert np.all(survival >= cure[:, None] - 1e-12)
    assert report["interpretation"]
    assert json.loads(json.dumps(report, allow_nan=False))["status"] == "ok"


class _FailingMixture(MixtureCure):
    def fit(self, X, y):
        warnings.warn("Controlled fit warning", UserWarning)
        raise RuntimeError("Controlled candidate failure")


def test_failed_and_nonconverged_candidates_are_saved_without_test_evaluation(
    monkeypatch, tmp_path
):
    def candidates(X_train, *, quick):
        return {
            "failed": Pipeline(
                [
                    ("preprocess", example.make_preprocessor(X_train)),
                    ("model", _FailingMixture(max_em_iter=1)),
                ]
            ),
            "capped": Pipeline(
                [
                    ("preprocess", example.make_preprocessor(X_train)),
                    ("model", MixtureCure(max_em_iter=1, tol=1e-30)),
                ]
            ),
        }

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError(
            "No held-out evaluation is allowed without an eligible candidate."
        )

    monkeypatch.setattr(example, "make_candidates", candidates)
    monkeypatch.setattr(example, "evaluate_survival", unexpected_evaluation)
    output = tmp_path / "new-directory" / "failed-analysis.json"
    assert example.main(["--quick", "--output", str(output)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "no_eligible_candidate"
    assert report["selected_model"] is None
    assert report["evaluation"] is None
    assert report["refit"] is None
    assert len(report["candidates"]) == 2
    failed, capped = report["candidates"]
    for record in failed["folds"]:
        assert not record["fit_succeeded"]
        assert "Controlled candidate failure" in record["error"]
        assert any("Controlled fit warning" in item for item in record["warnings"])
    for record in capped["folds"]:
        assert record["fit_succeeded"]
        assert not record["converged"]
        assert record["termination_reason"] == "max_em_iter"
        assert np.isfinite(record["score"])
        assert any("ConvergenceWarning" in item for item in record["warnings"])
    assert all(
        not item["eligible"] and item["mean_cv_concordance"] is None
        for item in report["candidates"]
    )
