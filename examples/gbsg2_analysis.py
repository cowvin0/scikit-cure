"""Training-only model selection and held-out evaluation on GBSG2.

Run ``python examples/gbsg2_analysis.py --output /tmp/gbsg2-results.json``.
``--quick`` retains the complete cohort while reducing CV folds and EM budgets.

GBSG2 is a recurrence-free survival dataset distributed with scikit-survival:
https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.datasets.load_gbsg2.html
The original study is Schumacher et al. (1994), doi:10.1200/JCO.1994.12.10.2086.

Before any splitting, all outcomes receive a prespecified six-year
administrative horizon. Subjects remain in the cohort; events after the horizon
become censored observations at six years. This defines the example's follow-up
window and keeps evaluation inside training censoring support. The event
indicator is True for an observed endpoint, including the source field named
``cens``; censored individuals do not have known cure status.

The example demonstrates a workflow, not clinical validation. Estimated cure
probabilities depend on model and extrapolation assumptions. Held-out survival
metrics within six years cannot establish the accuracy of long-term cure
probabilities or individual cure status.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.datasets import load_gbsg2
from sksurv.util import Surv, check_y_survival

import skcure
from skcure.linear_model import MixtureCure, PromotionTimeCure
from skcure.metrics import evaluate_survival


DAYS_PER_YEAR = 365.25
HORIZON_YEARS = 6.0
EVALUATION_TIMES = np.linspace(1.0, 5.0, 9)
DATASET_URL = (
    "https://scikit-survival.readthedocs.io/en/stable/api/generated/"
    "sksurv.datasets.load_gbsg2.html"
)


def administrative_targets(y: np.ndarray) -> np.ndarray:
    """Convert days to years and apply the fixed horizon without dropping rows."""
    event, time = check_y_survival(y)
    years = np.asarray(time, dtype=float) / DAYS_PER_YEAR
    return Surv.from_arrays(
        event & (years <= HORIZON_YEARS), np.minimum(years, HORIZON_YEARS)
    )


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build transformations whose imputation/scaling/levels are learned in fit."""
    numeric = list(X_train.select_dtypes(include="number").columns)
    categorical = [name for name in X_train.columns if name not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False,
            )),
        ]), categorical),
    ], sparse_threshold=0)


def make_candidates(X_train: pd.DataFrame, *, quick: bool) -> dict[str, Pipeline]:
    """Model family is the selection parameter; no test outcomes are inspected."""
    options = {"max_em_iter": 300 if quick else 1000, "em_rtol": 1e-7}
    return {
        name: Pipeline([
            ("preprocess", make_preprocessor(X_train)),
            ("model", estimator),
        ])
        for name, estimator in (
            ("mixture", MixtureCure(**options)),
            ("promotion_time", PromotionTimeCure(**options)),
        )
    }


def _fit(
    candidate: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    X_validation: pd.DataFrame | None = None,
    y_validation: np.ndarray | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Retain warnings and diagnostics even when a fit or validation score fails."""
    fitted = clone(candidate)
    if not isinstance(fitted, Pipeline):
        raise TypeError("The candidate must clone to a scikit-learn Pipeline.")
    captured: list[Any] = []
    result: dict[str, Any] = {"fit_succeeded": False, "error": None, "score": None}
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted.fit(X_train, y_train)
            result["fit_succeeded"] = True
            if X_validation is not None and y_validation is not None:
                score = float(fitted.score(X_validation, y_validation))
                if not np.isfinite(score):
                    raise FloatingPointError("Validation concordance is not finite.")
                result["score"] = score
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    model = fitted.named_steps["model"]
    result.update(
        converged=bool(getattr(model, "converged_", False)),
        termination_reason=getattr(model, "termination_reason_", "not_fitted"),
        n_iter=int(getattr(model, "n_iter_", 0)),
        final_loglik=getattr(model, "final_loglik_", None),
        loglik_history=list(getattr(model, "loglik_history_", [])),
        optimizer_results=list(getattr(model, "optimizer_results_", [])),
        warnings=[f"{item.category.__name__}: {item.message}" for item in captured],
    )
    result["usable"] = bool(
        result["fit_succeeded"] and result["error"] is None and result["converged"]
    )
    return fitted, result


def _json_ready(value: Any) -> Any:
    """Convert NumPy scalars/arrays; encode unavailable values as null, not NaN."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_analysis(*, quick: bool = False, random_state: int = 73) -> dict[str, Any]:
    """Return a reproducible model-selection/evaluation report for all 686 rows.

    Each candidate must complete every training-only CV fold with convergence
    to be eligible. Its mean held-out-fold Harrell concordance selects the
    family. Failed/nonconverged folds remain in the report. The test cohort is
    evaluated once, after selection and a successful full-training refit.
    """
    X, original_y = load_gbsg2()
    if not isinstance(X, pd.DataFrame) or original_y is None:
        raise TypeError("GBSG2 must provide a pandas DataFrame and survival outcomes.")
    original_y = np.asarray(original_y)
    y = administrative_targets(original_y)
    event = y["event"]
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=random_state, stratify=event,
    )
    X_train, y_train = X.iloc[train], y[train]
    n_folds = 2 if quick else 3
    splits = list(StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state,
    ).split(X_train, y_train["event"]))
    candidates = make_candidates(X_train, quick=quick)
    _, original_time = check_y_survival(original_y)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "no_eligible_candidate",
        "dataset": {
            "name": "GBSG2", "loader": "sksurv.datasets.load_gbsg2",
            "source_url": DATASET_URL,
            "reference_doi": "10.1200/JCO.1994.12.10.2086",
            "n_subjects": len(y), "n_features": X.shape[1],
            "event_definition": "Observed recurrence-free-survival endpoint; True indicates an event.",
            "time_units": "years", "days_per_year": DAYS_PER_YEAR,
            "administrative_horizon_years": HORIZON_YEARS,
            "administratively_truncated_subjects": int(np.sum(original_time / DAYS_PER_YEAR > HORIZON_YEARS)),
            "subjects_dropped": 0,
            "original_events": int(check_y_survival(original_y)[0].sum()),
            "events_within_horizon": int(event.sum()),
        },
        "versions": {
            "python": platform.python_version(), "skcure": skcure.__version__,
            **{name: version(name) for name in (
                "numpy", "scipy", "pandas", "scikit-learn", "scikit-survival",
            )},
        },
        "configuration": {
            "random_state": random_state, "quick": quick, "cv_folds": n_folds,
            "test_fraction": 0.25,
            "selection_metric": "Mean validation-fold Harrell concordance; larger is better.",
            "eligibility": "All CV folds must fit, converge, and return finite scores.",
            "tie_break": "First candidate in the reported candidate order.",
            "evaluation_times_years": EVALUATION_TIMES,
            "preprocessing": {
                "numeric": "Training-fold median imputation then standard scaling.",
                "categorical": "Training-fold mode imputation then one-hot encoding; first level dropped, unknown levels ignored.",
            },
        },
        "split": {
            "train_subject_indices": train, "test_subject_indices": test,
            "n_train": len(train), "n_test": len(test),
            "train_events": int(event[train].sum()), "test_events": int(event[test].sum()),
            "train_max_followup_years": float(y_train["time"].max()),
            "test_max_followup_years": float(y["time"][test].max()),
            "folds": [{
                "train_subject_indices": train[fit],
                "validation_subject_indices": train[validation],
            } for fit, validation in splits],
        },
        "candidates": [], "selected_model": None, "refit": None,
        "evaluation": None, "evaluation_warnings": [], "predictions": None,
        "interpretation": [
            "Censored subjects do not have known cure status.",
            "Cure probabilities are model-based extrapolations, not validated individual cure labels.",
            "Within-window survival evaluation does not establish long-term cure-probability accuracy.",
            "IPCW metrics use marginal independent-censoring assumptions and need censoring support.",
            "This example demonstrates software use; it does not establish clinical validity or treatment effects.",
        ],
    }
    for name, candidate in candidates.items():
        record: dict[str, Any] = {
            "model": name, "parameters": candidate.named_steps["model"].get_params(),
            "folds": [], "eligible": False, "mean_cv_concordance": None,
        }
        for number, (fit, validation) in enumerate(splits):
            _, outcome = _fit(
                candidate, X_train.iloc[fit], y_train[fit],
                X_validation=X_train.iloc[validation], y_validation=y_train[validation],
            )
            record["folds"].append({"fold": number, **outcome})
        record["eligible"] = all(item["usable"] for item in record["folds"])
        if record["eligible"]:
            record["mean_cv_concordance"] = float(np.mean([
                item["score"] for item in record["folds"]
            ]))
        report["candidates"].append(record)
    eligible = [item for item in report["candidates"] if item["eligible"]]
    if not eligible:
        return _json_ready(report)
    selected = max(eligible, key=lambda item: item["mean_cv_concordance"])
    name = selected["model"]
    report["selected_model"] = name
    fitted, diagnostics = _fit(candidates[name], X_train, y_train)
    report["refit"] = diagnostics
    if not diagnostics["usable"]:
        report["status"] = "unusable_refit"
        return _json_ready(report)
    evaluation_warnings: list[Any] = []
    try:
        with warnings.catch_warnings(record=True) as evaluation_warnings:
            warnings.simplefilter("always")
            evaluation = evaluate_survival(
                fitted, X.iloc[test], y[test], y_train=y_train, times=EVALUATION_TIMES,
            )
            report["evaluation"] = asdict(evaluation)
            examples = test[:5]
            transformed = fitted.named_steps["preprocess"].transform(X.iloc[examples])
            model = fitted.named_steps["model"]
            report["predictions"] = {
                "subject_indices": examples,
                "risk": model.predict(transformed),
                "cure_probability": model.predict_cure_probability(transformed),
                "survival_probability": model.predict_survival(transformed, EVALUATION_TIMES),
                "times_years": EVALUATION_TIMES,
                "transformed_feature_names": fitted[:-1].get_feature_names_out(),
            }
            report["status"] = "ok"
    except Exception as exc:
        report["status"] = "evaluation_failed"
        report["evaluation_error"] = f"{type(exc).__name__}: {exc}"
    report["evaluation_warnings"] = [
        f"{item.category.__name__}: {item.message}" for item in evaluation_warnings
    ]
    return _json_ready(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use two CV folds and a 300-iteration EM budget; retain all 686 subjects.")
    parser.add_argument("--random-state", type=int, default=73)
    parser.add_argument("--output", type=Path, default=Path("gbsg2-results.json"))
    arguments = parser.parse_args(argv)
    report = run_analysis(quick=arguments.quick, random_state=arguments.random_state)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"GBSG2: {report['split']['n_train']} training / {report['split']['n_test']} test subjects; no subjects dropped.")
    for candidate in report["candidates"]:
        if candidate["eligible"]:
            print(f"Candidate {candidate['model']}: mean CV concordance={candidate['mean_cv_concordance']:.3f}.")
        else:
            failures = sum(item["error"] is not None for item in candidate["folds"])
            nonconverged = sum(item["fit_succeeded"] and not item["converged"] for item in candidate["folds"])
            print(f"Candidate {candidate['model']}: excluded ({failures} failed, {nonconverged} nonconverged CV folds); details retained in JSON.")
    print(f"Status: {report['status']}; selected model: {report['selected_model']}.")
    if report["evaluation"] is not None:
        evaluation = report["evaluation"]
        print(f"Held-out IPCW concordance={evaluation['concordance_ipcw']:.3f}, mean AUC={evaluation['mean_auc']:.3f}, integrated Brier score={evaluation['integrated_brier_score']:.3f}.")
    print("Cure probabilities are model-based extrapolations; individual cure status is unknown.")
    print(f"Saved fold diagnostics, settings, predictions, and provenance to {arguments.output}.")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
