from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

import numpy as np
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning

_Parameters = TypeVar("_Parameters")
_Expected = TypeVar("_Expected")


def _em_converged(old: float, new: float, *, tol: float, rtol: float = 0.0) -> bool:
    """Recognize a small nonnegative improvement in observed likelihood.

    The relative criterion is invariant to multiplying the log likelihood by
    the sample size. A decrease is retained in history but never labeled as
    convergence, even when it is smaller than the requested tolerance.
    """
    improvement = new - old
    return bool(
        np.isfinite(improvement)
        and improvement >= 0.0
        and (
            improvement < tol
            or (rtol > 0.0 and improvement < rtol * max(abs(old), abs(new)))
        )
    )


def _warn_em_limit(max_iter: int) -> None:
    warnings.warn(
        f"EM did not converge within {max_iter} iterations. "
        "Inspect loglik_history_ and optimizer_results_, or increase max_em_iter.",
        ConvergenceWarning,
        stacklevel=3,
    )


def _solver_options(model, stage: str) -> dict:
    """Copy validated per-estimator settings without mutating user mappings."""
    settings = getattr(model, "solver_options", None)
    return {} if settings is None else dict(settings.get(stage, {}))


def _record_optimizer_result(model, stage: str, result) -> None:
    """Keep compact diagnostics, including unsuccessful optimizer attempts."""
    if not hasattr(model, "optimizer_results_"):
        model.optimizer_results_ = []
    diagnostics = {
        "stage": stage,
        "iteration": getattr(model, "n_iter_", 0),
        "success": bool(result.success),
        "status": getattr(result, "status", None),
        "message": str(getattr(result, "message", "")),
        "objective": float(getattr(result, "fun", np.nan)),
        "n_iter": getattr(result, "nit", None),
        "n_function_evaluations": getattr(result, "nfev", None),
        "n_gradient_evaluations": getattr(result, "njev", None),
    }
    model.optimizer_results_.append(diagnostics)


class _EMState(Protocol):
    """Mutable progress initialized by an estimator before each fit."""

    @property
    def n_iter(self) -> int: ...

    @n_iter.setter
    def n_iter(self, value: int) -> None: ...

    @property
    def converged(self) -> bool: ...

    @converged.setter
    def converged(self, value: bool) -> None: ...

    @property
    def loglik_history(self) -> list[float]: ...

    @property
    def termination_reason(self) -> str: ...

    @termination_reason.setter
    def termination_reason(self, value: str) -> None: ...


class _EstimatorProgress(Protocol):
    """Fitted progress attributes exposed by the estimator state adapter."""

    n_iter_: int
    converged_: bool
    loglik_history_: list[float]
    termination_reason_: str


def run_em(
    initial_parameters: _Parameters,
    *,
    e_step: Callable[[_Parameters], _Expected],
    m_steps: Sequence[Callable[[_Parameters, _Expected], _Parameters]],
    loglikelihood: Callable[[_Parameters], float],
    state: _EMState,
    tol: float,
    max_iter: int,
    rtol: float = 0.0,
    validate_initial_loglik: Callable[[float], None] | None = None,
    nonfinite_message: str = "Non-finite log-likelihood.",
) -> tuple[_Parameters, float]:
    """Run EM using model-supplied callbacks and return parameters and likelihood.

    ``state`` supplies an initialized iteration count, convergence flag, and
    history. The count advances before each E-step, and history records only
    completed iterations with finite likelihoods. This retains useful progress
    when a callback raises. All M-steps receive the same posterior expectations
    and the parameters returned by the preceding step.

    Convergence requires a nonnegative improvement smaller than ``tol`` or
    the optional relative threshold. It is checked before the iteration limit.
    The initial likelihood is evaluated
    once and optionally validated by the model; it is not added to history.
    Callers validate the tolerance and resolve a positive iteration limit.
    """
    parameters = initial_parameters
    state.termination_reason = "running"
    try:
        ll_old = loglikelihood(parameters)
        if validate_initial_loglik is not None:
            validate_initial_loglik(ll_old)
        ll_new = ll_old

        while True:
            state.n_iter += 1
            expected = e_step(parameters)
            for m_step in m_steps:
                parameters = m_step(parameters, expected)

            ll_new = loglikelihood(parameters)
            if not np.isfinite(ll_new):
                raise FloatingPointError(nonfinite_message)
            state.loglik_history.append(ll_new)

            if _em_converged(ll_old, ll_new, tol=tol, rtol=rtol):
                state.converged = True
                state.termination_reason = "converged"
                break
            if state.n_iter >= max_iter:
                state.termination_reason = "max_em_iter"
                _warn_em_limit(max_iter)
                break
            ll_old = ll_new
    except Exception:
        state.termination_reason = "failed"
        raise

    return parameters, ll_new


class _GeneralizedGammaEMSteps:
    """Share generalized-gamma latency maximization across EM estimators."""

    def __init__(self, model, t, delta):
        self.model = model
        self.t = t
        self.delta = delta

    def maximize_latency(self, parameters, expected):
        self.model._latency_optimizer.maximize(
            self.model._latency,
            self.model._latent_model,
            self.model._gg_log_value_and_grad,
            self.t,
            self.delta,
            expected,
            options=_solver_options(self.model, "latency"),
            result_callback=lambda result: _record_optimizer_result(
                self.model, "latency", result
            ),
        )
        return parameters


class _GGKMSteps(_GeneralizedGammaEMSteps):
    """Orchestrate GGKM likelihood, predictor, latency, and dispersion updates."""

    def loglikelihood(self, parameters):
        beta0, alpha = parameters
        return self.model._penalized_loglikelihood(beta0, alpha, self.t, self.delta)

    def expectation(self, parameters):
        beta0, alpha = parameters
        return self.model._e_step(beta0, alpha, self.t, self.delta)

    def maximize_predictor(self, parameters, Nhat):
        beta0, alpha = parameters
        predictor_pars = np.concatenate(
            (
                np.array(
                    [beta0],
                    dtype=float,
                ),
                alpha,
            )
        )
        result_predictor = minimize(
            fun=self.model._predictor_value_and_grad,
            x0=predictor_pars,
            jac=True,
            args=(Nhat,),
            method="L-BFGS-B",
            options=_solver_options(self.model, "incidence"),
        )
        _record_optimizer_result(self.model, "incidence", result_predictor)
        self.model._validate_predictor_result(result_predictor)
        beta0 = float(result_predictor.x[0])
        alpha = result_predictor.x[1:].copy()
        return beta0, alpha

    def maximize_dispersion(self, parameters, Nhat):
        beta0, alpha = parameters
        self.model._update_phi(beta0, alpha, self.t, self.delta)
        return parameters


def fit_ggkm_em(model, t, delta):
    """Fit GGKM after validation and kernel setup, preserving its EM updates."""
    n = len(t)
    alpha = np.zeros(n, dtype=float)
    approximate_cure = np.clip(1.0 - np.mean(delta), 0.05, 0.95)
    beta0 = model._initial_beta0(approximate_cure)
    steps = _GGKMSteps(model, t, delta)
    (beta0, alpha), ll_new = run_em(
        (beta0, alpha),
        e_step=steps.expectation,
        m_steps=(
            steps.maximize_predictor,
            steps.maximize_latency,
            steps.maximize_dispersion,
        ),
        loglikelihood=steps.loglikelihood,
        state=_EstimatorEMState(model),
        tol=model.tol,
        max_iter=model.max_em_iter,
        rtol=model.em_rtol,
        validate_initial_loglik=model._validate_initial_loglik,
        nonfinite_message="Non-finite penalized log-likelihood.",
    )
    model.beta0_ = beta0
    model.alpha_ = alpha
    model.final_loglik_ = ll_new
    try:
        model._latent_model.finalize_fit()
    except Exception:
        model.termination_reason_ = "failed"
        raise


class _EstimatorEMState:
    """Expose an estimator's existing progress without copying fit state."""

    def __init__(self, model: _EstimatorProgress):
        self.model = model

    @property
    def n_iter(self) -> int:
        return self.model.n_iter_

    @n_iter.setter
    def n_iter(self, value: int) -> None:
        self.model.n_iter_ = value

    @property
    def converged(self) -> bool:
        return self.model.converged_

    @converged.setter
    def converged(self, value: bool) -> None:
        self.model.converged_ = value

    @property
    def loglik_history(self) -> list[float]:
        return self.model.loglik_history_

    @property
    def termination_reason(self) -> str:
        return self.model.termination_reason_

    @termination_reason.setter
    def termination_reason(self, value: str) -> None:
        self.model.termination_reason_ = value


class _LinearCureSteps(_GeneralizedGammaEMSteps):
    """Orchestrate EM for generalized-gamma models with linear incidence."""

    def maximize_latent_parameters(
        self,
        coefficients,
        expected,
    ):
        self.model._update_after_latency(
            coefficients,
            self.t,
            self.delta,
        )

        return coefficients

    def loglikelihood(self, coefficients):
        return self.model._loglikelihood(
            coefficients,
            self.t,
            self.delta,
        )

    def expectation(self, coefficients):
        return self.model._e_step(
            coefficients,
            self.t,
            self.delta,
        )

    def maximize_predictor(self, coefficients, expected):
        matrix = self.model.W_train_
        magnitude = np.max(np.abs(matrix), axis=0)
        magnitude = np.where(magnitude > 0.0, magnitude, 1.0)
        normalized = matrix / magnitude
        scales = magnitude * np.sqrt(np.mean(normalized * normalized, axis=0))
        scales = np.where(scales > 0.0, scales, 1.0)
        design = matrix / scales

        def _predictor_value_and_grad(working_coefficients, expected):
            return self.model._predictor_value_and_grad(
                working_coefficients, expected, design_matrix=design
            )

        result = minimize(
            fun=_predictor_value_and_grad,
            x0=coefficients * scales,
            jac=True,
            args=(expected,),
            method="L-BFGS-B",
            options=_solver_options(self.model, "incidence"),
        )
        _record_optimizer_result(self.model, "incidence", result)
        result.x = np.asarray(result.x, dtype=float) / scales
        self.model._validate_predictor_result(result)
        return result.x.copy()


def fit_linear_cure_em(model, t, delta):
    """Fit a linear-incidence cure model using the shared GG EM components."""
    coefficients = model._initial_coefficients(delta)
    steps = _LinearCureSteps(
        model,
        t,
        delta,
    )

    coefficients, ll_new = run_em(
        coefficients,
        e_step=steps.expectation,
        m_steps=(
            steps.maximize_predictor,
            steps.maximize_latency,
            steps.maximize_latent_parameters,
        ),
        loglikelihood=steps.loglikelihood,
        state=_EstimatorEMState(model),
        tol=model.tol,
        max_iter=model.max_em_iter,
        rtol=model.em_rtol,
        validate_initial_loglik=model._validate_initial_loglik,
        nonfinite_message=f"Non-finite {model._model_name} log-likelihood.",
    )

    model._set_fitted_coefficients(coefficients)
    model.final_loglik_ = ll_new
    try:
        model._latent_model.finalize_fit()
    except Exception:
        model.termination_reason_ = "failed"
        raise
