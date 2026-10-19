"""Internal EM orchestration and optimizer steps for cure estimators.

The runner knows only the E-step, ordered M-steps, and observed likelihood.
Distribution equations, initialization, input validation, and publication of
fitted estimator attributes remain the caller's responsibility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

import numpy as np
from scipy.optimize import minimize

_Parameters = TypeVar("_Parameters")
_Expected = TypeVar("_Expected")


class _EMState(Protocol):
    """Mutable progress initialized by an estimator before each fit."""

    n_iter: int
    converged: bool
    loglik_history: list[float]


def run_em(
    initial_parameters: _Parameters,
    *,
    e_step: Callable[[_Parameters], _Expected],
    m_steps: Sequence[Callable[[_Parameters, _Expected], _Parameters]],
    loglikelihood: Callable[[_Parameters], float],
    state: _EMState,
    tol: float,
    max_iter: int,
    validate_initial_loglik: Callable[[float], None] | None = None,
    nonfinite_message: str = "Non-finite log-likelihood.",
) -> tuple[_Parameters, float]:
    """Run EM using model-supplied callbacks and return parameters and likelihood.

    ``state`` supplies an initialized iteration count, convergence flag, and
    history. The count advances before each E-step, and history records only
    completed iterations with finite likelihoods. This retains useful progress
    when a callback raises. All M-steps receive the same posterior expectations
    and the parameters returned by the preceding step.

    Convergence uses the strict absolute change ``abs(new - old) < tol`` and
    is checked before the iteration limit. The initial likelihood is evaluated
    once and optionally validated by the model; it is not added to history.
    Callers validate the tolerance and resolve a positive iteration limit.
    """
    parameters = initial_parameters
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

        diff_ll = abs(ll_new - ll_old)
        if diff_ll < tol:
            state.converged = True
            break
        if state.n_iter >= max_iter:
            break
        ll_old = ll_new

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
        )
        self.model._validate_predictor_result(result_predictor)
        beta0 = float(result_predictor.x[0])
        alpha = result_predictor.x[1:].copy()
        return beta0, alpha

    def maximize_dispersion(self, parameters, Nhat):
        beta0, alpha = parameters
        self.model._update_phi(beta0, alpha, self.t, self.delta)
        return parameters


def fit_ggkm_em(model, t, delta, *, max_em_iter):
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
        max_iter=max_em_iter,
        validate_initial_loglik=model._validate_initial_loglik,
        nonfinite_message="Non-finite penalized log-likelihood.",
    )
    model.beta0_ = beta0
    model.alpha_ = alpha
    model.final_loglik_ = ll_new
    model._latent_model.finalize_fit()


class _EstimatorEMState:
    """Expose an estimator's existing progress without copying fit state."""

    def __init__(self, model):
        self.model = model

    @property
    def n_iter(self):
        return self.model.n_iter_

    @n_iter.setter
    def n_iter(self, value):
        self.model.n_iter_ = value

    @property
    def converged(self):
        return self.model.converged_

    @converged.setter
    def converged(self, value):
        self.model.converged_ = value

    @property
    def loglik_history(self):
        return self.model.loglik_history_


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
        result = minimize(
            fun=self.model._predictor_value_and_grad,
            x0=coefficients,
            jac=True,
            args=(expected,),
            method="L-BFGS-B",
        )

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
        max_iter=model._max_em_iter,
        validate_initial_loglik=model._validate_initial_loglik,
        nonfinite_message=f"Non-finite {model._model_name} log-likelihood.",
    )

    model._set_fitted_coefficients(coefficients)
    model.final_loglik_ = ll_new
    model._latent_model.finalize_fit()
