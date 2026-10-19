from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammainc, gammaincc, gammaln

from skcure.distributions import generalized_gamma
from skcure.utils.pgamma_derivate import pgamma_shape_derivative_vec
from skcure.utils.numerics import _maximum_log_float, _safe_exp


class _LatencyOptimizer(Protocol):
    """Protocol for latency-parameter M-step optimizers."""

    def maximize(
        self,
        latency,
        latent_model,
        objective,
        t,
        delta,
        expected,
    ) -> None:
        """Update latency parameters in place."""
        ...


class _GeneralizedGammaLatencyOptimizer:
    """Generalized-gamma latency optimizer used by GGKM.

    The optimizer owns the shared generalized-gamma objective and
    log-parameter transformation. Small latent-count-specific numerical
    differences are delegated to the supplied latent-count model so that
    the current Poisson, negative-binomial, and binomial behavior is
    preserved exactly during the refactor.
    """

    @staticmethod
    def _invalid_value_and_grad(latent_model):
        """Return the legacy GG invalid-objective sentinel for each family."""
        if latent_model.name == "poisson":
            return (
                np.inf,
                np.full(
                    3,
                    np.nan,
                ),
            )

        return (
            1.0e100,
            np.zeros(
                3,
                dtype=float,
            ),
        )

    @staticmethod
    def _finish_objective(latent_model, Q, grad):
        """Format the natural-scale GG objective using legacy family semantics."""
        if latent_model.name == "poisson":
            return (
                -float(Q),
                -grad,
            )

        return (
            -Q,
            -grad,
        )

    @staticmethod
    def _log_objective_is_invalid(latent_model, value):
        """Apply the legacy family rule for log-scale objective rejection."""
        if latent_model.name == "poisson":
            return False

        return value >= 1.0e99

    @staticmethod
    def _validate_result(latent_model, result):
        """Validate the GG optimizer result without changing legacy semantics."""
        if latent_model.name == "negative_binomial":
            invalid = not result.success or not np.all(np.isfinite(result.x))
        else:
            invalid = (
                not result.success
                or not np.isfinite(result.fun)
                or not np.all(np.isfinite(result.x))
            )

        if invalid:
            raise RuntimeError(
                "Generalized-gamma log-parameter "
                "optimization failed: "
                f"{result.message}"
            )

    @staticmethod
    def _validate_log_parameters(latent_model, log_pars):
        """Validate optimized GG log-parameters using the legacy family rules."""
        if latent_model.name == "poisson":
            return

        log_max = _maximum_log_float()

        if np.any(log_pars > log_max):
            raise FloatingPointError(
                "Generalized-gamma log-parameter "
                "optimization returned values outside "
                "the representable floating-point range."
            )

    @staticmethod
    def _prepare_parameters(latent_model, pars):
        """Prepare natural-scale GG parameters using legacy family semantics."""
        if latent_model.name != "poisson":
            return pars

        pars = np.asarray(
            pars,
            dtype=float,
        )

        if pars.shape != (3,):
            return None

        return pars

    @staticmethod
    def _prepare_argument(latent_model, t, a, p):
        """Prepare the GG transformed time argument without changing numerics."""
        if latent_model.name != "poisson":
            (
                log_t,
                log_ta,
                log_u,
                u,
            ) = generalized_gamma._argument_from_parameters(
                t,
                a,
                p,
            )

            if np.any(np.isnan(u)):
                return None

            return (
                log_t,
                log_ta,
                log_u,
                u,
            )

        log_t = np.log(t)
        log_ta = log_t - np.log(a)

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            log_u = p * log_ta
        max_log = _maximum_log_float()

        if np.any(np.isnan(log_u)) or np.any(log_u > max_log):
            return None

        with np.errstate(
            under="ignore",
            over="ignore",
            invalid="ignore",
        ):
            u = np.exp(log_u)

        if np.any(~np.isfinite(u)):
            return None

        return (
            log_t,
            log_ta,
            log_u,
            u,
        )

    @staticmethod
    def _validate_distribution_values(latent_model, s, u):
        """Validate GG distribution values using the legacy family rules."""
        if latent_model.name == "negative_binomial":
            FGG = gammainc(
                s,
                u,
            )

            SGG = gammaincc(
                s,
                u,
            )

            if np.any(~np.isfinite(FGG)) or np.any(~np.isfinite(SGG)):
                return None

            return SGG

        SGG = gammaincc(
            s,
            u,
        )

        if np.any(~np.isfinite(SGG)):
            return None

        return SGG

    @staticmethod
    def _event_grad_p(
        latent_model,
        inv_p,
        u_event,
        log_ta_event,
        d,
        s,
        psi_s,
    ):
        """Return the GG event gradient in p using legacy family arithmetic."""
        if latent_model.name == "poisson":
            return inv_p - u_event * log_ta_event + d * psi_s * inv_p * inv_p

        if latent_model.name == "binomial":
            d_over_p2 = s * inv_p
            return inv_p - u_event * log_ta_event + d_over_p2 * psi_s

        return inv_p - u_event * log_ta_event + s * psi_s * inv_p

    @staticmethod
    def _survival_grad_p(
        latent_model,
        weights,
        d,
        s,
        inv_p,
        G_over_S,
        H_over_S,
        log_ta_surv,
    ):
        """Return the GG survival gradient in p using legacy family arithmetic."""
        if latent_model.name == "binomial":
            d_over_p2 = s * inv_p
            return weights * (d_over_p2 * G_over_S - H_over_S * log_ta_surv)

        return weights * (d * G_over_S * inv_p * inv_p - H_over_S * log_ta_surv)

    @staticmethod
    def _exp_ratio(latent_model, log_value):
        """Exponentiate GG ratio terms using the legacy family-specific guards."""
        if latent_model.name != "poisson":
            return _safe_exp(log_value)

        log_value = np.asarray(
            log_value,
            dtype=float,
        )

        result = np.zeros_like(
            log_value,
            dtype=float,
        )

        max_log = _maximum_log_float()
        positive_infinity = np.isposinf(log_value)
        too_large = log_value > max_log
        regular = np.isfinite(log_value) & ~too_large

        result[positive_infinity | too_large] = np.inf
        result[regular] = np.exp(log_value[regular])

        return result

    def value_and_grad(
        self,
        pars,
        latent_model,
        t,
        delta,
        expected,
    ):
        """Return GG M-step objective and gradient on the natural scale."""
        pars = self._prepare_parameters(
            latent_model,
            pars,
        )

        if pars is None:
            return self._invalid_value_and_grad(latent_model)

        a = float(pars[0])
        d = float(pars[1])
        p = float(pars[2])

        if (
            not np.isfinite(a)
            or not np.isfinite(d)
            or not np.isfinite(p)
            or a <= 0.0
            or d <= 0.0
            or p <= 0.0
        ):
            return self._invalid_value_and_grad(latent_model)

        s = d / p

        if not np.isfinite(s) or s <= 0.0:
            return self._invalid_value_and_grad(latent_model)

        log_gamma_s = gammaln(s)
        psi_s = digamma(s)

        if not np.isfinite(log_gamma_s) or not np.isfinite(psi_s):
            return self._invalid_value_and_grad(latent_model)

        gg_argument = self._prepare_argument(
            latent_model,
            t,
            a,
            p,
        )

        if gg_argument is None:
            return self._invalid_value_and_grad(latent_model)

        (
            log_t,
            log_ta,
            log_u,
            u,
        ) = gg_argument

        SGG = self._validate_distribution_values(
            latent_model,
            s,
            u,
        )

        if SGG is None:
            return self._invalid_value_and_grad(latent_model)

        log_SGG = latent_model.log_nonnegative(SGG)
        log_fGG = generalized_gamma._gg_log_density(
            log_t,
            u,
            a,
            d,
            p,
            log_gamma_s,
        )

        event_mask = delta > 0.5

        latent_weight = expected - delta
        latent_weight = np.maximum(
            latent_weight,
            0.0,
        )

        survival_mask = latent_weight > 0.0
        Q = 0.0

        if np.any(event_mask):
            event_log_f = log_fGG[event_mask]

            if np.any(~np.isfinite(event_log_f)):
                return self._invalid_value_and_grad(latent_model)

            Q += np.sum(event_log_f)

        if np.any(survival_mask):
            log_S_surv = log_SGG[survival_mask]

            if np.any(~np.isfinite(log_S_surv)):
                return self._invalid_value_and_grad(latent_model)

            Q += np.sum(latent_weight[survival_mask] * log_S_surv)

        if not np.isfinite(Q):
            return self._invalid_value_and_grad(latent_model)

        inv_p = 1.0 / p

        grad_a = 0.0
        grad_d = 0.0
        grad_p = 0.0

        if np.any(event_mask):
            u_event = u[event_mask]
            log_ta_event = log_ta[event_mask]

            with np.errstate(
                over="ignore",
                invalid="ignore",
                divide="ignore",
            ):
                event_grad_a = (p * u_event - d) / a
                event_grad_d = log_ta_event - psi_s * inv_p

                event_grad_p = self._event_grad_p(
                    latent_model,
                    inv_p,
                    u_event,
                    log_ta_event,
                    d,
                    s,
                    psi_s,
                )

            if (
                np.any(~np.isfinite(event_grad_a))
                or np.any(~np.isfinite(event_grad_d))
                or np.any(~np.isfinite(event_grad_p))
            ):
                return self._invalid_value_and_grad(latent_model)

            grad_a += np.sum(event_grad_a)
            grad_d += np.sum(event_grad_d)
            grad_p += np.sum(event_grad_p)

        if np.any(survival_mask):
            u_surv = u[survival_mask]
            log_u_surv = log_u[survival_mask]
            log_ta_surv = log_ta[survival_mask]
            log_S_surv = log_SGG[survival_mask]
            weights = latent_weight[survival_mask]

            with np.errstate(
                over="ignore",
                invalid="ignore",
            ):
                log_H = s * log_u_surv - u_surv - log_gamma_s
                log_H_over_S = log_H - log_S_surv

            if np.any(np.isnan(log_H_over_S)):
                return self._invalid_value_and_grad(latent_model)

            H_over_S = self._exp_ratio(
                latent_model,
                log_H_over_S,
            )

            if np.any(~np.isfinite(H_over_S)):
                return self._invalid_value_and_grad(latent_model)

            G = pgamma_shape_derivative_vec(
                u_surv,
                s,
            )
            G = np.asarray(
                G,
                dtype=float,
            )

            if np.any(~np.isfinite(G)):
                return self._invalid_value_and_grad(latent_model)

            G_over_S = np.zeros_like(
                G,
                dtype=float,
            )

            nonzero_G = G != 0.0

            if np.any(nonzero_G):
                with np.errstate(
                    divide="ignore",
                    invalid="ignore",
                    over="ignore",
                ):
                    log_abs_G_over_S = (
                        np.log(np.abs(G[nonzero_G])) - log_S_surv[nonzero_G]
                    )

                if np.any(np.isnan(log_abs_G_over_S)):
                    return self._invalid_value_and_grad(latent_model)

                abs_G_over_S = self._exp_ratio(
                    latent_model,
                    log_abs_G_over_S,
                )

                if np.any(~np.isfinite(abs_G_over_S)):
                    return self._invalid_value_and_grad(latent_model)

                G_over_S[nonzero_G] = np.sign(G[nonzero_G]) * abs_G_over_S

            with np.errstate(
                over="ignore",
                invalid="ignore",
            ):
                survival_grad_a = weights * (p / a) * H_over_S

                survival_grad_d = -weights * G_over_S * inv_p

                survival_grad_p = self._survival_grad_p(
                    latent_model,
                    weights,
                    d,
                    s,
                    inv_p,
                    G_over_S,
                    H_over_S,
                    log_ta_surv,
                )

            if (
                np.any(~np.isfinite(survival_grad_a))
                or np.any(~np.isfinite(survival_grad_d))
                or np.any(~np.isfinite(survival_grad_p))
            ):
                return self._invalid_value_and_grad(latent_model)

            grad_a += np.sum(survival_grad_a)
            grad_d += np.sum(survival_grad_d)
            grad_p += np.sum(survival_grad_p)

        grad = np.array(
            [
                grad_a,
                grad_d,
                grad_p,
            ],
            dtype=float,
        )

        if np.any(~np.isfinite(grad)):
            return self._invalid_value_and_grad(latent_model)

        return self._finish_objective(
            latent_model,
            Q,
            grad,
        )

    def log_value_and_grad(
        self,
        log_pars,
        latent_model,
        t,
        delta,
        expected,
    ):
        """Return GG objective and gradient on the log-parameter scale."""
        log_pars = np.asarray(
            log_pars,
            dtype=float,
        )

        if log_pars.shape != (3,) or np.any(~np.isfinite(log_pars)):
            return self._invalid_value_and_grad(latent_model)

        max_log = _maximum_log_float()
        if np.any(log_pars > max_log):
            return self._invalid_value_and_grad(latent_model)

        pars = np.exp(log_pars)
        if np.any(~np.isfinite(pars)) or np.any(pars <= 0.0):
            return self._invalid_value_and_grad(latent_model)

        value, grad_pars = self.value_and_grad(
            pars,
            latent_model,
            t,
            delta,
            expected,
        )

        grad_pars = np.asarray(
            grad_pars,
            dtype=float,
        )

        if not np.isfinite(value) or np.any(~np.isfinite(grad_pars)):
            return self._invalid_value_and_grad(latent_model)

        if self._log_objective_is_invalid(latent_model, value):
            return self._invalid_value_and_grad(latent_model)

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            grad_log_pars = grad_pars * pars

        if np.any(~np.isfinite(grad_log_pars)):
            return self._invalid_value_and_grad(latent_model)

        return (
            value,
            grad_log_pars,
        )

    def maximize(
        self,
        latency,
        latent_model,
        objective,
        t,
        delta,
        expected,
    ) -> None:
        """Maximize the GG latency M-step and update the latency distribution."""
        log_gg_lower = np.log(1.0e-5)

        log_gg_start = np.log(
            np.array(
                [
                    latency.a,
                    latency.d,
                    latency.p,
                ],
                dtype=float,
            )
        )

        result_gg = minimize(
            fun=objective,
            x0=log_gg_start,
            jac=True,
            args=(
                t,
                delta,
                expected,
            ),
            method="L-BFGS-B",
            bounds=[
                (
                    log_gg_lower,
                    None,
                ),
                (
                    log_gg_lower,
                    None,
                ),
                (
                    log_gg_lower,
                    None,
                ),
            ],
        )

        self._validate_result(
            latent_model,
            result_gg,
        )
        self._validate_log_parameters(
            latent_model,
            result_gg.x,
        )

        gg_pars = np.exp(result_gg.x)

        if np.any(~np.isfinite(gg_pars)) or np.any(gg_pars <= 0.0):
            raise FloatingPointError(
                "Generalized-gamma "
                "log-parameter transformation "
                "returned invalid positive "
                "parameters."
            )

        latency.set_parameters(
            a=float(gg_pars[0]),
            d=float(gg_pars[1]),
            p=float(gg_pars[2]),
        )


__all__ = [
    "_LatencyOptimizer",
    "_GeneralizedGammaLatencyOptimizer",
]
