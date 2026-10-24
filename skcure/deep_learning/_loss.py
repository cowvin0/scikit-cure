"""Stable DeepPTM training endpoint, imported only with the optional backend.

The upstream model architecture is retained. Its original loss subtracts two
exponentials to evaluate ``theta * (1 - exp(-H0))``; this endpoint evaluates
that quantity in the log domain instead.
"""

from typing import Any

import numpy as np
import tensorflow as _tensorflow  # pyright: ignore[reportMissingModuleSource]
from deepcure.models import DeepPTM  # pyright: ignore[reportMissingImports]

# TensorFlow's generated stubs omit Keras and several Tensor operators. Keep
# that optional backend at an explicit dynamic boundary, as in the wrapper.
tf: Any = _tensorflow


class _StrictlyPositive(tf.keras.constraints.Constraint):
    """Keep projected baseline coefficients inside their positive domain."""

    def __call__(self, weights):
        return tf.maximum(weights, tf.cast(np.finfo(np.float64).tiny, weights.dtype))


def _log_softplus(eta):
    # softplus(eta) is exp(eta) to double precision below this cutoff. Clamping
    # the unused branch also keeps its gradient finite when exp(eta) underflows.
    cutoff = tf.cast(-36.0, eta.dtype)
    regular = tf.math.log(tf.nn.softplus(tf.maximum(eta, cutoff)))
    return tf.where(eta < cutoff, eta, regular)


def _log_cdf_from_log_hazard(log_hazard):
    """Return log(1 - exp(-exp(log_hazard))) without tail underflow."""
    cutoff = tf.cast(-36.0, log_hazard.dtype)
    safe_log_hazard = tf.clip_by_value(log_hazard, cutoff, -cutoff)
    regular = tf.math.log(-tf.math.expm1(-tf.exp(safe_log_hazard)))
    return tf.where(log_hazard < cutoff, log_hazard, regular)


def _promotion_time_loss(event, eta, log_hazard, log_rate, parametrization):
    """Individual negative log-likelihoods for either incidence link."""
    log_theta = eta if parametrization == 1 else _log_softplus(eta)
    cumulative_hazard = tf.exp(log_theta + _log_cdf_from_log_hazard(log_hazard))
    observed = event > 0
    # Evaluate the density term only for events. In particular a censored
    # infinite baseline hazard must not introduce 0 * inf or a NaN gradient.
    event_hazard = tf.exp(tf.where(observed, log_hazard, tf.zeros_like(log_hazard)))
    event_density = log_theta + log_rate - event_hazard
    return cumulative_hazard - tf.where(observed, event_density, tf.zeros_like(eta))


class _StableDeepPTMEndpoint(tf.keras.layers.Layer):
    """Double-precision likelihood with positive PE or Weibull coefficients."""

    def __init__(self, t_func, parametrization, breakpoints=None):
        super().__init__(name="endpoint", dtype="float64")
        self.t_func = t_func
        self.parametrization = parametrization
        self.breakpoints = (
            None if breakpoints is None else tf.convert_to_tensor(breakpoints, tf.float64)
        )
        self.n_parameters = 2 if breakpoints is None else len(breakpoints) - 1

    def build(self, input_shape):
        self._lamms = self.add_weight(
            name="weibull_coef" if self.t_func == "weibull" else "lambdas",
            shape=(self.n_parameters,),
            initializer=tf.keras.initializers.RandomUniform(minval=0.1, maxval=1.0),
            trainable=True,
            constraint=_StrictlyPositive(),
            dtype="float64",
        )
        super().build(input_shape)

    def call(self, eta_pred, event=None, timeset=None):
        if event is not None:
            eta = tf.reshape(tf.cast(eta_pred, tf.float64), (-1,))
            times = tf.reshape(tf.cast(timeset, tf.float64), (-1,))
            events = tf.reshape(tf.cast(event, tf.float64), (-1,))
            if self.t_func == "weibull":
                shape, scale = self._lamms[0], self._lamms[1]
                log_time = tf.math.log(times)
                log_hazard = shape * (log_time - tf.math.log(scale))
                log_rate = tf.math.log(shape) + log_hazard - log_time
            else:
                knots = self.breakpoints
                if knots is None:
                    raise RuntimeError("A piecewise baseline requires breakpoints.")
                exposure = tf.clip_by_value(
                    times[:, None] - knots[:-1], 0.0, knots[1:] - knots[:-1]
                )
                # Continue the final interval beyond the training support.
                exposure = tf.concat(
                    [exposure[:, :-1], tf.maximum(times - knots[-2], 0.0)[:, None]],
                    axis=1,
                )
                log_rates = tf.math.log(self._lamms)
                log_hazard = tf.reduce_logsumexp(
                    tf.math.log(exposure) + log_rates, axis=1
                )
                interval = tf.minimum(
                    tf.searchsorted(knots[1:], times, side="left"),
                    self.n_parameters - 1,
                )
                log_rate = tf.gather(log_rates, interval)
            self.add_loss(
                _promotion_time_loss(
                    events, eta, log_hazard, log_rate, self.parametrization
                )
            )
        return eta_pred


class _StableDeepPTM(DeepPTM):
    """Upstream network with a stable endpoint installed during construction."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "deep_ptm")
        super().__init__(*args, **kwargs)
        self.dtype_policy = tf.keras.mixed_precision.Policy("float64")

    def __setattr__(self, name, value):
        if name == "endpoint" and not isinstance(value, _StableDeepPTMEndpoint):
            # Substitute before Keras tracks the layer: assigning a replacement
            # afterwards retains the old endpoint in model.layers/get_layer().
            value = _StableDeepPTMEndpoint(
                self.t_func, self.parametrization, self.break_val
            )
        super().__setattr__(name, value)
