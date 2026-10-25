from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skcure.utils.numerics import (
    _exp_minus_exp,
    _maximum_log_float,
    _safe_exp,
)


def test_safe_exp_preserves_nan_and_extreme_values_under_strict_errors():
    with np.errstate(all="raise"):
        actual = _safe_exp(np.array([np.nan, -np.inf, np.inf, 1000.0, -1000.0]))
    assert np.isnan(actual[0])
    np.testing.assert_array_equal(actual[1:], [0.0, np.inf, np.inf, 0.0])


def test_maximum_log_float_matches_numpy_float_limit():
    expected = float(np.log(np.finfo(float).max))

    assert _maximum_log_float() == expected


def test_safe_exp_matches_numpy_for_regular_values():
    log_x = np.array(
        [
            -10.0,
            -1.0,
            0.0,
            1.0,
            10.0,
        ]
    )

    actual = _safe_exp(log_x)
    expected = np.exp(log_x)

    assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_safe_exp_handles_special_values():
    log_x = np.array(
        [
            -np.inf,
            np.inf,
            np.nan,
        ]
    )

    actual = _safe_exp(log_x)

    assert actual[0] == 0.0
    assert np.isposinf(actual[1])
    assert np.isnan(actual[2])


def test_safe_exp_underflows_values_below_smallest_subnormal():
    log_min = np.log(np.nextafter(0.0, 1.0))
    log_x = np.array([log_min - 1.0])

    actual = _safe_exp(log_x)

    assert actual[0] == 0.0


def test_safe_exp_preserves_smallest_subnormal_at_boundary():
    smallest_subnormal = np.nextafter(0.0, 1.0)
    log_min = np.log(smallest_subnormal)

    with np.errstate(all="raise"):
        actual = _safe_exp(np.array([log_min]))

    assert actual[0] == smallest_subnormal


def test_safe_exp_retains_values_that_round_up_to_smallest_subnormal():
    log_min = np.log(np.nextafter(0.0, 1.0))
    log_x = np.array([log_min - 0.25, log_min - 0.5])
    expected = np.exp(log_x)
    with np.errstate(all="raise"):
        actual = _safe_exp(log_x)
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual > 0.0)


def test_safe_exp_overflows_values_above_largest_float():
    log_max = _maximum_log_float()
    log_x = np.array([log_max + 1.0])

    actual = _safe_exp(log_x)

    assert np.isposinf(actual[0])


def test_safe_exp_preserves_upper_boundary():
    log_max = _maximum_log_float()

    actual = _safe_exp(np.array([log_max]))
    expected = np.exp(np.array([log_max]))

    assert np.isfinite(actual[0])

    assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_safe_exp_preserves_shape():
    log_x = np.array(
        [
            [-1.0, 0.0],
            [1.0, 2.0],
        ]
    )

    actual = _safe_exp(log_x)

    assert actual.shape == log_x.shape


def test_exp_minus_exp_handles_tail_underflow_under_strict_errors():
    log_x = np.array([-np.inf, -1000.0, 0.0, 7.0, 1000.0, np.inf, np.nan])
    with np.errstate(all="raise"):
        actual = _exp_minus_exp(log_x)
    np.testing.assert_array_equal(actual[[0, 1, 3, 4, 5]], [1.0, 1.0, 0.0, 0.0, 0.0])
    assert actual[2] == pytest.approx(0.36787944117144233, rel=2e-15)
    assert np.isnan(actual[-1])


def test_exp_minus_exp_retains_last_subnormal_before_rounding_to_zero():
    # The probability becomes subnormal long before exp(log_x) overflows.
    # Use an independent high-precision reference on both sides of the last
    # positive float64 probability; an arbitrary tail cutoff would lose it.
    log_x = np.array([6.6, 6.6135, 6.6136])
    with localcontext() as context:
        context.prec = 80
        expected = np.array([
            float((-Decimal.from_float(float(value)).exp()).exp())
            for value in log_x
        ])
    with np.errstate(all="raise"):
        actual = _exp_minus_exp(log_x)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] > actual[1] > 0.0
    assert actual[1] == np.nextafter(0.0, 1.0)
    assert actual[2] == 0.0


def test_exp_minus_exp_probabilities_are_bounded_and_monotone():
    log_x = np.array([-1000.0, -40.0, -1.0, 0.0, 1.0, 6.0, 6.6, 7.0, 1000.0])
    with np.errstate(all="raise"):
        actual = _exp_minus_exp(log_x)
    assert np.isfinite(actual).all()
    assert np.all((actual >= 0.0) & (actual <= 1.0))
    assert np.all(np.diff(actual) <= 0.0)


@pytest.mark.parametrize("function", [_safe_exp, _exp_minus_exp])
@pytest.mark.parametrize("values", [0.0, [0, 1], [[0.0, 1.0], [2.0, 3.0]], np.empty((0, 2))])
def test_exponential_helpers_preserve_array_contract(function, values):
    with np.errstate(all="raise"):
        actual = function(values)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == np.shape(values)
    assert actual.dtype == np.float64


@pytest.mark.parametrize("function", [_safe_exp, _exp_minus_exp])
def test_exponential_helpers_restore_callers_error_policy(function):
    original = np.geterr()
    with np.errstate(all="raise"):
        function([-1000.0, 0.0, 1000.0])
        assert np.geterr() == dict.fromkeys(original, "raise")
        with pytest.raises(FloatingPointError):
            np.exp(np.array([1000.0]))
        with pytest.raises(FloatingPointError):
            np.divide(1.0, 0.0)
        with pytest.raises(FloatingPointError):
            np.sqrt(-1.0)
    assert np.geterr() == original
