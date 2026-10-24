import numpy as np
from numpy.testing import assert_allclose

from skcure.utils.numerics import (
    _exp_log_quantity,
    _maximum_log_float,
    _safe_exp,
)


def test_exp_log_quantity_preserves_nan_and_extreme_values():
    actual = _exp_log_quantity(np.array([np.nan, -np.inf, np.inf, 1000.0, -1000.0]))
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

    actual = _safe_exp(np.array([log_min]))

    assert actual[0] == smallest_subnormal


def test_safe_exp_retains_values_that_round_up_to_smallest_subnormal():
    log_min = np.log(np.nextafter(0.0, 1.0))
    log_x = np.array([log_min - 0.25, log_min - 0.5])
    np.testing.assert_array_equal(_safe_exp(log_x), np.exp(log_x))
    assert np.all(_safe_exp(log_x) > 0.0)


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
