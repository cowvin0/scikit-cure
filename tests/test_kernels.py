"""Kernel interoperability and legacy GGKM kernel regression tests."""

import narwhals as nw
import narwhals.stable.v2 as nw2
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.util import Surv

from skcure.ggkm import GGKM
from skcure.utils.kernels import cauchy_kernel, exponential_kernel, wavelet_kernel

SKLEARN_KERNELS = [
    "linear",
    "rbf",
    "poly",
    "polynomial",
    "laplacian",
    "sigmoid",
    "cosine",
    "chi2",
    "additive_chi2",
]


@pytest.fixture
def feature_matrices():
    # Nonnegative features let the same inputs exercise chi-squared kernels.
    X = np.array([[0.0, 1.0, 2.0], [1.5, 0.0, 0.5], [2.0, 3.0, 1.0]])
    Y = np.array([[0.5, 2.0, 1.5], [3.0, 1.0, 0.0]])
    return X, Y


@pytest.mark.parametrize("kernel", [exponential_kernel, cauchy_kernel, wavelet_kernel])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize(
    "wrap", [lambda frame: frame, nw.from_native, nw2.from_native],
    ids=["native", "narwhals", "stable-v2"],
)
def test_standalone_kernels_accept_dataframe_inputs(
    kernel, backend, wrap, feature_matrices, monkeypatch
):
    X, Y = feature_matrices
    columns = ["age", "marker", "treatment"]
    if backend == "pandas":
        frame_X = pd.DataFrame(X, columns=columns)
        frame_Y = pd.DataFrame(Y, columns=columns)
    else:
        frame_X = pl.DataFrame(X, schema=columns, orient="row")
        frame_Y = pl.DataFrame(Y, schema=columns, orient="row")

    def unexpected_to_pandas(*args, **kwargs):
        raise AssertionError("Kernel inputs must not pass through pandas")

    monkeypatch.setattr(pl.DataFrame, "to_pandas", unexpected_to_pandas)
    for other in (None, wrap(frame_Y)):
        expected = kernel(X, None if other is None else Y)
        actual = kernel(wrap(frame_X), other)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("kernel", [exponential_kernel, cauchy_kernel, wavelet_kernel])
@pytest.mark.parametrize("lazy_argument", ["X", "Y"])
@pytest.mark.parametrize(
    "wrap", [lambda frame: frame, nw.from_native, nw2.from_native],
    ids=["native", "narwhals", "stable-v2"],
)
def test_standalone_kernels_reject_lazy_inputs_without_collection(
    kernel, lazy_argument, wrap, feature_matrices, monkeypatch
):
    X, Y = feature_matrices
    lazy = wrap(pl.DataFrame(X, orient="row").lazy())

    def unexpected_collect(*args, **kwargs):
        raise AssertionError("Kernels must not implicitly collect lazy inputs")

    monkeypatch.setattr(pl.LazyFrame, "collect", unexpected_collect)
    with pytest.raises(TypeError, match="collect"):
        if lazy_argument == "X":
            kernel(lazy, Y)
        else:
            kernel(X, lazy)


def test_ggkm_does_not_inherit_kernel_implementation():
    assert all(base.__name__ != "KernelFunc" for base in GGKM.__mro__)


@pytest.mark.parametrize("kernel", SKLEARN_KERNELS)
@pytest.mark.parametrize("training", [False, True], ids=["rectangular", "training"])
@pytest.mark.parametrize(
    "parameters",
    [
        {"gamma": 1.0, "degree": 3, "coef0": 0.0},
        {"gamma": 0.4, "degree": 2, "coef0": 0.7},
    ],
    ids=["defaults", "custom-parameters"],
)
def test_named_kernels_match_sklearn(kernel, training, parameters, feature_matrices):
    X, Y = feature_matrices
    if training:
        Y = X
    model = GGKM(kernel=kernel, **parameters)
    expected = pairwise_kernels(X, Y, metric=kernel, filter_params=True, **parameters)

    observed = model._compute_kernel(X, Y)

    assert observed.shape == (len(X), len(Y))
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-14)


def test_gaussian_alias_matches_rbf(feature_matrices):
    X, Y = feature_matrices
    expected = pairwise_kernels(X, Y, metric="rbf", gamma=0.4)
    observed = GGKM(kernel="gaussian", gamma=0.4)._compute_kernel(X, Y)
    np.testing.assert_allclose(observed, expected)


@pytest.mark.parametrize("kernel", ["linear", "cosine", "additive_chi2"])
def test_parameters_not_used_by_named_kernel_are_ignored(kernel, feature_matrices):
    X, Y = feature_matrices
    observed = GGKM(kernel=kernel, gamma=5.0, degree=7, coef0=8.0)._compute_kernel(X, Y)
    expected = pairwise_kernels(X, Y, metric=kernel)
    np.testing.assert_allclose(observed, expected)


@pytest.mark.parametrize("kernel", ["exponential", "cauchy", "wavelet"])
@pytest.mark.parametrize("training", [False, True], ids=["rectangular", "training"])
def test_custom_named_kernels_preserve_formulas(kernel, training, feature_matrices):
    X, Y = feature_matrices
    if training:
        Y = X
    gamma = 0.4
    scale = 2.5
    squared_distance = cdist(X, Y, metric="sqeuclidean")
    if kernel == "exponential":
        expected = np.exp(-gamma * cdist(X, Y, metric="euclidean"))
    elif kernel == "cauchy":
        expected = 1.0 / (1.0 + gamma * squared_distance)
    else:
        differences = X[:, None, :] - Y[None, :, :]
        expected = np.exp(-0.5 * squared_distance / scale**2)
        expected *= np.prod(np.cos(1.75 * differences / scale), axis=2)

    observed = GGKM(kernel=kernel, gamma=gamma, A=scale)._compute_kernel(X, Y)

    assert observed.shape == (len(X), len(Y))
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize(
    "scale,expected",
    [
        (1.0e-308, np.eye(3)),
        (1.0e308, np.ones((3, 3))),
    ],
)
def test_wavelet_extreme_scale_preserves_finite_diagonal(scale, expected):
    X = np.array([[-1.0, 2.0], [0.0, 0.0], [1.0, 3.0]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = wavelet_kernel(X, A=scale)
    np.testing.assert_array_equal(actual, expected)


def test_wavelet_scales_overflowing_raw_differences_before_evaluation():
    X = np.array([[-1.0e308], [1.0e308]])
    expected_off_diagonal = np.exp(-2.0) * np.cos(3.5)
    expected = np.array([[1.0, expected_off_diagonal], [expected_off_diagonal, 1.0]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = wavelet_kernel(X, A=1.0e308)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)


def test_wavelet_tiny_scale_resolves_proportionally_tiny_differences():
    X = np.array([[0.0, 0.0], [1.0e-308, 2.0e-308]])
    expected_off_diagonal = np.exp(-2.5) * np.cos(1.75) * np.cos(3.5)
    expected = np.array([[1.0, expected_off_diagonal], [expected_off_diagonal, 1.0]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = wavelet_kernel(X, A=1.0e-308)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("kernel", ["chi2", "additive_chi2"])
def test_chi_squared_kernels_reject_negative_features(kernel, feature_matrices):
    X, Y = feature_matrices
    X[0, 0] = -1.0
    with pytest.raises(ValueError):
        GGKM(kernel=kernel)._compute_kernel(X, Y)


def test_unknown_kernel_is_rejected(feature_matrices):
    with pytest.raises(ValueError):
        GGKM(kernel="not-a-kernel")._compute_kernel(*feature_matrices)


@pytest.mark.parametrize("kernel", ["rbf", "cauchy", "wavelet"])
@pytest.mark.parametrize(
    "X,Y",
    [
        (np.ones(3), np.ones((2, 3))),
        (np.ones((2, 3)), np.ones(3)),
        (np.ones((2, 3)), np.ones((2, 2))),
        (np.array([[np.nan, 1.0]]), np.ones((2, 2))),
        (np.ones((2, 2)), np.array([[1.0, np.inf]])),
    ],
    ids=["one-dimensional-X", "one-dimensional-Y", "feature-count", "nan", "inf"],
)
def test_kernel_inputs_require_compatible_finite_matrices(kernel, X, Y):
    with pytest.raises(ValueError):
        GGKM(kernel=kernel)._compute_kernel(X, Y)


@pytest.fixture(scope="module")
def survival_data():
    rng = np.random.default_rng(47)
    X = rng.normal(size=(12, 2))
    y = Surv.from_arrays(
        np.tile([True, False, True], 4),
        rng.uniform(0.5, 3.0, len(X)),
    )
    return X, y


def test_matrix_callable_survives_clone_fit_and_prediction(survival_data):
    X, y = survival_data
    calls = []

    def matrix_rbf(X1, X2, scale):
        assert X1.ndim == X2.ndim == 2
        calls.append((X1.shape, X2.shape, scale))
        return pairwise_kernels(X1, X2, metric="rbf", gamma=scale)

    parameters = {"scale": 0.4}
    original = GGKM(
        kernel=matrix_rbf, kernel_params=parameters, lambda_reg=0.1, max_em_iter=1
    )
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        model = clone(original).fit(X, y)
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        expected_model = GGKM(
            kernel="rbf", gamma=0.4, lambda_reg=0.1, max_em_iter=1
        ).fit(X, y)

    np.testing.assert_allclose(model.K_, expected_model.K_)
    np.testing.assert_allclose(model.predict(X[:3]), expected_model.predict(X[:3]))
    assert (X.shape, X.shape, 0.4) in calls
    assert (X[:3].shape, X.shape, 0.4) in calls
    assert original.kernel_params is parameters
    assert model.kernel_params == parameters
    assert model.kernel_params is not parameters


@pytest.fixture(scope="module")
def fitted_precomputed_models(survival_data):
    X, y = survival_data
    gram = pairwise_kernels(X, X, metric="rbf", gamma=0.4)
    common = {"lambda_reg": 0.1, "max_em_iter": 1}
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        model = GGKM(kernel="precomputed", **common).fit(gram, y)
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        reference = GGKM(kernel="rbf", gamma=0.4, **common).fit(X, y)
    return model, reference


def test_precomputed_kernels_match_feature_based_fit_and_predictions(
    survival_data,
    fitted_precomputed_models,
):
    X, _ = survival_data
    model, reference = fitted_precomputed_models
    new_X = np.array([[0.2, -0.4], [1.1, 0.3], [-0.3, 0.8]])
    new_gram = pairwise_kernels(new_X, X, metric="rbf", gamma=0.4)

    np.testing.assert_allclose(model.K_, reference.K_)
    np.testing.assert_allclose(model.predict(new_gram), reference.predict(new_X))
    np.testing.assert_allclose(
        model.predict_survival_function(new_gram, return_array=True),
        reference.predict_survival_function(new_X, return_array=True),
    )
    assert model.__sklearn_tags__().input_tags.pairwise
    assert not reference.__sklearn_tags__().input_tags.pairwise


def test_precomputed_training_kernel_must_be_square(survival_data):
    X, y = survival_data
    gram = pairwise_kernels(X, X, metric="rbf", gamma=0.4)
    with pytest.raises(ValueError):
        GGKM(kernel="precomputed", max_em_iter=1).fit(gram[:, :-1], y)


@pytest.mark.parametrize("n_columns", [11, 13])
def test_precomputed_prediction_requires_one_column_per_training_sample(
    fitted_precomputed_models,
    n_columns,
):
    model, _ = fitted_precomputed_models
    with pytest.raises(ValueError):
        model.predict(np.ones((3, n_columns)))


def test_precomputed_kernels_support_grid_search_pairwise_slicing(survival_data):
    X, y = survival_data
    gram = pairwise_kernels(X, X, metric="rbf", gamma=0.4)
    search = GridSearchCV(
        GGKM(kernel="precomputed", max_em_iter=1),
        {"lambda_reg": [0.1, 0.2]},
        cv=KFold(n_splits=2),
        error_score="raise",
    )
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        search.fit(gram, y)

    assert np.all(np.isfinite(search.cv_results_["mean_test_score"]))
    assert search.predict(gram[:3]).shape == (3,)
