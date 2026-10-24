"""Small real-backend checks, enabled when the deep-learning extra is present."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sksurv.util import Surv

tf = pytest.importorskip("tensorflow")
pytest.importorskip("deepcure")

from skcure.deep_learning import DeepPromotionTimeCure
from skcure.deep_learning._loss import _StableDeepPTMEndpoint


def _data():
    X = np.column_stack((np.linspace(-0.8, 0.8, 16), np.sin(np.arange(16))))
    event = np.tile([True, True, False, True], 4)
    time = np.linspace(0.15, 2.0, 16)
    return X, Surv.from_arrays(event, time)


@pytest.mark.parametrize("t_func", ["pe", "weibull"])
@pytest.mark.parametrize("parametrization", [1, 2])
def test_actual_backend_trains_network_and_baseline(
    t_func, parametrization, monkeypatch,
):
    X, y = _data()
    snapshots = []
    original_fit_model = DeepPromotionTimeCure._fit_model

    def capture_updates(self, model, X_fit, time, event, **kwargs):
        inputs = {
            "x": tf.convert_to_tensor(X_fit, dtype=tf.float64),
            "time_event": tf.convert_to_tensor(time, dtype=tf.float64),
            "label_event": tf.convert_to_tensor(event, dtype=tf.float64),
        }
        model(inputs, training=False)
        before_network = model.stack_eta[0].get_weights()[0].copy()
        before_baseline = model.endpoint.get_weights()[0].copy()
        result = original_fit_model(self, model, X_fit, time, event, **kwargs)
        snapshots.append((
            before_network, model.stack_eta[0].get_weights()[0].copy(),
            before_baseline, model.endpoint.get_weights()[0].copy(),
        ))
        return result

    monkeypatch.setattr(DeepPromotionTimeCure, "_fit_model", capture_updates)
    model = DeepPromotionTimeCure(
        t_func=t_func, parametrization=parametrization, hidden_layers=(4,),
        dropout=0, n_intervals=2, optimizer="adam", learning_rate=1e-4,
        use_lr_decay=False, epochs=2, batch_size=16, validation_fraction=0,
        random_state=73,
    ).fit(X, y)
    before_network, after_network, before_baseline, after_baseline = snapshots[0]
    assert isinstance(model.model_.endpoint, _StableDeepPTMEndpoint)
    assert model.model_.compute_dtype == "float64"
    assert model.model_.stack_eta[0].compute_dtype == "float64"
    assert np.any(before_network != after_network)
    assert np.any(before_baseline != after_baseline)
    assert np.all(np.isfinite(model.history_["loss"]))
    assert np.all(model.baseline_parameters_ > 0)
    assert model.predict(X).shape == (len(X),)
    assert np.all(np.isfinite(model.predict(X)))

    times = np.array([0.0, 0.2, 0.7, 1.8, 1e6])
    survival = model.predict_survival(X[:3], times)
    assert np.all(np.isfinite(survival))
    assert np.all(np.diff(survival, axis=1) <= 1e-13)
    np.testing.assert_array_equal(survival[:, 0], np.ones(3))
    assert np.all(survival[:, -1] >= model.predict_cure_probability(X[:3]))

    # Upstream survival with type='S1' evaluates its fitted baseline directly.
    # Comparing inside its support catches endpoint ordering/scale mistakes
    # without invoking upstream's separate inference-model reconstruction.
    theta = model.predict_theta(X[:3])
    expected = []
    for time in times[1:-1]:
        native = model.model_.surv_predict({
            "x": tf.convert_to_tensor(X[:3], dtype=tf.float64),
            "time_event": tf.fill([3], tf.cast(time, tf.float64)),
            "label_event": tf.zeros(3, dtype=tf.float64),
        }, type="S1")
        expected.append(np.exp(-theta * (1 - np.asarray(native).ravel())))
    np.testing.assert_allclose(survival[:, 1:-1], np.asarray(expected).T, rtol=2e-6, atol=1e-8)


def _endpoint(t_func, parametrization, parameters):
    endpoint = _StableDeepPTMEndpoint(
        t_func, parametrization,
        np.array([0.0, 2.0]) if t_func == "pe" else None,
    )
    endpoint(tf.zeros((1, 1), dtype=tf.float64))
    endpoint.set_weights([np.asarray(parameters, dtype=float)])
    return endpoint


@pytest.mark.parametrize("t_func", ["pe", "weibull"])
@pytest.mark.parametrize("parametrization", [1, 2])
def test_training_loss_keeps_small_baseline_cdf_and_its_predictor_gradient(
    t_func, parametrization,
):
    parameters = [1e-8] if t_func == "pe" else [1.0, 1e8]
    endpoint = _endpoint(t_func, parametrization, parameters)
    eta = tf.Variable([[20.0]], dtype=tf.float64)
    with tf.GradientTape() as tape:
        endpoint(eta, event=tf.constant([0.0], tf.float64), timeset=tf.constant([1.0], tf.float64))
        loss = tf.reduce_sum(endpoint.losses)
    gradient = tape.gradient(loss, eta).numpy().item()
    theta = np.exp(20.0) if parametrization == 1 else np.logaddexp(0.0, 20.0)
    expected_loss = theta * (-np.expm1(-1e-8))
    theta_derivative = np.exp(20.0) if parametrization == 1 else 1 / (1 + np.exp(-20.0))
    np.testing.assert_allclose(loss.numpy(), expected_loss, rtol=2e-14)
    np.testing.assert_allclose(gradient, theta_derivative * (-np.expm1(-1e-8)), rtol=2e-14)


@pytest.mark.parametrize("t_func", ["pe", "weibull"])
def test_training_loss_combines_extreme_predictor_and_underflowing_baseline(t_func):
    parameters = [np.exp(-500.0)] if t_func == "pe" else [2.0, 1.0]
    endpoint = _endpoint(t_func, 1, parameters)
    eta = tf.Variable([[1000.0]], dtype=tf.float64)
    with tf.GradientTape() as tape:
        endpoint(
            eta, event=tf.constant([0.0], tf.float64),
            timeset=tf.constant([np.exp(-500.0)], tf.float64),
        )
        loss = tf.reduce_sum(endpoint.losses)
    np.testing.assert_allclose(loss.numpy(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(tape.gradient(loss, eta).numpy(), [[1.0]], rtol=1e-12)


@pytest.mark.parametrize("t_func", ["pe", "weibull"])
def test_softplus_event_loss_remains_finite_for_very_negative_predictor(t_func):
    endpoint = _endpoint(t_func, 2, [1.0] if t_func == "pe" else [1.0, 1.0])
    eta = tf.Variable([[-1000.0]], dtype=tf.float64)
    with tf.GradientTape() as tape:
        endpoint(eta, event=tf.constant([1.0], tf.float64), timeset=tf.constant([1.0], tf.float64))
        loss = tf.reduce_sum(endpoint.losses)
    np.testing.assert_allclose(loss.numpy(), 1001.0)
    np.testing.assert_allclose(tape.gradient(loss, eta).numpy(), [[-1.0]])


@pytest.mark.parametrize("t_func", ["pe", "weibull"])
def test_censored_extreme_baseline_hazard_has_finite_loss_and_gradients(t_func):
    parameters = [np.exp(500.0)] if t_func == "pe" else [2.0, 1.0]
    endpoint = _endpoint(t_func, 1, parameters)
    eta = tf.Variable([[0.0]], dtype=tf.float64)
    with tf.GradientTape() as tape:
        endpoint(
            eta, event=tf.constant([0.0], tf.float64),
            timeset=tf.constant([np.exp(500.0)], tf.float64),
        )
        loss = tf.reduce_sum(endpoint.losses)
    gradients = tape.gradient(loss, [eta, *endpoint.trainable_weights])
    np.testing.assert_allclose(loss.numpy(), 1.0)
    np.testing.assert_allclose(gradients[0].numpy(), [[1.0]])
    assert all(np.all(np.isfinite(gradient.numpy())) for gradient in gradients)


@pytest.mark.parametrize("t_func", ["pe", "weibull"])
@pytest.mark.parametrize("parametrization", [1, 2])
def test_endpoint_loss_matches_promotion_time_likelihood(t_func, parametrization):
    endpoint = _endpoint(t_func, parametrization, [0.7] if t_func == "pe" else [1.3, 0.8])
    eta = np.array([-2.0, 0.0, 3.0])
    time = np.array([0.2, 0.8, 1.7])
    event = np.array([1.0, 0.0, 1.0])
    theta = np.exp(eta) if parametrization == 1 else np.logaddexp(0.0, eta)
    if t_func == "pe":
        hazard = 0.7 * time
        log_rate = np.full(3, np.log(0.7))
    else:
        hazard = (time / 0.8) ** 1.3
        log_rate = np.log(1.3) - np.log(0.8) + 0.3 * np.log(time / 0.8)
    expected = theta * (-np.expm1(-hazard)) - event * (np.log(theta) + log_rate - hazard)
    endpoint(
        tf.constant(eta[:, None], tf.float64),
        event=tf.constant(event, tf.float64), timeset=tf.constant(time, tf.float64),
    )
    np.testing.assert_allclose(endpoint.losses[0].numpy(), expected, rtol=2e-14)


def test_baseline_projection_cannot_reach_zero():
    endpoint = _endpoint("pe", 1, [1.0])
    projected = endpoint.trainable_weights[0].constraint(tf.constant([-1.0], tf.float64))
    assert projected.numpy()[0] > 0.0


def test_actual_backend_early_stopping_refits_the_selected_checkpoint():
    X, y = _data()
    model = DeepPromotionTimeCure(
        hidden_layers=(3,), dropout=0, n_intervals=2,
        optimizer="adam", learning_rate=1e-4, use_lr_decay=False,
        epochs=4, batch_size=16, validation_fraction=0.25,
        patience=1, min_delta=1e6, refit_full_data=True, random_state=73,
    ).fit(X, y)
    assert model.best_epoch_ == 1
    assert len(model.history_["val_loss"]) >= 2
    assert len(model.refit_history_["loss"]) == model.n_iter_ == 1
    assert np.all(np.isfinite(model.predict(X)))


@pytest.mark.parametrize("t_func", ["pe", "weibull"])
@pytest.mark.parametrize("parametrization", [1, 2])
def test_fitted_pickle_preserves_network_after_constructor_parameters_change(
    t_func, parametrization,
):
    X, y = _data()
    model = DeepPromotionTimeCure(
        t_func=t_func, parametrization=parametrization, hidden_layers=(4, 2),
        dropout=0.25, n_intervals=2, optimizer="adam", learning_rate=1e-4,
        epochs=1, validation_fraction=0, random_state=73,
    ).fit(X, y)
    times = np.array([0.0, 0.3, 1.5, 20.0])
    expected_risk = model.predict(X)
    expected_cure = model.predict_cure_probability(X)
    expected_survival = model.predict_survival(X, times)
    model.set_params(
        hidden_layers=(7,), t_func="weibull" if t_func == "pe" else "pe",
        parametrization=3 - parametrization, optimizer="rmsprop",
    )
    original_network = model.model_
    payload = pickle.dumps(model)
    assert model.model_ is original_network
    np.testing.assert_array_equal(model.predict(X), expected_risk)
    restored = pickle.loads(payload)
    check_is_fitted(restored)
    assert restored.get_params() == model.get_params()
    assert restored.t_func_ == t_func
    assert restored.parametrization_ == parametrization
    assert restored.model_ is not model.model_
    assert isinstance(restored.model_.endpoint, _StableDeepPTMEndpoint)
    assert restored.model_.optimizer.__class__.__name__ == "Adam"
    assert len(restored.model_.stack_eta) == len(model.model_.stack_eta)
    np.testing.assert_array_equal(restored.unique_times_, model.unique_times_)
    np.testing.assert_array_equal(restored.baseline_parameters_, model.baseline_parameters_)
    np.testing.assert_allclose(restored.predict(X), expected_risk, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(restored.predict_cure_probability(X), expected_cure)
    np.testing.assert_allclose(restored.predict_survival(X, times), expected_survival)
    # Dropout must remain disabled when predicting from the restored network.
    np.testing.assert_array_equal(restored.predict(X), restored.predict(X))
    copied = clone(restored)
    assert copied.get_params() == restored.get_params()
    with pytest.raises(NotFittedError):
        copied.predict(X)
