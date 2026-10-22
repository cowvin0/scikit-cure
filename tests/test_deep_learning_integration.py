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
