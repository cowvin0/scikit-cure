"""Scikit-cure interface to DeepCure's Deep Promotion Time Cure Model."""

from math import isfinite
from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from skcure.base import BaseCureEstimator
from skcure.utils.numerics import _exp_minus_exp, _maximum_log_float, _safe_exp


class DeepPromotionTimeCure(BaseCureEstimator):
    """Deep Promotion Time Cure Model fitted through the ``deepcure`` package.

    This wraps the model of Medina-Olivares, Lessmann, and Klein (2023),
    arXiv:2305.11575. TensorFlow and DeepCure are optional dependencies loaded
    during fitting and neural-network prediction.

    Parameters
    ----------
    hidden_layers : sequence of int, default=(32, 16)
        Hidden dense-layer sizes. An empty sequence gives a linear predictor.
    activation : str, default="relu"
        Keras activation for hidden layers. The output layer is linear.
    dropout : float, default=0.2
        Dropout probability after each hidden layer, in [0, 1).
    t_func : {"pe", "weibull"}, default="pe"
        Baseline competing-risk event-time distribution.
    n_intervals : int, default=5
        Requested piecewise-exponential intervals. Tied event quantiles can
        reduce the actual count. Ignored for a Weibull baseline.
    parametrization : {1, 2}, default=1
        DeepCure's link: 1 gives ``theta = exp(eta)``, 2 gives softplus.
    optimizer : {"rmsprop", "adam"}, default="rmsprop"
        Optimizer used to fit the upstream network and baseline parameters.
    learning_rate : float, default=1e-3
        Positive optimizer learning rate.
    decay_steps : int, default=100
        Positive inverse-time learning-rate decay interval.
    decay_rate : float, default=0.5
        Non-negative inverse-time decay rate.
    use_lr_decay : bool, default=True
        Apply inverse-time learning-rate decay.
    epochs : int, default=500
        Maximum epochs for the initial training run.
    batch_size : int, default=32
        Maximum observations per training batch.
    validation_fraction : float, default=0.15
        Internal holdout fraction, in [0, 1). Zero disables early stopping and
        fits all observations for ``epochs`` epochs.
    patience : int, default=10
        Epochs without sufficient validation improvement before stopping.
    min_delta : float, default=0.0
        Minimum validation-loss improvement used for checkpoint selection.
    random_state : int, RandomState instance or None, default=42
        Controls the holdout, network initialization, dropout, and shuffling.
    verbose : {0, 1, 2}, default=0
        Keras training verbosity.
    refit_full_data : bool, default=True
        After validation selects an epoch, initialize a fresh model and fit all
        observations for that many epochs. False retains the selected model.
    deterministic : bool, default=True
        Enable deterministic TensorFlow operations. TensorFlow's determinism
        setting applies to the process.

    Attributes
    ----------
    model_ : deepcure.models.DeepPTM
        Fitted upstream TensorFlow model.
    baseline_parameters_ : ndarray
        Fitted interval rates, or Weibull parameters ordered as shape, scale.
    breakpoints_ : ndarray or None
        Finite piecewise-exponential breakpoints; None for Weibull models.
    t_func_, parametrization_ : str, int
        Configuration used by the fitted model.
    history_, refit_history_ : dict, dict or None
        Keras training histories; the second is present when full-data refitting
        follows validation selection.
    best_epoch_ : int
        Selected epoch, counting from one. Respects ``min_delta``.
    n_iter_ : int
        Completed epochs in the run producing ``model_``.
    unique_times_ : ndarray
        Sorted unique training times, including censoring observations.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray
        Feature names when all training column names are strings.

    Notes
    -----
    Population survival is ``exp(-theta * F0(t))`` and cure probability is
    ``exp(-theta)``. Piecewise-exponential prediction continues the final fitted
    rate beyond the last finite breakpoint, making the baseline proper and
    the limiting population survival equal to the reported cure probability.
    Every call to fit initializes a new network.
    Training uses double precision and a local stable likelihood endpoint on
    the upstream DeepPTM architecture. Baseline coefficients are constrained to
    remain strictly positive.
    """

    n_features_in_: int
    baseline_parameters_: np.ndarray
    breakpoints_: np.ndarray | None

    def __init__(
        self,
        hidden_layers=(32, 16),
        activation="relu",
        dropout=0.20,
        t_func="pe",
        n_intervals=5,
        parametrization=1,
        optimizer="rmsprop",
        learning_rate=1e-3,
        decay_steps=100,
        decay_rate=0.5,
        use_lr_decay=True,
        epochs=500,
        batch_size=32,
        validation_fraction=0.15,
        patience=10,
        min_delta=0.0,
        random_state=42,
        verbose=0,
        refit_full_data=True,
        deterministic=True,
    ):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.t_func = t_func
        self.n_intervals = n_intervals
        self.parametrization = parametrization
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.use_lr_decay = use_lr_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state
        self.verbose = verbose
        self.refit_full_data = refit_full_data
        self.deterministic = deterministic

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)

    def __getstate__(self):
        state = super().__getstate__()
        if not isinstance(state, dict):
            raise TypeError("Estimator state must be a dictionary.")
        state = state.copy()
        model = state.pop("model_", None)
        if model is not None and self.__sklearn_is_fitted__():
            # Upstream DeepPTM's Keras config cannot reconstruct its custom
            # endpoint. Save standard layers and weights instead.
            tf, _ = self._imports()
            state["_keras_state"] = {
                "layers": [
                    tf.keras.layers.serialize(layer) for layer in model.stack_eta
                ],
                "weights": model.get_weights(),
                "optimizer": tf.keras.optimizers.serialize(model.optimizer),
            }
        return state

    def __setstate__(self, state):
        network = state.pop("_keras_state", None)
        super().__setstate__(state)
        if network is None:
            return
        tf, _ = self._imports()
        from ._loss import _StableDeepPTM

        model = _StableDeepPTM(
            stack_eta=[
                tf.keras.layers.deserialize(layer) for layer in network["layers"]
            ],
            t_func=self.t_func_,
            break_val=(
                None
                if self.breakpoints_ is None
                else tf.convert_to_tensor(self.breakpoints_, dtype=tf.float64)
            ),
            parametrization=self.parametrization_,
        )
        model.compile(optimizer=tf.keras.optimizers.deserialize(network["optimizer"]))
        model(
            {
                "x": tf.zeros((1, self.n_features_in_), dtype=tf.float64),
                "label_event": tf.zeros(1, dtype=tf.float64),
                "time_event": tf.convert_to_tensor(
                    self.unique_times_[:1], dtype=tf.float64
                ),
            }
        )
        model.set_weights(network["weights"])
        self.model_ = model

    @staticmethod
    def _imports():
        try:
            import tensorflow as tf  # pyright: ignore[reportMissingModuleSource]
            from deepcure.models import DeepPTM  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "DeepPromotionTimeCure requires TensorFlow and DeepCure. "
                "Install the optional dependencies with "
                "`pip install -r requirements-deep-learning.txt` from the project "
                "checkout, which installs the tested DeepCure revision."
            ) from exc
        return tf, DeepPTM

    def _validate_parameters(self):
        def integer(name, minimum):
            value = getattr(self, name)
            if (
                not isinstance(value, Integral)
                or isinstance(value, (bool, np.bool_))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")

        def real(name, minimum, *, strict=False):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, (bool, np.bool_))
                or not isfinite(value)
                or (value <= minimum if strict else value < minimum)
            ):
                relation = ">" if strict else ">="
                raise ValueError(f"{name} must be finite and {relation} {minimum}.")

        if not isinstance(self.hidden_layers, (tuple, list)) or any(
            not isinstance(size, Integral)
            or isinstance(size, (bool, np.bool_))
            or size < 1
            for size in self.hidden_layers
        ):
            raise ValueError("hidden_layers must be a sequence of positive integers.")
        if not isinstance(self.activation, str):
            raise TypeError("activation must be a Keras activation name.")
        if self.t_func not in ("pe", "weibull"):
            raise ValueError("t_func must be 'pe' or 'weibull'.")
        if self.optimizer not in ("rmsprop", "adam"):
            raise ValueError("optimizer must be 'rmsprop' or 'adam'.")
        integer("parametrization", 1)
        if self.parametrization not in (1, 2):
            raise ValueError("parametrization must be 1 or 2.")
        if self.t_func == "pe":
            integer("n_intervals", 1)
        for name in ("epochs", "batch_size", "decay_steps"):
            integer(name, 1)
        integer("patience", 0)
        integer("verbose", 0)
        if self.verbose > 2:
            raise ValueError("verbose must be 0, 1, or 2.")
        real("learning_rate", 0.0, strict=True)
        for name in ("decay_rate", "min_delta", "dropout", "validation_fraction"):
            real(name, 0.0)
        if self.dropout >= 1:
            raise ValueError("dropout must belong to [0, 1).")
        if self.validation_fraction >= 1:
            raise ValueError("validation_fraction must belong to [0, 1).")
        for name in ("use_lr_decay", "refit_full_data", "deterministic"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean.")
        check_random_state(self.random_state)

    def _split_for_early_stopping(self, event, seed):
        indices = np.arange(len(event))
        if self.validation_fraction == 0:
            return indices, indices[:0]
        n_validation = min(
            int(np.ceil(len(event) * self.validation_fraction)), len(event) - 1
        )
        _, counts = np.unique(event, return_counts=True)
        stratify = (
            event
            if len(counts) == 2
            and counts.min() >= 2
            and n_validation >= 2
            and len(event) - n_validation >= 2
            else None
        )
        train, validation = train_test_split(
            indices, test_size=n_validation, random_state=seed, stratify=stratify
        )
        if not np.any(event[train]):
            # Keep a failure observation for the likelihood, including when the
            # data contain only one event and stratification is unavailable.
            position = np.flatnonzero(event[validation])[0]
            train[0], validation[position] = validation[position], train[0]
        return train, validation

    def _build_breakpoints(self, t, event, *, support_max) -> np.ndarray | None:
        if self.t_func != "pe":
            return None
        breaks = np.quantile(t[event], np.linspace(0.0, 1.0, self.n_intervals + 1))
        breaks[0] = 0.0
        # The local endpoint and network use the same double-precision knots.
        with np.errstate(over="ignore"):
            breaks[-1] = np.nextafter(float(support_max), np.inf)
        breaks = np.unique(breaks).astype(float)
        if len(breaks) < 2 or np.any(~np.isfinite(breaks)):
            raise ValueError("Training times cannot form finite baseline breakpoints.")
        return breaks

    def _build_model(self, breakpoints, seed):
        tf, _ = self._imports()
        from ._loss import _StableDeepPTM

        tf.keras.utils.set_random_seed(seed)
        if self.deterministic:
            tf.config.experimental.enable_op_determinism()
        layers = []
        for index, size in enumerate(self.hidden_layers):
            layers.append(
                tf.keras.layers.Dense(
                    int(size), activation=self.activation, name=f"hidden_{index + 1}",
                    dtype="float64",
                )
            )
            if self.dropout:
                layers.append(tf.keras.layers.Dropout(self.dropout, dtype="float64"))
        layers.append(tf.keras.layers.Dense(1, activation="linear", name="eta", dtype="float64"))
        model = _StableDeepPTM(
            stack_eta=layers,
            t_func=self.t_func,
            break_val=(
                None
                if breakpoints is None
                else tf.convert_to_tensor(breakpoints, dtype=tf.float64)
            ),
            parametrization=int(self.parametrization),
        )
        learning_rate = float(self.learning_rate)
        if self.use_lr_decay:
            learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
                learning_rate, int(self.decay_steps), float(self.decay_rate)
            )
        optimizer_class = (
            tf.keras.optimizers.Adam
            if self.optimizer == "adam"
            else tf.keras.optimizers.RMSprop
        )
        model.compile(optimizer=optimizer_class(learning_rate=learning_rate))
        return model

    @staticmethod
    def _validate_history(history, *, validation=False):
        for name in (("loss", "val_loss") if validation else ("loss",)):
            values = np.asarray(history.get(name, []), dtype=float)
            if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
                raise FloatingPointError(f"DeepCure returned invalid {name} history.")

    def _fit_model(
        self, model, X, t, event, *, validation_data=None, epochs, early_stopping=False
    ):
        tf, _ = self._imports()

        def dataset(features, times, events, *, shuffle=False):
            with np.errstate(over="ignore", under="ignore"):
                arrays = {
                    "x": np.asarray(features, dtype=np.float64),
                    "time_event": np.asarray(times, dtype=np.float64),
                    "label_event": np.asarray(events, dtype=np.float64),
                }
            if any(np.any(~np.isfinite(array)) for array in arrays.values()):
                raise ValueError("DeepCure inputs must be representable as float64.")
            if np.any(arrays["time_event"] <= 0):
                raise ValueError(
                    "DeepCure training times must remain positive in float64."
                )
            result = tf.data.Dataset.from_tensor_slices(arrays)
            if shuffle:
                result = result.shuffle(len(features), seed=self._seed_)
            result = result.batch(min(self.batch_size, len(features)))
            options = tf.data.Options()
            # All data are already in memory; a small input pool also avoids
            # creating a new large worker pool for each cross-validation fit.
            options.threading.private_threadpool_size = 1
            options.threading.max_intra_op_parallelism = 1
            options.experimental_deterministic = self.deterministic
            return result.with_options(options)

        callbacks = [tf.keras.callbacks.TerminateOnNaN()]
        stopper = None
        if early_stopping:
            stopper = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=self.patience,
                min_delta=self.min_delta,
                restore_best_weights=True,
            )
            callbacks.append(stopper)
        validation = None if validation_data is None else dataset(*validation_data)
        result = model.fit(
            dataset(X, t, event, shuffle=True),
            validation_data=validation,
            epochs=int(epochs),
            callbacks=callbacks,
            verbose=self.verbose,
            shuffle=False,
        )
        history = {name: list(values) for name, values in result.history.items()}
        self._validate_history(history, validation=early_stopping)
        best_epoch = (
            int(stopper.best_epoch) + 1 if stopper is not None else len(history["loss"])
        )
        return history, best_epoch

    def fit(self, X, y) -> "DeepPromotionTimeCure":
        """Fit numeric features and a structured boolean-event/time target.

        Training times must be strictly positive and include at least one
        observed event. Feature names and the prediction time grid follow the
        same conventions as the other scikit-cure estimators.
        """
        self._is_fitted = False
        self._validate_parameters()
        X, event, t = self._validate_fit_data(X, y)
        self._imports()
        self._seed_ = int(check_random_state(self.random_state).randint(0, 2**31 - 1))
        train, validation = self._split_for_early_stopping(event, self._seed_)
        support_max = np.max(t)
        breaks = self._build_breakpoints(
            t[train], event[train], support_max=support_max
        )
        model = self._build_model(breaks, self._seed_)
        selection = len(validation) > 0
        history, best_epoch = self._fit_model(
            model,
            X[train],
            t[train],
            event[train],
            validation_data=(
                (X[validation], t[validation], event[validation]) if selection else None
            ),
            epochs=self.epochs,
            early_stopping=selection,
        )
        self._validate_history(history, validation=selection)
        if not isinstance(best_epoch, Integral) or not 1 <= best_epoch <= len(
            history["loss"]
        ):
            raise RuntimeError("DeepCure returned an invalid selected epoch.")
        self.history_ = history
        self.best_epoch_ = int(best_epoch)
        self.refit_history_ = None
        if selection and self.refit_full_data:
            breaks = self._build_breakpoints(t, event, support_max=support_max)
            model = self._build_model(breaks, self._seed_)
            self.refit_history_, _ = self._fit_model(
                model, X, t, event, epochs=self.best_epoch_, early_stopping=False
            )
            self._validate_history(self.refit_history_)
        baseline = np.asarray(model.endpoint.get_weights()[0], dtype=float).ravel()
        expected = len(breaks) - 1 if breaks is not None else 2
        if (
            baseline.shape != (expected,)
            or np.any(~np.isfinite(baseline))
            or np.any(baseline <= 0)
        ):
            raise FloatingPointError("DeepCure fitted invalid baseline parameters.")
        self.model_ = model
        self.baseline_parameters_ = baseline.copy()
        self.breakpoints_ = None if breaks is None else breaks.copy()
        self.t_func_ = self.t_func
        self.parametrization_ = int(self.parametrization)
        self.n_iter_ = len(
            self.refit_history_["loss"]
            if self.refit_history_ is not None
            else self.history_["loss"]
        )
        self.unique_times_ = np.unique(t)
        self._is_fitted = True
        return self

    def _predict_eta_from_validated_X(self, X) -> np.ndarray:
        tf, _ = self._imports()
        values = tf.convert_to_tensor(X, dtype=tf.float64)
        for layer in self.model_.stack_eta:
            values = layer(values, training=False)
        eta = np.asarray(values, dtype=float).reshape(-1)
        if eta.shape != (len(X),) or np.any(~np.isfinite(eta)):
            raise FloatingPointError(
                "DeepCure returned a non-finite incidence predictor."
            )
        return eta

    def predict(self, X) -> np.ndarray:
        """Return the neural predictor; larger values indicate greater risk."""
        return self.predict_eta(X)

    def predict_eta(self, X) -> np.ndarray:
        """Return the fitted neural incidence predictor, with dropout disabled."""
        return self._predict_eta_from_validated_X(self._validate_predict_data(X))

    def _log_theta(self, eta):
        if self.parametrization_ == 1:
            return eta
        eta = np.asarray(eta, dtype=float)
        result = eta.copy()
        regular = eta >= -36.0
        result[regular] = np.log(np.logaddexp(0.0, eta[regular]))
        return result

    def predict_theta(self, X) -> np.ndarray:
        """Return the fitted Poisson mean number of latent event causes."""
        log_theta = self._log_theta(self.predict_eta(X))
        if np.any(log_theta > _maximum_log_float()):
            raise FloatingPointError("Predicted latent mean exceeds double precision.")
        return _safe_exp(log_theta)

    def predict_latent_mean(self, X) -> np.ndarray:
        """Return the expected latent count under the fitted neural link."""
        return self.predict_theta(X)

    def predict_cure_probability(self, X) -> np.ndarray:
        """Return the limiting population survival probability, exp(-theta)."""
        return _exp_minus_exp(self._log_theta(self.predict_eta(X)))

    def _baseline_log_cdf(self, times: np.ndarray) -> np.ndarray:
        """Evaluate log F0 directly, including arbitrarily small positive times."""
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            if self.t_func_ == "weibull":
                shape, scale = self.baseline_parameters_
                log_hazard = shape * (np.log(times) - np.log(scale))
            else:
                breakpoints = self.breakpoints_
                if breakpoints is None:
                    raise RuntimeError(
                        "Piecewise-exponential baseline requires fitted breakpoints."
                    )
                widths = np.diff(breakpoints)
                elapsed = np.clip(times[:, None] - breakpoints[:-1], 0.0, widths)
                elapsed[:, -1] = np.maximum(times - breakpoints[-2], 0.0)
                log_hazard = logsumexp(
                    np.log(elapsed) + np.log(self.baseline_parameters_), axis=1
                )
            regular = log_hazard >= -36.0
            result = log_hazard.copy()
            result[regular] = np.log(
                -np.expm1(-np.exp(np.minimum(log_hazard[regular], 36.0)))
            )
            return result

    def _baseline_cdf(self, times: np.ndarray) -> np.ndarray:
        with np.errstate(under="ignore"):
            return np.exp(self._baseline_log_cdf(times))

    def predict_log_survival(self, X, times) -> np.ndarray:
        """Evaluate log population survival without survival underflow."""
        X = self._validate_predict_data(X)
        times = self._validate_times(times)
        log_theta = self._log_theta(self._predict_eta_from_validated_X(X))
        log_cdf = self._baseline_log_cdf(times)
        return -_safe_exp(log_theta[:, None] + log_cdf[None, :])

    def predict_survival(self, X, times) -> np.ndarray:
        """Evaluate continuous population survival at finite non-negative times."""
        X = self._validate_predict_data(X)
        times = self._validate_times(times)
        log_theta = self._log_theta(self._predict_eta_from_validated_X(X))
        log_cdf = self._baseline_log_cdf(times)
        return _exp_minus_exp(log_theta[:, None] + log_cdf[None, :])
