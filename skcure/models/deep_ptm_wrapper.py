from __future__ import annotations

import random
import numpy as np


class DeepPTMModel:

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

        self.hidden_layers = tuple(int(v) for v in hidden_layers)
        self.activation = activation
        self.dropout = float(dropout)
        self.t_func = t_func
        self.n_intervals = int(n_intervals)
        self.parametrization = int(parametrization)
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.decay_steps = int(decay_steps)
        self.decay_rate = float(decay_rate)
        self.use_lr_decay = bool(use_lr_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.validation_fraction = float(validation_fraction)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.random_state = int(random_state)
        self.verbose = int(verbose)
        self.refit_full_data = bool(refit_full_data)
        self.deterministic = bool(deterministic)
        self.model_ = None
        self.break_val_ = None
        self.best_epoch_ = None
        self.history_ = None
        self.refit_history_ = None
        self.n_features_in_ = None
        self._validate_parameters()

    def _validate_parameters(
        self,
    ):

        if any(units <= 0 for units in self.hidden_layers):
            raise ValueError("All hidden-layer sizes must " "be positive.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must satisfy " "0 <= dropout < 1.")

        if self.t_func not in {
            "pe",
            "weibull",
        }:
            raise ValueError("t_func must be either " "'pe' or 'weibull'.")

        if self.n_intervals < 1:
            raise ValueError("n_intervals must be positive.")

        if self.parametrization not in {
            1,
            2,
        }:
            raise ValueError("parametrization must be " "1 or 2.")

        if self.optimizer not in {
            "rmsprop",
            "adam",
        }:
            raise ValueError("optimizer must be either " "'rmsprop' or 'adam'.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.decay_steps < 1:
            raise ValueError("decay_steps must be positive.")
        if not np.isfinite(self.decay_rate) or self.decay_rate < 0.0:
            raise ValueError("decay_rate must be " "finite and non-negative.")
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.validation_fraction <= 0.0 or self.validation_fraction >= 0.5:
            raise ValueError(
                "validation_fraction must " "satisfy 0 < validation_fraction < 0.5."
            )
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative.")

    @staticmethod
    def _imports():

        try:

            import tensorflow as tf

        except ImportError as exc:

            raise ImportError("TensorFlow is required to use " "DeepPTMModel.") from exc

        try:

            from deepcure.models import (
                DeepPTM,
            )

        except ImportError as exc:

            raise ImportError(
                "The 'deepcure' package is required "
                "to use DeepPTMModel. Install it with "
                "'pip install "
                "git+https://github.com/"
                "vhmedina/deepcure.git'."
            ) from exc

        return (
            tf,
            DeepPTM,
        )

    def _set_random_seed(
        self,
    ):

        tf, _ = self._imports()

        random.seed(self.random_state)

        np.random.seed(self.random_state)

        tf.keras.utils.set_random_seed(self.random_state)

        if self.deterministic:

            try:

                (tf.config.experimental.enable_op_determinism())

            except Exception:

                pass

    @staticmethod
    def _validate_training_data(
        X,
        t,
        delta,
    ):

        X = np.asarray(
            X,
            dtype=float,
        )

        t = np.asarray(
            t,
            dtype=float,
        ).reshape(-1)

        delta = np.asarray(
            delta,
            dtype=float,
        ).reshape(-1)

        if X.ndim == 1:

            X = X.reshape(
                -1,
                1,
            )

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        n = X.shape[0]

        if len(t) != n or len(delta) != n:
            raise ValueError(
                "X, t, and delta must contain " "the same number of observations."
            )

        if n < 2:
            raise ValueError("At least two observations " "are required.")

        if np.any(~np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        if np.any(~np.isfinite(t)):
            raise ValueError("t must contain only finite values.")

        if np.any(t <= 0.0):
            raise ValueError("All survival times must be " "strictly positive.")

        if not np.all(
            np.isin(
                delta,
                [
                    0.0,
                    1.0,
                ],
            )
        ):
            raise ValueError("delta must contain only 0 and 1.")

        if np.sum(delta) < 1:
            raise ValueError("DeepPTM requires at least " "one observed event.")

        return (
            X,
            t,
            delta,
        )

    @staticmethod
    def _validate_X(
        X,
        n_features,
    ):

        X = np.asarray(
            X,
            dtype=float,
        )

        if X.ndim == 1:

            X = X.reshape(
                -1,
                1,
            )

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[1] != n_features:
            raise ValueError("X has an incorrect number " "of covariates.")

        if np.any(~np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        return X

    def _build_breakpoints(
        self,
        t,
        delta,
        *,
        support_max=None,
    ):

        if self.t_func != "pe":
            return None

        t = np.asarray(
            t,
            dtype=float,
        ).reshape(-1)

        delta = np.asarray(
            delta,
            dtype=float,
        ).reshape(-1)

        if t.size == 0 or delta.size != t.size:
            raise ValueError("Invalid data supplied for " "PE breakpoint construction.")

        if np.any(~np.isfinite(t)):
            raise ValueError("Times used for PE breakpoints " "must be finite.")

        if np.any(t <= 0.0):
            raise ValueError("Times used for PE breakpoints " "must be positive.")

        event_times = t[delta == 1.0]

        if event_times.size == 0:
            raise ValueError(
                "At least one event is required " "to construct PE breakpoints."
            )

        quantiles = np.linspace(
            0.0,
            1.0,
            self.n_intervals + 1,
        )

        break_values = np.quantile(
            event_times,
            quantiles,
        )

        break_values[0] = 0.0

        if support_max is None:
            maximum_time = float(np.max(t))

        else:
            support_max = float(support_max)

            if not np.isfinite(support_max) or support_max <= 0.0:
                raise ValueError("support_max must be a finite " "positive value.")

            maximum_time = max(
                float(np.max(t)),
                support_max,
            )

        break_values[-1] = np.nextafter(
            maximum_time,
            np.inf,
        )

        break_values = np.unique(break_values)

        if break_values.size < 2:
            raise ValueError(
                "Unable to construct valid " "piecewise-exponential breakpoints."
            )

        if np.any(~np.isfinite(break_values)):
            raise ValueError("PE breakpoints must be finite.")

        if np.any(np.diff(break_values) <= 0.0):
            raise ValueError("PE breakpoints must be " "strictly increasing.")

        return np.asarray(
            break_values,
            dtype=float,
        )

    def _to_tensor_dict(
        self,
        X,
        t,
        delta,
    ):

        tf, _ = self._imports()

        return {
            "x": tf.convert_to_tensor(
                X,
                dtype=tf.float64,
            ),
            "label_event": (
                tf.convert_to_tensor(
                    delta,
                    dtype=tf.float64,
                )
            ),
            "time_event": (
                tf.convert_to_tensor(
                    t,
                    dtype=tf.float64,
                )
            ),
        }

    def _build_stack_eta(
        self,
    ):

        tf, _ = self._imports()

        stack_eta = []

        for (
            layer_index,
            units,
        ) in enumerate(self.hidden_layers):

            stack_eta.append(
                tf.keras.layers.Dense(
                    units,
                    activation=(self.activation),
                    use_bias=True,
                    dtype=tf.float64,
                    name=(f"hidden_" f"{layer_index + 1}"),
                )
            )

            if self.dropout > 0.0:

                stack_eta.append(
                    tf.keras.layers.Dropout(
                        rate=(self.dropout),
                        name=(f"dropout_" f"{layer_index + 1}"),
                    )
                )

        stack_eta.append(
            tf.keras.layers.Dense(
                1,
                activation="linear",
                use_bias=True,
                dtype=tf.float64,
                name="eta",
            )
        )

        return stack_eta

    def _build_optimizer(
        self,
    ):

        tf, _ = self._imports()

        if self.use_lr_decay:

            learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=(self.learning_rate),
                decay_steps=(self.decay_steps),
                decay_rate=(self.decay_rate),
            )

        else:

            learning_rate = self.learning_rate

        if self.optimizer == "rmsprop":

            return tf.keras.optimizers.RMSprop(learning_rate=(learning_rate))

        return tf.keras.optimizers.Adam(learning_rate=(learning_rate))

    def _build_model(
        self,
        break_values,
    ):

        tf, DeepPTM = self._imports()

        stack_eta = self._build_stack_eta()

        if self.t_func == "pe":

            if break_values is None:

                raise ValueError("break_values are required " "when t_func='pe'.")

            break_val = tf.convert_to_tensor(
                break_values,
                dtype=tf.float64,
            )

            model = DeepPTM(
                stack_eta=stack_eta,
                t_func="pe",
                break_val=(break_val),
                parametrization=(self.parametrization),
            )

        else:

            model = DeepPTM(
                stack_eta=stack_eta,
                t_func="weibull",
                parametrization=(self.parametrization),
            )

        model.compile(optimizer=(self._build_optimizer()))

        return model

    def _split_for_early_stopping(
        self,
        X,
        t,
        delta,
    ):

        from sklearn.model_selection import (
            train_test_split,
        )

        indices = np.arange(len(t))

        values, counts = np.unique(
            delta,
            return_counts=True,
        )

        can_stratify = len(values) == 2 and np.min(counts) >= 2

        stratify = delta if can_stratify else None

        (
            train_idx,
            validation_idx,
        ) = train_test_split(
            indices,
            test_size=(self.validation_fraction),
            random_state=(self.random_state),
            shuffle=True,
            stratify=stratify,
        )

        if np.sum(delta[train_idx]) < 1:

            raise RuntimeError(
                "The internal DeepPTM training " "subset contains no observed events."
            )

        return (
            train_idx,
            validation_idx,
        )

    @staticmethod
    def _best_epoch_from_history(
        history,
    ):

        history_dict = history.history

        validation_loss = history_dict.get("val_loss")

        if validation_loss is None or len(validation_loss) == 0:

            loss = history_dict.get(
                "loss",
                [],
            )

            if len(loss) == 0:

                raise RuntimeError(
                    "DeepPTM training history " "contains no loss values."
                )

            return len(loss)

        validation_loss = np.asarray(
            validation_loss,
            dtype=float,
        )

        finite = np.isfinite(validation_loss)

        if not np.any(finite):

            raise RuntimeError("DeepPTM produced no finite " "validation loss.")

        finite_indices = np.flatnonzero(finite)

        best_index = finite_indices[np.argmin(validation_loss[finite])]

        return int(best_index + 1)

    def fit(
        self,
        X,
        t,
        delta,
    ):

        tf, _ = self._imports()

        (
            X,
            t,
            delta,
        ) = self._validate_training_data(
            X,
            t,
            delta,
        )

        self.n_features_in_ = X.shape[1]

        self.model_ = None

        self.break_val_ = None

        self.best_epoch_ = None

        self.history_ = None

        self.refit_history_ = None

        #
        # ------------------------------------------------
        # Stage 1
        #
        # Split the data supplied to fit() into an
        # internal training and early-stopping set.
        # ------------------------------------------------
        #
        (
            train_idx,
            validation_idx,
        ) = self._split_for_early_stopping(
            X,
            t,
            delta,
        )

        X_train = X[train_idx]

        t_train = t[train_idx]

        delta_train = delta[train_idx]

        X_validation = X[validation_idx]

        t_validation = t[validation_idx]

        delta_validation = delta[validation_idx]

        fit_support_max = float(np.max(t))

        break_values_stage1 = self._build_breakpoints(
            t_train,
            delta_train,
            support_max=(fit_support_max),
        )

        if self.t_func == "pe":
            if np.max(t_train) >= break_values_stage1[-1]:
                raise RuntimeError(
                    "Internal training times " "exceed the DeepPTM PE support."
                )

            if np.max(t_validation) >= break_values_stage1[-1]:
                raise RuntimeError(
                    "Internal validation times " "exceed the DeepPTM PE support."
                )

        tf.keras.backend.clear_session()

        self._set_random_seed()

        stage1_model = self._build_model(break_values_stage1)

        train_data = self._to_tensor_dict(
            X_train,
            t_train,
            delta_train,
        )

        validation_data = self._to_tensor_dict(
            X_validation,
            t_validation,
            delta_validation,
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=(self.patience),
            min_delta=(self.min_delta),
            mode="min",
            restore_best_weights=True,
        )

        history = stage1_model.fit(
            train_data,
            epochs=(self.epochs),
            batch_size=min(
                self.batch_size,
                len(train_idx),
            ),
            validation_data=(validation_data),
            callbacks=[early_stopping],
            verbose=(self.verbose),
            shuffle=True,
        )

        best_epoch = self._best_epoch_from_history(history)

        if best_epoch < 1:
            raise RuntimeError("Unable to determine a valid " "DeepPTM training epoch.")

        self.history_ = history.history
        self.best_epoch_ = best_epoch

        if self.refit_full_data:
            break_values_final = self._build_breakpoints(
                t,
                delta,
                support_max=float(np.max(t)),
            )

            if self.t_func == "pe":
                if np.max(t) >= break_values_final[-1]:
                    raise RuntimeError(
                        "Full training data exceed " "the DeepPTM PE support."
                    )

            tf.keras.backend.clear_session()

            self._set_random_seed()

            final_model = self._build_model(break_values_final)
            full_data = self._to_tensor_dict(
                X,
                t,
                delta,
            )

            refit_history = final_model.fit(
                full_data,
                epochs=(best_epoch),
                batch_size=min(
                    self.batch_size,
                    len(t),
                ),
                verbose=(self.verbose),
                shuffle=True,
            )

            self.model_ = final_model
            self.refit_history_ = refit_history.history

            if self.t_func == "pe":
                self.break_val_ = np.asarray(
                    break_values_final,
                    dtype=float,
                )

        else:

            self.model_ = stage1_model

            if self.t_func == "pe":
                self.break_val_ = np.asarray(
                    break_values_stage1,
                    dtype=float,
                )

        return self

    def _check_is_fitted(
        self,
    ):

        if self.model_ is None:
            raise RuntimeError("The model must be fitted " "before prediction.")

    def predict_eta(
        self,
        X_new,
    ):

        self._check_is_fitted()

        tf, _ = self._imports()
        X_new = self._validate_X(
            X_new,
            self.n_features_in_,
        )

        inputs = {
            "x": tf.convert_to_tensor(
                X_new,
                dtype=tf.float64,
            )
        }

        eta = self.model_.eta_predict(inputs)
        eta = np.asarray(
            eta,
            dtype=float,
        ).reshape(-1)

        if np.any(~np.isfinite(eta)):
            raise FloatingPointError("DeepPTM produced non-finite " "eta predictions.")

        return eta

    def predict_theta(
        self,
        X_new,
    ):

        self._check_is_fitted()

        tf, _ = self._imports()

        X_new = self._validate_X(
            X_new,
            self.n_features_in_,
        )

        inputs = {
            "x": tf.convert_to_tensor(
                X_new,
                dtype=tf.float64,
            )
        }

        theta = self.model_.theta_predict(inputs)

        theta = np.asarray(
            theta,
            dtype=float,
        ).reshape(-1)

        if np.any(np.isnan(theta)):

            raise FloatingPointError("DeepPTM produced NaN " "theta predictions.")

        if np.any(theta < 0.0):

            raise FloatingPointError("DeepPTM produced negative " "theta predictions.")

        return theta

    def predict_survival(
        self,
        X_new,
        t_grid,
    ):

        self._check_is_fitted()

        tf, _ = self._imports()

        X_new = self._validate_X(
            X_new,
            self.n_features_in_,
        )

        t_grid = np.asarray(
            t_grid,
            dtype=float,
        ).reshape(-1)

        if t_grid.size == 0:

            raise ValueError("t_grid must not be empty.")

        if np.any(~np.isfinite(t_grid)):

            raise ValueError("t_grid must contain only " "finite values.")

        if np.any(t_grid < 0.0):

            raise ValueError("t_grid must contain " "non-negative times.")

        n_subjects = X_new.shape[0]

        X_tensor = tf.convert_to_tensor(
            X_new,
            dtype=tf.float64,
        )

        survival = np.empty(
            (
                n_subjects,
                len(t_grid),
            ),
            dtype=float,
        )

        dummy_event = tf.zeros(
            n_subjects,
            dtype=tf.float64,
        )

        for (
            j,
            time_point,
        ) in enumerate(t_grid):

            time_tensor = tf.fill(
                [n_subjects],
                tf.cast(
                    time_point,
                    tf.float64,
                ),
            )

            inputs = {
                "x": (X_tensor),
                "label_event": (dummy_event),
                "time_event": (time_tensor),
            }

            predicted = self.model_.surv_predict(
                inputs,
                type="Sp",
            )

            survival[:, j] = np.asarray(
                predicted,
                dtype=float,
            ).reshape(-1)

        if np.any(~np.isfinite(survival)):

            raise FloatingPointError(
                "DeepPTM produced non-finite " "survival probabilities."
            )

        return np.clip(
            survival,
            0.0,
            1.0,
        )

    def predict_cure_probability(
        self,
        X_new,
    ):

        theta = self.predict_theta(X_new)
        cure_probability = np.exp(-theta)

        if np.any(~np.isfinite(cure_probability)):
            raise FloatingPointError(
                "DeepPTM produced non-finite " "cure probabilities."
            )

        return np.clip(
            cure_probability,
            0.0,
            1.0,
        )
