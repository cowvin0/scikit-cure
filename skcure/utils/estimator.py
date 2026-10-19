"""Adapters for research code using separate survival targets."""

import numpy as np
from sksurv.util import Surv

from skcure.base import BaseCureEstimator


def fit_cure_model(model, X, t, delta, **fit_params):
    """Fit a library estimator or a research model from times and events.

    Library estimators receive a scikit-survival structured target and use
    constructor parameters for solver settings. Research models retain their
    separate time/event arguments and fit-time settings.
    """
    if isinstance(model, BaseCureEstimator):
        delta = np.asarray(delta)
        if delta.ndim != 1 or not np.all(np.isin(delta, [0, 1])):
            raise ValueError("delta must be a one-dimensional array containing only 0 and 1.")

        y = Surv.from_arrays(delta.astype(bool), t)
        if fit_params:
            model.set_params(**fit_params)
        return model.fit(X, y)

    return model.fit(X, t, delta, **fit_params)
