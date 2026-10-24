"""Run outside the checkout against an installed wheel or sdist."""

from importlib.metadata import metadata, version
import sys
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sksurv.util import Surv

import skcure
from skcure.datasets import simulate_cure_survival
from skcure.ensemble import GradientBoostingCure
from skcure.ggkm import GGKM
from skcure.linear_model import MixtureCure, PromotionTimeCure


assert "jax" not in sys.modules, "Core imports eagerly loaded JAX."
assert "tensorflow" not in sys.modules, "Core imports eagerly loaded TensorFlow."
assert skcure.__version__ == version("skcure")
project_metadata = metadata("skcure")
assert project_metadata["License-Expression"] == "GPL-3.0-or-later"
assert not any(" @ " in value for value in project_metadata.get_all("Requires-Dist", []))
assert "deepcure" not in sys.modules

for family in ("binomial", "poisson", "negative_binomial"):
    sample = simulate_cure_survival(32, latent_distribution=family, K_bin=3, random_state=73)
    np.testing.assert_array_equal(sample.cured, sample.latent_count == 0)
    truth = sample.survival_probability([0, 1, 5])
    assert truth.shape == (32, 3)
    assert np.all(truth[:, 0] == 1)
    assert np.all(np.diff(truth, axis=1) <= 0)

rng = np.random.default_rng(73)
X = rng.normal(size=(24, 2))
y = Surv.from_arrays(np.tile([True, False, True, True], 6), np.linspace(.2, 3, 24))
for estimator in (GGKM(max_em_iter=2), MixtureCure(max_em_iter=2),
                  PromotionTimeCure(max_em_iter=2),
                  GradientBoostingCure(n_estimators=1, max_em_iter=2)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = clone(estimator).fit(X, y)
    survival = fitted.predict_survival(X[:3], [0, 1, 3])
    assert survival.shape == (3, 3)
    assert np.all(np.isfinite(survival))
    assert np.all(np.diff(survival, axis=1) <= 1e-12)
    assert fitted.predict_latent_mean(X[:3]).shape == (3,)
    assert fitted.termination_reason_ != "running"
assert "jax" not in sys.modules
assert "tensorflow" not in sys.modules
print(f"Installed skcure {skcure.__version__}: imports, metadata, fit, and predictions OK.")
