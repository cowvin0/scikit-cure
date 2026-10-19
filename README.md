# skcure

`skcure` develops cure models using the estimator API of
[scikit-survival](https://scikit-survival.readthedocs.io/). Public estimators share
the structured survival target, risk prediction, concordance score, and
survival/cumulative-hazard step function interface.

| Estimator | Model |
| --- | --- |
| `GGKM` | Generalized gamma kernel cure model with Poisson, negative binomial, or binomial latent counts. |
| `GGBinomialGB` | Gradient-boosted binomial generalized gamma cure model, including Bernoulli counts with `K_bin=1`. |
| `GGPoissonGB` | Gradient-boosted Poisson generalized gamma cure model. |
| `GGNBGB` | Gradient-boosted negative binomial generalized gamma cure model. |
| `MixtureCureModel` | Generalized gamma mixture cure model with logistic susceptibility. |
| `PromotionTimeModel` | Generalized gamma promotion time cure model. |

Import these estimators from `skcure.models` or directly from `skcure`.

The primal and spline baselines used for the GGKM paper are research code in
[`ggkm_paper.models`](ggkm_paper/models). The primal estimators reuse the shared
linear cure-model, latent-count, generalized-gamma latency, and EM components
from `skcure`; the spline estimators remain paper-specific wrappers around those
primal models. They are imported from `ggkm_paper.models` when running
experiments from a repository checkout and are excluded from library
distributions. See the [paper instructions](ggkm_paper/README.md).

The package is under development. Other model and utility modules in this
research repository remain experimental and have not been converted to the
public estimator API.

## Installation

Python 3.10 or later is required. Install the package from this repository:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

JAX is used for generalized gamma derivatives. A standard CPU JAX installation
is sufficient; GPU support is not required.

## Usage

As in scikit-survival, `y` must be a one-dimensional structured array with a
boolean event indicator as its first field and the observed event or censoring
time as its second field. An event indicator of `True` means the event was
observed. `GGKM` requires strictly positive training times and at least one
observed event.

```python
import numpy as np
from sksurv.util import Surv

from skcure.models import GGKM

rng = np.random.default_rng(73)
X = rng.normal(size=(24, 2))
y = Surv.from_arrays(
    event=np.tile([True, False, True, True], 6),
    time=np.round(rng.uniform(0.4, 3.2, size=24), 3),
)

model = GGKM(
    latent_distribution="poisson",
    kernel="rbf",
    gamma=0.3,
    lambda_reg=0.1,
    tol=1e-5,
    max_em_iter=2,
)
model.fit(X, y)

risk = model.predict(X)
concordance = model.score(X, y)
cure_probability = model.predict_cure_probability(X)

# Standard scikit-survival outputs: one StepFunction per sample.
survival_functions = model.predict_survival_function(X)
cumulative_hazard_functions = model.predict_cumulative_hazard_function(X)

# Arrays use the training grid stored in model.unique_times_.
survival_array = model.predict_survival_function(X, return_array=True)

# Evaluate the parametric survival model at a chosen time grid.
survival_at_times = model.predict_survival(X, times=[1.0, 3.0, 6.0])
```

This short example limits fitting to two EM iterations to demonstrate the API;
it does not establish convergence. For an analysis, choose solver settings for
the data, inspect `converged_`, and evaluate performance on held-out data.

## Estimator contract

`skcure.base.BaseCureEstimator` inherits scikit-survival's
`SurvivalAnalysisMixin` and scikit-learn's `BaseEstimator`. It provides shared
validation and survival prediction behavior for cure estimators. `GGKM` follows
this contract:

| Method or attribute | Behavior |
| --- | --- |
| `fit(X, y)` | Validates numeric features and the structured survival target, fits the model, and returns `self`. |
| `predict(X)` | Returns the linear predictor `eta`; larger values indicate greater risk. |
| `score(X, y)` | Returns the concordance index through `SurvivalAnalysisMixin`. |
| `predict_survival_function(X, return_array=False)` | Returns an array of `sksurv.functions.StepFunction` objects, or a matrix when `return_array=True`. |
| `predict_cumulative_hazard_function(X, return_array=False)` | Returns cumulative hazard StepFunctions, or a matrix when `return_array=True`. |
| `unique_times_` | Sorted unique observed training times, including censoring times; defines the columns of prediction matrices. |
| `predict_survival(X, times)` | Evaluates parametric survival probabilities at the supplied times. |
| `predict_cure_probability(X)` | Returns the estimated long-term cure probability for each sample. |

Hyperparameters, including `tol` and `max_em_iter`, belong in the constructor
and can be accessed with `get_params` or changed with `set_params`. Validation
happens in `fit`. Fitting preserves constructor values such as `a`, `d`, `p`,
and `phi`; fitted values are exposed separately as `a_`, `d_`, `p_`, and `phi_`.
The estimator supports cloning, scikit-learn pipelines, and parameter search
with survival targets.

With `warm_start=True`, a subsequent fit reuses the fitted generalized gamma
parameters and estimated negative binomial dispersion. Predictor coefficients
are initialized again on every fit. Changing the latent distribution or its
configuration creates a fresh latent model.

The parametric estimators use the same `fit(X, y)`, `predict`,
`score`, and curve methods. Their solver controls (`tol`, `max_em_iter`) also
belong in the constructor, and their learned gamma parameters are `a_`, `d_`,
and `p_`. Negative binomial models expose the fixed or estimated dispersion as
`phi_`, with `phi=None` requesting estimation.

`PromotionTimeModel` is defined in `skcure.models.promotion_time`, and
`MixtureCureModel` in `skcure.models.mixture_cure`. Both remain available from
`skcure`, `skcure.models`, and their previous `skcure.models.baseline_model`
import path. Existing serialized estimators using that path can still load.

For all these models, the default `warm_start=False` resets learned parameters
on refit. Setting `warm_start=True` reuses gamma parameters and estimated
dispersion while initializing regression coefficients again. This reproduces
the older parametric models' implicit reuse of gamma parameters on refit.
Concordance scores use increasing risk: higher linear predictors mean lower
survival or cure probability.

The regression fixtures in `tests/data/legacy_models.json` were captured from
the original parametric implementations before their API migration. They cover
predictions, learned parameters, and likelihood histories. The paper-specific
tests in `ggkm_paper/tests` verify estimator behavior, delegation to shared
`skcure` components, regularization, dispersion handling, warm starts, and
spline wrapping without freezing private optimizer state or bit-for-bit
implementation traces.

Existing research code that passes times and event indicators separately to
`fit`, or passes solver settings to `fit`, must be updated: construct `y` with
`Surv.from_arrays` and move solver settings to the estimator constructor.

## Shared model architecture and EM implementation

`GGKM`, the linear parametric estimators, and the paper-specific primal
estimators reuse the internal EM machinery in `skcure.models._em`.
`run_em` accepts an E-step, ordered M-step callbacks, a likelihood callback,
and mutable iteration/history state, allowing models with different predictor
representations to share the same orchestration.

Latent-count behavior is centralized in the `_LatentCountModel` interface and
its Poisson, negative-binomial, and binomial implementations. Predictor
implementations provide a fixed linear predictor `eta`; the latent-count model
owns the corresponding incidence objective, observed likelihood, E-step,
population survival, and cure-probability calculations. Contract tests verify
that the concrete latent families satisfy this interface.

Susceptible event-time behavior is separated from the latent-count family.
`GeneralizedGamma`, defined in `skcure.distributions.generalized_gamma`, provides
the generalized-gamma density, log-density, CDF, and survival function.
`_GeneralizedGammaLatencyOptimizer` owns the generalized-gamma M-step. The
generic latency boundary is structural: latent-count models depend on the
latency distribution interface rather than on generalized-gamma-specific
parameters.

`PromotionTimeModel` and `MixtureCureModel` share their linear predictor
implementation through `_BaseLinearCureModel`. `PromotionTimeModel` uses
`_PoissonLatentModel`, while `MixtureCureModel` uses `_BinomialLatentModel`
with one trial (`K_bin=1`). Their likelihood evaluation, E-steps, predictor
objectives, latency updates, survival prediction, and cure-probability
prediction therefore reuse the same core components used elsewhere in the
library.

The paper-specific `GGPoissonPrimal` and `GGNBPrimal` models also build on
`_BaseLinearCureModel`. Their paper layer adds the L2 penalty on slope
coefficients, while the intercept remains unpenalized. `GGNBPrimal` delegates
dispersion estimation to `_NegativeBinomialLatentModel`, preserving the EM
update order

```text
predictor -> generalized-gamma latency -> dispersion
```

The spline baselines remain paper-specific. `SplinePoissonModel` and
`SplineNegativeBinomialModel` apply `SplineTransformer` to the original
features and delegate fitting and prediction to the corresponding paper primal
estimator. This keeps feature engineering in `ggkm_paper` while the statistical
and numerical machinery remains shared with `skcure`.

The pre-extraction GGKM implementation is still covered by
[`tests/data/ggkm_em_reference.json`](tests/data/ggkm_em_reference.json), with
provenance documented in
[`tests/data/ggkm_em_reference.md`](tests/data/ggkm_em_reference.md).
The paper primal and spline tests no longer freeze private optimizer traces or
complete estimator instance state. Instead they test public regression behavior
and the intended delegation boundaries to shared `skcure` components.

## Gradient boosting

`GGBinomialGB`, `GGPoissonGB`, and `GGNBGB` use the same estimator API. For
example, `GGPoissonGB(n_estimators=3, max_em_iter=3, random_state=73).fit(X, y)`
fits a short ensemble to a structured survival target. Move existing fit-time
`tol` and `max_em_iter` settings to the constructor. Learned parameters use
`a_`, `d_`, `p_`, and, for negative binomial models, `phi_`.

The original algorithm adds up to `n_estimators` trees **per EM iteration**;
this behavior is preserved. `models_` and `model_weights_` store all fitted
trees and their learning-rate-scaled line-search weights. `n_estimators_` is
the total number of fitted trees. A warm fit reuses gamma parameters and
estimated dispersion, then initializes a new ensemble.

The three models share a boosting loop and reuse the same latent-count
objectives, E-steps, generalized-gamma latency machinery, and survival/cure
calculations used by the other cure estimators. Generalized-gamma numerical
work is centralized in `skcure.distributions.generalized_gamma` and the shared
latency optimizer. Boosting uses log-domain generalized-gamma arguments and
`gammaincc` tails,
avoids arbitrary negative-binomial predictor clipping, and avoids forming
overflowing Poisson intensities when predicting survival. An unrepresentable
latent mean requested through `predict_theta` raises `FloatingPointError`.
If a bounded gamma or dispersion update fails, its positive log-parameter
formulation is retried; a failed retry is reported instead of silently keeping
old parameters.

These numerical fixes can change the fitted optimization path. The legacy
boosting fixtures and measured differences are documented in
[`tests/data/boost_legacy.md`](tests/data/boost_legacy.md). Fixed-latency
comparisons isolate and check the retained tree-boosting algorithm; full fits
are compared separately and are not claimed to be numerically identical.

## Kernels

Standard kernel names use
[`sklearn.metrics.pairwise.pairwise_kernels`](https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html).
Supported names include `linear`, `rbf`, `poly`/`polynomial`, `laplacian`,
`sigmoid`, `cosine`, `chi2`, and `additive_chi2`. The `gaussian` alias maps to
`rbf`. The existing `exponential`, `cauchy`, and `wavelet` kernels are implemented
as standalone functions in `skcure.utils.kernels`.

Custom callables retain GGKM's vectorized interface: they receive two feature
matrices and return a matrix of shape `(len(X), len(Y))`. Parameters from
`kernel_params` are passed to the callable. This differs from scikit-learn's
row-pair callable interface.

```python
def scaled_linear(X, Y, scale=1.0):
    return scale * (X @ Y.T)

model = GGKM(kernel=scaled_linear, kernel_params={"scale": 2.5})
```

With `kernel="precomputed"`, pass a square training Gram matrix to `fit` and
a matrix of shape `(n_new_samples, n_train_samples)` to prediction methods.
Columns must follow the original training sample order. The estimator's
pairwise tag lets scikit-learn parameter search slice both matrix dimensions.

Scikit-learn's distance calculation can differ from the previous RBF
implementation at floating-point precision, so fitted parameters and
predictions can differ slightly.

## Package layout

```text
skcure/
    __init__.py
    base.py
    distributions/
        __init__.py
        base.py
        generalized_gamma.py
    models/
        __init__.py
        ggkm.py
        gdbt_cure.py
        promotion_time.py
        mixture_cure.py
        baseline_model.py
        _linear_cure.py
        _latency.py
        _em.py
        latent.py
        state.py
    utils/
        __init__.py
        estimator.py
        kernels.py
        numerics.py
        pgamma_derivate.py

ggkm_paper/
    models/
        __init__.py
        gg_primal.py
        splines.py
    scripts/
    evaluate/
    tests/
```

`skcure` and its package submodules form the installable library. Paper-only
models, tests, experiments, datasets, and evaluation code under `ggkm_paper`
remain repository research code and are excluded from library distributions.
The public API is intentionally smaller than the repository: internal modules
whose names begin with an underscore are implementation details, and other
research utilities should not be treated as stable public interfaces unless
they are explicitly exported.

Build source and wheel distributions with:

```bash
python -m build
```
