# scikit-cure

scikit-cure is a Python library for cure models that extends the
[scikit-survival](https://scikit-survival.readthedocs.io/) estimator API.
It combines survival and cure-probability prediction with
[scikit-learn](https://scikit-learn.org/) pipelines, preprocessing, and
cross-validation. The Python package is named `skcure`; the source tree is
prepared for the first alpha, `0.1.0a1`.

## About cure models

Survival analysis models the time until an event while accounting for censored
observations. Cure models additionally allow a proportion of individuals to
never experience the event of interest. They estimate both the long-term cure
probability and survival over time.

## Models

| Estimator | Namespace | Approach |
| --- | --- | --- |
| `GGKM` | `skcure.ggkm` | Generalized gamma kernel cure model. |
| `GradientBoostingCure` | `skcure.ensemble` | Gradient-boosted generalized gamma cure model. |
| `MixtureCure` | `skcure.linear_model` | Mixture cure model with logistic susceptibility. |
| `PromotionTimeCure` | `skcure.linear_model` | Promotion time cure model with a linear predictor. |
| `DeepPromotionTimeCure` | `skcure.deep_learning` | Neural promotion time cure model wrapping [DeepCure](https://github.com/vhmedina/deepcure). |

`GGKM` and `GradientBoostingCure` select Poisson, negative binomial, or binomial
latent counts through `latent_distribution`. For boosting, `n_estimators` is
the maximum number of trees added per EM iteration; `max_total_estimators`
limits the full fit (default 1000).

## Requirements

Python 3.10 or later, NumPy 2.0+, SciPy 1.13+, scikit-learn, scikit-survival,
and Narwhals.
Dependency versions are specified in [pyproject.toml](pyproject.toml).
`DeepPromotionTimeCure` additionally requires TensorFlow and DeepCure, available
through the [optional installation](docs/installation.md). JAX is optional.

## Installation

From a repository checkout:

```bash
python -m pip install .
```

To include the optional neural model dependencies:

```bash
python -m pip install -r requirements-deep-learning.txt
```

## Example

As in scikit-survival, `y` is a structured array containing a boolean event
indicator followed by the observed event or censoring time. `True` means the
event was observed. Training times must be strictly positive, with at least
one observed event.

`X` accepts NumPy arrays, pandas and Polars DataFrames, and Narwhals DataFrames.
For Polars installation and preprocessing, see [DataFrame inputs](docs/dataframes.md).

```python
import numpy as np
from sksurv.util import Surv
from skcure.ggkm import GGKM

rng = np.random.default_rng(73)
X = rng.normal(size=(24, 2))
y = Surv.from_arrays(
    event=np.tile([True, False, True, True], 6),
    time=np.round(rng.uniform(0.4, 3.2, size=24), 3),
)

model = GGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=2).fit(X, y)

risk = model.predict(X)
c_index = model.score(X, y)
cure_probability = model.predict_cure_probability(X)
survival_functions = model.predict_survival_function(X)
survival_at_times = model.predict_survival(X, times=[0.0, 1.0, 3.0])
```

Higher risk scores indicate greater risk; `score` returns the concordance
index. Survival and cumulative-hazard functions follow scikit-survival's
`StepFunction` interface. Set `return_array=True` to obtain values on the
training grid, `unique_times_`.

See the [user guide](docs/README.md) for prediction meanings, fitting controls,
assumptions, and observed-data evaluation. A runnable
[evaluation example](examples/evaluation.py) and a
[simulation benchmark](benchmarks/simulation.py) are included.
See the [GBSG2 real-data walkthrough](docs/real_data.md) for preprocessing,
training-only model selection, and held-out evaluation, and the
[performance guide](docs/performance.md) for measured runtime/memory costs.

The example uses a small iteration budget for demonstration. For an analysis,
tune the solver settings and evaluate performance on held-out data.

## Support and contributing

Use [GitHub Issues](https://github.com/cowvin0/scikit-cure/issues) for bug reports
and questions. Contributions through pull requests are welcome.
See the [contributing guide](CONTRIBUTING.md) and [release notes](CHANGELOG.md).
For survival targets, metrics, and API conventions, see the
[scikit-survival documentation](https://scikit-survival.readthedocs.io/).

To install development dependencies and run the tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

## References

- Pölsterl, S. (2020). [scikit-survival: A Library for Time-to-Event Analysis
  Built on Top of scikit-learn](https://jmlr.org/papers/v21/20-729.html).
  *Journal of Machine Learning Research*, 21(212), 1–6.
- Medina-Olivares, V., Lessmann, S., and Klein, N. (2023).
  [The Deep Promotion Time Cure Model](https://arxiv.org/abs/2305.11575).
  *arXiv:2305.11575*.

Software citation metadata is provided in [CITATION.cff](CITATION.cff).
scikit-cure is licensed under [GPL-3.0-or-later](LICENSE).
