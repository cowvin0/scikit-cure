# Models and predictions

```python
from skcure.ggkm import GGKM
from skcure.ensemble import GradientBoostingCure
from skcure.linear_model import MixtureCure, PromotionTimeCure
from skcure.deep_learning import DeepPromotionTimeCure
```

All estimators accept `fit(X, y)` with the scikit-survival structured target:
the first field is a Boolean event indicator and the second is the observed
time. `True` means the event was observed. Training times must be positive
and finite and at least one event must be observed. Use a scikit-learn
pipeline to fit transformations within each training fold.

## Shared prediction interface

| Method | Meaning |
| --- | --- |
| `predict(X)` | Incidence predictor, with larger values indicating greater event risk. |
| `predict_eta(X)` | Incidence predictor on the model's link scale. |
| `predict_latent_mean(X)` | Expected number of latent event causes conditional on covariates. |
| `predict_cure_probability(X)` | Limiting population survival probability. |
| `predict_survival(X, times)` | Population survival on an explicit time grid. |
| `predict_survival_function(X)` | Survival as scikit-survival `StepFunction` objects. |
| `predict_cumulative_hazard_function(X)` | Cumulative hazard as `StepFunction` objects. |
| `score(X, y)` | Harrell's concordance index, using the incidence predictor. |

`predict` returns a ranking score, not a probability or a time. Shared
latency parameters mean that larger incidence predictors imply lower
population survival. `predict_theta` remains available for compatibility;
use `predict_latent_mean` when comparing latent families, particularly
binomial models where the old theta parameter is a success probability.

The function methods accept `return_array=True` for predictions on the
training grid, `unique_times_`. Their step functions have finite domains.
Use `predict_survival` to evaluate the underlying parametric model beyond
that grid, subject to the extrapolation assumptions below.

## Latent count families

GGKM and boosting accept `latent_distribution="poisson"`,
`"negative_binomial"`, or `"binomial"`. Let `eta` denote the incidence
predictor and `F0(t)` the generalized-gamma latent-time CDF. The population
survival is the probability-generating function of the latent count,
evaluated at `1 - F0(t)`:

| Family | Latent mean | Population survival | Cure probability |
| --- | --- | --- | --- |
| Poisson | `theta = exp(eta)` | `exp(-theta * F0(t))` | `exp(-theta)` |
| Negative binomial | `theta = exp(eta)` | `(1 + phi * theta * F0(t))**(-1/phi)` | `(1 + phi * theta)**(-1/phi)` |
| Binomial | `K * p`, `p = expit(eta)` | `(1 - p * F0(t))**K` | `(1 - p)**K` |

The negative-binomial family approaches Poisson as dispersion `phi` tends
to zero. `MixtureCure` uses the binomial family with `K=1` and a linear
logistic susceptibility predictor. `PromotionTimeCure` uses Poisson counts
and a linear log-mean predictor. GGKM uses a kernel predictor; boosting uses
regression trees. These models fit a common generalized-gamma latent-time
distribution rather than a separate baseline for each person.

`DeepPromotionTimeCure` wraps the neural promotion-time model of
[Medina-Olivares et al.](https://arxiv.org/abs/2305.11575). It uses Poisson
counts with `theta=exp(eta)` for `parametrization=1` or
`theta=log(1+exp(eta))` for `parametrization=2`, and a Weibull or
piecewise-exponential latent-time distribution. Both links are monotone.

## Statistical assumptions

Censoring does not identify cure status: a censored person can still
experience an event later. Cure probabilities depend on the chosen count
and latency distributions and on sufficient follow-up. Short follow-up can
make a large cure fraction difficult to distinguish from a long latency
tail. Predictions beyond observed follow-up are model-based extrapolations.

Likelihood fitting and the observed-data evaluation helpers assume
independent censoring under their respective model and weighting schemes.
Assess sensitivity to follow-up, latency specification, and regularization;
do not interpret a survival plateau alone as proof that individual subjects
are cured.
