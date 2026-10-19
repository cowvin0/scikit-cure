# Original gradient boosting reference results

`boost_legacy.json` preserves results from the user's original
`skcure/models/gdbt_cure.py`, captured before refactoring that module. The source
SHA-256 and versions of NumPy, SciPy, scikit-learn, and scikit-survival appear in
the JSON. Capturing used an independent copy loaded with `importlib`, with the
original `pgamma_shape_derivative_vec` utility.

The original binomial estimator raises `NameError: name 'expit' is not defined`:
the module uses the sigmoid without importing it. The capture first verified
that failure, then injected only `scipy.special.expit` into the independent
module's globals so its intended algorithm could be recorded. No numerical
algorithm or optimizer was changed for the capture.

Inputs use `numpy.random.default_rng(73)`: 24 observations with two normal
features, times drawn uniformly from 0.4 to 3.2 and rounded to two decimals, and
events repeating `[True, False, True, True]`. The JSON contains the actual data,
prediction times, and all estimator and fit settings. Four cases cover binomial
counts (`K_bin=3`), Poisson counts, and fixed (`phi=0.7`) and estimated
negative-binomial dispersion. Each uses three depth-two trees per EM iteration,
three EM iterations, and a learning rate of 0.05.

Each case records learned parameters, likelihood history, predictions, cure and
survival probabilities, training predictors, ensemble weights, and individual
tree predictions. `fixed_gg_steps` additionally records three E-step/boosting
updates while holding the initial latency and dispersion parameters fixed.
These isolate preservation of the tree fitting algorithm from changes in the
numerically sensitive latency optimization.

The original free fits emitted 0, 68, 56, and 76 runtime warnings respectively
for binomial, Poisson, fixed NB, and estimated NB. Several warnings came from
generalized-gamma optimizer trial values. Their fitted latency exponents were
46.26, 81.56, 760.30, and 86.57; estimated NB dispersion reached its lower bound
of `1e-8`. These small, deliberately short fits record implementation behavior,
not convergence or statistical performance.

`test_boosting.py` checks frozen-latency boosting updates against these original
references, then separately verifies observed likelihoods, posterior latent
counts, survival and cure formulas, and both functional and latency derivatives
using independent expressions and finite differences. It also exercises the
estimator API and numerical extremes without treating the legacy overflow and
tail clipping as desired behavior.

## Measured comparison after refactoring

`boost_legacy_comparison.json` records the measured changes with the dependency
versions in the capture. It is an audit report rather than a set of tolerances
that accepts different fitted models. With latency and dispersion fixed, the
maximum absolute differences across the four cases are `3.85e-7` in predictors,
`8.35e-7` in ensemble weights, `6.67e-8` in tree predictions, and `7.28e-7` in
likelihood histories. The shared log-domain expressions perturb the numerical
finite differences in the existing L-BFGS line searches. The regression tests
therefore allow absolute errors of `1e-6` for predictors, weights, and histories,
and `1e-7` for individual tree predictions. The first full EM update's likelihood
also matches the legacy reference within `1e-6` for each case.

Later full-fit results differ because the generalized-gamma optimizer uses the
shared survival-tail calculation and rejects invalid trial values. That changes
the optimizer trajectory in these near-degenerate cases. The original clipped
tail and overflowing trial calculations are not preserved for exact parity.

| Case | Maximum predictor change | Maximum cure change | Maximum survival change | Final likelihood change |
| --- | ---: | ---: | ---: | ---: |
| Binomial | `9.35e-8` | `2.48e-8` | `0.00827` | `-0.0840` |
| Poisson | `0.000166` | `0.0000607` | `0.00156` | `-0.1043` |
| Fixed NB | `0.0000912` | `0.0000251` | `0.03195` | `-1.3051` |
| Estimated NB | `0.0000981` | `0.0000360` | `0.000767` | `-0.0715` |

The fixed-NB latency exponent changes most, from approximately `760.30` to
`75.92`. The refactor does not claim identical free-fit predictions or improved
likelihood in these short fits. In particular, its fixed-NB survival curves
differ by up to 3.2 percentage points. The tests verify the preserved boosting
algorithm, the shared mathematical objectives, and warning-free finite fits;
the report makes the remaining numerical differences reviewable.
