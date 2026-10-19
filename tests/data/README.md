# Original model reference results

`legacy_models.json` contains results captured from the original
`skcure/models/gg_primal.py` and `skcure/models/baseline_model.py` before their
estimator API conversion. The capture loaded independent copies of both original
modules with `importlib`; the original baseline models imported the original
primal classes. Source SHA-256 hashes and dependency versions are in the JSON.

Inputs use `numpy.random.default_rng(73)`: 24 observations with two normal
features, times drawn uniformly from 0.4 to 3.2 and rounded to two decimals, and
event indicators repeating `[True, False, True, True]`. The JSON records the
actual inputs, prediction times, constructor parameters, and solver settings.

Nine scenarios cover all six estimators, fixed and estimated negative-binomial
dispersion, and mixture/promotion models with and without intercepts. Each fit
uses three EM iterations and `tol=1e-6`. The promotion model without an intercept
uses initial `a=d=p=1`; the original solver failed on this data with initial
`a=1.2, d=1.3, p=1.1` at the second iteration. Spline estimators use their original
default latency/dispersion initializations and record the transformed design.

`test_other_models.py` checks the mixture and promotion model cases for
predictions, cure/survival probabilities, fitted parameters, coefficients,
likelihood histories, iteration counts, and convergence flags. The paper-only
primal and spline cases are checked by
`ggkm_paper/tests/test_paper_other_models.py` against an unchanged copy of this
fixture. This verifies the original optimization paths for the recorded cases;
the fixtures are not convergence or statistical-performance benchmarks.
