# GGKM EM extraction reference

`ggkm_em_reference.json` was captured from an independent `importlib` load of
the unmodified GGKM implementation immediately before its EM loop was moved
into `skcure/models/_em.py`. The original module's SHA-256 and the Python,
NumPy, SciPy, scikit-learn, and scikit-survival versions are recorded in the
JSON's `provenance` object. The original generalized-gamma and latent-count
helpers were used unchanged during both capture and comparison.

The input is deterministic: `numpy.random.default_rng(73)` supplies 24 rows
with two normal features and observation times drawn uniformly from 0.4 to
3.2, rounded to two decimal places. Event indicators repeat
`[True, False, True, True]`. Most fits use `gamma=0.3`, `lambda_reg=0.1`, and
three EM iterations. The test source specifies every scenario and the fixed
prediction inputs and times.

`test_em_ggkm.py` verifies 24 successful scenarios containing 32 fit calls:

- Poisson, negative binomial with estimated and fixed dispersion, Bernoulli,
  and binomial with three trials.
- RBF, linear, polynomial, Cauchy, matrix-valued callable, and precomputed
  kernels.
- Nondefault latency initialization, early convergence, iteration limits,
  and distribution-specific default iteration limits.
- Cold and warm refits, including changes to the latent family, fixed
  dispersion, and binomial trial count.

The reference preserves constructor parameters, all learned and working
parameters, latent configuration, the training design and kernel matrix,
likelihood histories, convergence flags, iteration counts, and observed times.
It also preserves risk and cure predictions, Poisson intensity predictions,
continuous survival at unordered and repeated times including zero, survival
and cumulative-hazard arrays and `StepFunction` outputs, and concordance scores.
The actual SciPy optimizer calls record their objective names, starts,
arguments, bounds, method, derivative settings, final parameters, objective
values, gradients, status, and evaluation counts.

Another 21 scenarios inject a failure at initial likelihood evaluation, the
E-step, predictor optimization, latency optimization, dispersion update,
post-M-step likelihood evaluation, and finalization for each latent family.
Their exceptions and partial state after a failed warm refit must also match.
Together the successful and failed scenarios record 214 optimizer calls.

Every captured float is compared by its IEEE-754 float64 bit pattern using
`numpy.testing.assert_array_equal`; there are no numerical tolerances. JSON
round trips preserve the captured finite float64 values, including signed
zeros. This establishes exact preservation of the recorded optimization paths
in the recorded environment. Different dependency versions or numerical
platforms can take different optimization paths and are outside that claim.

Run the comparisons with:

```bash
python -m pytest tests/test_em_ggkm.py -q
```
