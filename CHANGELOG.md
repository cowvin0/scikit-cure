# Release notes

## 0.1.0a1 — unreleased alpha

- Use a consistent training memory layout in GGKM so kernel rounding from
  NumPy versus DataFrame inputs does not alter the optimizer's fitted result.
- Consolidate exponential helpers and handle expected tail overflow/underflow
  consistently, preserving NaNs, infinities, and representable tiny values.
- Support eager Polars and Narwhals DataFrame feature inputs across estimators,
  preserving feature-name checks and avoiding conversion through pandas. Share
  input normalization with standalone kernels and future model implementations.
- Extend scientific validation to genuine Poisson, negative-binomial, and
  binomial count-generating mechanisms, preserving existing simulation defaults.
- Add survival-aware scikit-learn estimator checks and reproducible runtime
  and process-memory benchmarks.
- Add a complete GBSG2 real-data analysis with training-only model selection,
  held-out evaluation, and explicit interpretation limits.
- Prepare the first alpha metadata and document the public API and release
  verification procedure. No package publication is performed by CI.

- Recover finite generalized-gamma latency line-search failures with one
  validated restart, addressing a Windows warm-refit failure. Keep optional
  JAX imports compatible with type checking when the extra is absent.
- Organize estimators in `skcure.ggkm`, `skcure.ensemble`,
  `skcure.linear_model`, and `skcure.deep_learning`.
- Unify gradient boosting cure families in `GradientBoostingCure` and add
  a scikit-survival-compatible `DeepPromotionTimeCure` wrapper.
- Stabilize generalized-gamma tails and gradients, latent count likelihoods,
  incidence updates, and neural losses at extreme parameter values.
- Add relative EM convergence tolerance, per-stage solver options, explicit
  termination reasons, inner optimizer diagnostics, and warnings at fit caps.
  All EM estimators retain `max_em_iter=1000` by default.
- Add a total boosting tree budget (default 1000), opt-in validation stopping
  with complete best-state restoration, and staged tree predictions.
- Add `predict_latent_mean` consistently across estimator families and
  document partial warm-start initialization reuse.
- Distinguish model-implied cure ROC/AUC diagnostics from observed-outcome
  evaluation; preserve old metric names as aliases.
- Add IPCW survival evaluation, ground-truth cure simulations, and a
  simulation benchmark.
- Use SciPy gamma shape derivatives by default. JAX becomes an optional,
  explicitly selected helper backend without import-time precision changes.
- Require NumPy 2.0+ and SciPy 1.13+. Keep optional neural dependencies and
  the pinned upstream DeepCure source installation separate from core metadata.
- Add user documentation, CI and distribution checks, citation metadata,
  and the GPL-3.0-or-later license.
