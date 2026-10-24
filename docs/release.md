# First alpha release

The source tree is prepared as `0.1.0a1`. This records an unreleased alpha;
changing the version does not publish a package or create a GitHub release.

## Public API scope

The first alpha focuses on the existing estimator families:

- `skcure.ggkm.GGKM`
- `skcure.ensemble.GradientBoostingCure`
- `skcure.linear_model.MixtureCure` and `PromotionTimeCure`
- `skcure.deep_learning.DeepPromotionTimeCure`
- `skcure.base.BaseCureEstimator`

The supported interface includes explicit scikit-learn constructor parameters,
`fit(X, y)` with a structured survival target, risk/cure/latent-mean/survival
prediction, and the documented fitted diagnostics. `skcure.metrics` and
`skcure.datasets` provide the documented evaluation and simulation helpers.
Modules and attributes whose names start with an underscore are internal.

Keep the estimator interface stable while validating this alpha. Additions
should preserve existing constructor and prediction semantics; any unavoidable
public change requires an explicit release note. Alpha status allows further
changes before a stable release and is not a promise of indefinite compatibility.
New estimator families and uncertainty estimation are outside this release scope.

## Verification

Run the ordinary regression suite and selected upstream estimator checks:

```bash
python -m pytest -q
python -m pytest -q tests/test_estimator_checks.py
python -m pyright
python ci/check_version.py
```

The compatibility suite runs applicable scikit-learn checks directly on public
estimators. Fitting checks use test-only subclasses that map generic numeric
targets to positive, fully observed survival targets; constructors, fitting,
predictions, and serialization still delegate to the real estimators. Separate
checks exercise censored structured targets through pipelines and grid search.
This selected suite does not claim that classifier or ordinary regression
checks apply unchanged to survival estimators.

Run reproducible scientific, scaling, and real-data exercises:

```bash
python benchmarks/simulation.py --quick --output /tmp/scientific-smoke.json
python benchmarks/performance.py --quick --output /tmp/performance-smoke.json
python examples/gbsg2_analysis.py --quick --output /tmp/gbsg2-smoke.json
```

Quick modes verify the workflows with bounded budgets. They do not establish
statistical accuracy or practical scalability. Use repeated full simulations
and measured scaling studies before making those claims; see the
[evaluation guide](evaluation.md) and [performance guide](performance.md).

Build and validate both distribution formats:

```bash
python -m build
python -m twine check dist/*
```

The release-check workflow also installs the wheel and source distribution in
fresh environments away from the checkout, and exercises imports, metadata,
fitting, and predictions without optional backends. CI covers the minimum
supported core dependencies, current dependencies, Linux/macOS/Windows,
optional backends, and typing without JAX. Its workflows build artifacts for
review and do not upload distributions to a package index.

Before publication, verify the new commit's CI, review the recorded benchmark
failures and model assumptions, and keep versions synchronized in
`pyproject.toml`, `skcure/__init__.py`, and `CITATION.cff`. Release metadata
should describe checks actually completed for that commit.

## Current interpretation limits

Censored observations do not reveal cure status. Cure fractions depend on
latency/count assumptions and the information in follow-up; predictive accuracy
on a finite horizon does not establish long-term cure calibration. Kernel
fitting uses a dense training Gram matrix, and optional neural fitting depends
on the pinned upstream DeepCure revision. See [models](models.md),
[fitting controls](fitting.md), and [installation](installation.md).
