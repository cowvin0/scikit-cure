# Contributing

Bug reports and pull requests are welcome at
[cowvin0/scikit-cure](https://github.com/cowvin0/scikit-cure).
Include a small reproducible example, package versions, and the traceback
when reporting a problem. For numerical issues, include the model settings
and convergence diagnostics.

Use a virtual environment named `.venv` and install development tools:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m pyright
python -m build
python -m twine check dist/*
```

Tests using real neural dependencies skip when those dependencies are absent.
Install `requirements-deep-learning.txt` to run them. JAX derivative comparison
tests similarly require `.[jax]`; core imports and tests must work without
these extras. The CI matrix checks the minimum core versions, current
dependencies, multiple operating systems, optional backends, and installable
wheel/source artifacts. A manual release-check workflow builds and validates
artifacts without publishing them.

`tests/test_estimator_checks.py` runs selected upstream scikit-learn checks
with survival-aware target adapters, plus censored-target pipeline checks.
Keep expected short-fit convergence warnings scoped to those fits; unexpected
warnings should remain visible. The CI validation job also runs quick scientific,
performance, and real-data workflows and stores their JSON reports.

Keep constructor parameters cloneable and explicit, use the scikit-survival
target format, and preserve public prediction semantics. Add tests for
mathematical invariants and meaningful failure cases when changing fitting
or numerical routines. Frozen model fixtures in `tests/data` guard numerical
compatibility; changes to them need an explanation of the mathematical change.

## Adding models and DataFrame support

New cure estimators should inherit `BaseCureEstimator` and use
`_validate_fit_data(X, y)` for fitting and `_validate_predict_data(X)` for
prediction. These methods supply the shared DataFrame handling, feature-name
checks, and numeric validation. Validate each prediction input once, then pass
the validated array to internal calculations.

Utilities and transformers that do not inherit the estimator base can use
`ensure_eager_dataframe` from the private `skcure._dataframe` module before
their usual scikit-learn validation. This helper unwraps Narwhals frames and
rejects lazy inputs without collecting them. Preserve the native DataFrame
until feature names have been checked; keep dtype, sparse support, and other
numeric requirements at the validation call site. Avoid backend-specific
branches in numerical code and imports from scikit-survival's private modules.

Add new estimators to the shared fixture in `tests/test_dataframe_inputs.py`
so they receive the same NumPy, pandas, Polars, and Narwhals compatibility
checks. Extend those checks for any additional input contract.

Update the user guide and `CHANGELOG.md` for public behavior changes. Before
a release, set matching versions in `pyproject.toml`, `skcure/__init__.py`,
and `CITATION.cff`, and run `python ci/check_version.py`.
See [alpha release scope and verification](docs/release.md) before preparing
distribution artifacts.

The project is licensed under GPL-3.0-or-later; see [LICENSE](LICENSE).
