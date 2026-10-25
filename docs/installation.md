# Installation

Use Python 3.10 or later. Core requirements are NumPy 2.0+, SciPy 1.13+,
scikit-learn 1.6+, scikit-survival 0.25+, and Narwhals 2.0.1+. Dependency resolvers select
compatible versions for your Python interpreter; recent scikit-survival
versions can require a newer Python. See [pyproject.toml](../pyproject.toml).

From a repository checkout:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install .
```

TensorFlow, DeepCure, and JAX are optional. Importing `skcure` does not load
them. GGKM, linear models, and boosting use NumPy/SciPy derivatives by default.

## Polars inputs

Narwhals is a core dependency; Polars is optional. To install Polars together
with scikit-cure from the checkout:

```bash
python -m pip install '.[polars]'
```

An existing compatible Polars installation is sufficient. See
[DataFrame inputs](dataframes.md) for fitting, preprocessing, and target
construction. The development extra includes Polars for integration tests.

## Neural model

From the repository root, install the neural dependencies and the tested
DeepCure revision together:

```bash
python -m pip install -r requirements-deep-learning.txt
```

The `deep-learning` extra supplies TensorFlow, pandas, and numba. The
requirements file additionally installs DeepCure at a fixed Git revision.
For an installed wheel, install that upstream revision explicitly:

```bash
python -m pip install 'skcure[deep-learning]'
python -m pip install 'deepcure @ git+https://github.com/vhmedina/deepcure.git@4cb1dd2ba0a603c9b93c8584620b1488df12621b'
```

The wheel command assumes a wheel is available to your package installer; it
is not a claim that a release has been published. Git and access to GitHub
are needed for the upstream dependency. Direct VCS dependencies are kept out
of wheel metadata so the core package can be distributed normally.

## Optional derivative backend

```bash
python -m pip install '.[jax]'
```

JAX is available only as an explicitly selected backend for the internal
gamma shape derivative helper (`backend="jax"`). Ordinary estimator fitting
continues to use the default SciPy implementation. The optional helper uses
a temporary precision context and does not change global JAX configuration.
