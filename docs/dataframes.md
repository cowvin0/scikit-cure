# DataFrame inputs

All estimators accept eager Polars and pandas DataFrames as `X`, alongside
NumPy arrays. They also accept DataFrames wrapped by Narwhals. Install Polars
from a repository checkout with:

```bash
python -m pip install '.[polars]'
```

Features must be numeric and finite. Encode categorical columns and impute
missing values before fitting, usually in a scikit-learn pipeline. Models
record string column names in `feature_names_in_` and check both names and
order during prediction. A model fitted on pandas can predict from Polars,
and vice versa, when columns match.

## Fit and predict

```python
import polars as pl
from skcure.datasets import simulate_cure_survival
from skcure.linear_model import MixtureCure

cohort = simulate_cure_survival(120, n_features=2, random_state=73)
X = pl.DataFrame(cohort.X, schema=["age", "marker"], orient="row")
y = cohort.y

model = MixtureCure().fit(X, y)
risk = model.predict(X)
cure_probability = model.predict_cure_probability(X)
survival = model.predict_survival(X, times=[0.0, 1.0, 3.0])
```

The example's feature names are illustrative; its data are simulated.
Prediction outputs retain the existing NumPy array and `StepFunction`
formats. Input DataFrames are not modified.

Survival targets retain the scikit-survival structured-array format. For
outcomes stored in Polars columns, construct `y` explicitly:

```python
from sksurv.util import Surv

# outcomes is a Polars DataFrame with boolean event and numeric time columns.
y = Surv.from_arrays(
    event=outcomes["event"].to_numpy(),
    time=outcomes["time"].to_numpy(),
)
```

`True` indicates an observed event. Match outcome rows to feature rows;
DataFrame indices do not align them automatically.

## Preprocessing and Narwhals

Scikit-learn transformers can pass Polars output directly to the estimator:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(
    StandardScaler().set_output(transform="polars"),
    MixtureCure(),
)
pipeline.fit(X, cohort.y)
risk = pipeline.predict(X)
```

A Narwhals wrapper is also accepted directly:

```python
import narwhals.stable.v2 as nw

cure_probability = model.predict_cure_probability(nw.from_native(X))
```

The standalone `exponential_kernel`, `cauchy_kernel`, and `wavelet_kernel`
functions in `skcure.utils.kernels` accept the same eager feature inputs and
return NumPy kernel matrices. For two input matrices, put their columns in
the same feature order; kernels compare columns by position.

```python
from skcure.utils.kernels import cauchy_kernel

kernel_matrix = cauchy_kernel(nw.from_native(X), gamma=0.3)
```

The shared validation boundary uses Narwhals to recognize DataFrames and
unwrap their native backend. Scikit-learn validates column metadata before
converting features to the NumPy arrays used by the numerical solvers. This
avoids a Polars-to-pandas conversion and reuses existing buffers when the
backend and dtype allow it. Casting, heterogeneous columns, or a backend's
memory layout can still require a copy; fitting is not universally zero-copy.

Lazy frames are not evaluated implicitly. Select and filter upstream, then
call `.collect()` explicitly before `fit` or any prediction method:

```python
X = lazy_features.collect()
model.fit(X, cohort.y)
```

Collection must produce the same feature columns and row order as the target.
