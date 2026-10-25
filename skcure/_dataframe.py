"""Shared DataFrame normalization before scikit-learn input validation.

Keep native eager frames intact so callers can validate feature names before
converting data to numerical arrays. Estimators, transformers, and standalone
utilities share this boundary without depending on a cure-model base class.
"""

from typing import Any

import narwhals.stable.v2 as nw
import numpy as np


def ensure_eager_dataframe(X: Any) -> Any:
    """Unwrap eager frames and reject lazy input without copying feature data.

    Array-like inputs pass through to the caller's usual validation. This
    helper neither checks numeric values nor casts dtypes; callers choose
    their own dtype, memory layout, sparse support, and feature-name checks.
    """
    if X is None or isinstance(X, np.ndarray):
        return X

    native = nw.to_native(X, pass_through=True)
    if nw.dependencies.is_into_dataframe(native):
        return native
    frame = nw.from_native(native, pass_through=True)
    if isinstance(frame, nw.LazyFrame):
        raise TypeError(
            "Lazy DataFrames are not supported. Call .collect() before "
            "passing data to scikit-cure."
        )
    return native
