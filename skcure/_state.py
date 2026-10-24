import numpy as np

from dataclasses import dataclass, field


@dataclass
class GeneralizedGammaParameters:
    """Mutable generalized gamma parameters during EM fitting."""

    a: float
    d: float
    p: float


@dataclass
class GGKMFitState:
    """Working state for one fit; existence alone does not imply a fitted model."""

    beta0: float | None = None
    alpha: np.ndarray | None = None
    K: np.ndarray | None = None
    X_train: np.ndarray | None = None
    loglik_history: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0
    final_loglik: float | None = None
