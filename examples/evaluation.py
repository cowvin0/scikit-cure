"""Evaluate observed survival outcomes using held-out data.

Run from a source checkout with ``python examples/evaluation.py``
after installing the package. The generated cure labels are simulation truth,
never inferred by treating censored real-world observations as cured.
"""

from dataclasses import asdict

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skcure.datasets import simulate_cure_survival
from skcure.linear_model import MixtureCure
from skcure.metrics import evaluate_survival


def main():
    train = simulate_cure_survival(500, followup=6, random_state=73)
    test = simulate_cure_survival(250, followup=5, random_state=74)
    model = make_pipeline(StandardScaler(), MixtureCure(tol=1e-6)).fit(train.X, train.y)
    times = np.linspace(.3, 4.5, 20)
    evaluation = evaluate_survival(model, test.X, test.y, y_train=train.y, times=times)
    print("Observed-data survival evaluation:", asdict(evaluation))

if __name__ == "__main__":
    main()
