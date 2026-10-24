# Fitting controls and diagnostics

GGKM, linear models, and boosting default to `max_em_iter=1000`. A finite
iteration budget can stop before convergence. A cap emits scikit-learn's
`ConvergenceWarning`; inspect the fitted state before using predictions for
an analysis:

```python
from skcure.linear_model import PromotionTimeCure

model = PromotionTimeCure(
    max_em_iter=1000,
    tol=1e-6,
    em_rtol=1e-8,
    solver_options={
        "incidence": {"maxiter": 500, "gtol": 1e-7},
        "latency": {"maxiter": 500, "ftol": 1e-10},
        "dispersion": {"maxiter": 200},
    },
)
# model.fit(X, y)
# print(model.converged_, model.termination_reason_, model.n_iter_)
# print(model.optimizer_results_)
```

An EM step converges when its finite, nonnegative likelihood improvement
is below `tol` or the enabled relative threshold
`em_rtol * max(abs(old_loglik), abs(new_loglik))`. The default `em_rtol=0`
disables the relative criterion. A decrease never establishes convergence.
The likelihood can include a penalty, depending on the model.

`solver_options` controls L-BFGS-B separately for incidence, latency, and
negative-binomial dispersion; unused stages have no effect. Supported
options are `ftol`, `gtol`, `maxiter`, `maxfun`, `maxls`, and `maxcor`.
The constructor mapping is retained for scikit-learn cloning and copied
before use. Invalid stages, options, and values raise `ValueError`.

`optimizer_results_` retains compact records for attempted inner solves:
stage, EM iteration, success, status, message, objective, and available
iteration/evaluation counts. An unsuccessful inner solve can sometimes
supply an acceptable update; inspect these records as well as the outer
convergence flag. `loglik_history_` records completed EM iterations.
The generalized-gamma latency solver can restart once after an abnormal
line-search termination at a finite, valid point. Both attempts are recorded;
the returned update still requires successful optimization and a finite,
non-worsening objective. Explicit solver options are retained on the retry.
`termination_reason_` is `converged`, `max_em_iter`, or `failed` for GGKM
and linear models; a raised exception can leave partial diagnostics.

## Boosting budgets and validation

```python
from skcure.ensemble import GradientBoostingCure

model = GradientBoostingCure(
    n_estimators=10,
    max_total_estimators=300,
    n_iter_no_change=5,
    validation_fraction=0.2,
    validation_tol=1e-4,
    random_state=73,
)
```

`n_estimators` bounds trees added per EM iteration. `max_total_estimators`
bounds the whole fit and defaults to 1000; set it to `None` to remove that
budget. Reaching the budget before convergence emits a warning and records
`termination_reason_="max_total_estimators"`.

Validation stopping is opt-in: `n_iter_no_change=None` disables it. When
enabled, a held-out fraction is excluded from all M-step fitting and scores
complete EM states using mean observed-data log likelihood. After the
patience limit, fitting records `validation_early_stopping` and restores
the best trees, latency, and dispersion. This can select the initial
zero-tree state. Validation stopping is model selection and does not imply
EM convergence.

`validation_loglik_history_` and `loglik_history_` retain attempted
iterations; `best_iteration_` identifies the restored checkpoint.
`final_loglik_` describes the returned fitted state. History length can
therefore exceed the selected iteration count.
The convergence flag describes the attempted training run; an earlier
checkpoint selected by validation need not itself meet the EM criterion.

`staged_predict_eta`, `staged_predict`, and `staged_predict_survival` yield
predictions after each retained tree. Survival stages use the final fitted
latency and dispersion, so they describe tree prefixes rather than full
historical EM states. There are no stages if the selected model has zero
trees; otherwise the final stage equals the corresponding final prediction.

## Repeated fitting

The EM models' `warm_start=True` reuses fitted latency and, where relevant,
dispersion as initialization. The incidence predictor is fitted anew;
boosting does not append to trees from the previous fit. This is partial
initialization reuse, so do not treat repeated fitting as continuing a
previous optimization trajectory. Use cloned estimators for cross-validation.

The neural wrapper has its own epoch, batch, optimizer, and validation
settings, and exposes training `history_`; the EM-specific controls above
do not apply to it.
