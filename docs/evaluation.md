# Evaluation and simulation

## Observed-data performance

Use held-out outcomes to evaluate survival discrimination and probability
accuracy. `evaluate_survival` combines scikit-survival's IPCW concordance,
time-dependent AUC, and integrated Brier score:

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skcure.datasets import simulate_cure_survival
from skcure.linear_model import MixtureCure
from skcure.metrics import evaluate_survival

train = simulate_cure_survival(500, followup=6, random_state=73)
test = simulate_cure_survival(250, followup=5, random_state=74)
model = make_pipeline(StandardScaler(), MixtureCure(tol=1e-6))
model.fit(train.X, train.y)
result = evaluate_survival(
    model, test.X, test.y, y_train=train.y,
    times=np.linspace(0.3, 4.5, 20),
)
print(result.concordance_ipcw, result.mean_auc, result.integrated_brier_score)
```

The training outcomes estimate the censoring distribution. Evaluation times
must be strictly increasing, within test follow-up, and supported by positive
training censoring survival; unsupported intervals raise informative errors.
At least two times are required, each with an observed test event at or before
it. Training follow-up must also cover all test follow-up and extend beyond
every test event time. The helper keeps all test observations; `tau` truncates
concordance only, without changing the AUC or Brier evaluation population.
Do not substitute test outcomes for `y_train`. The Brier score measures
overall probability error, which includes calibration and discrimination;
it is not a cure-status accuracy score.

The IPCW estimates use a marginal censoring distribution estimated from the
training population. They assume censoring is independent of the event time
and covariates; covariate-dependent censoring needs an appropriate conditional
weighting method rather than this helper.

`model_implied_cure_roc` and `model_implied_cure_auc` use predicted cure
probabilities as both scores and class weights. They measure separation
implied by those predictions and can be large even for wrong predictions.
They do not validate cure discrimination or calibration against outcomes.
The existing `cure_tpr_fpr` and `auc_cure` names remain compatibility aliases.

## Simulations and benchmark

`simulate_cure_survival` generates a logistic mixture-cure cohort with known
cure probabilities, individual cure labels, latent event times, and analytic
population survival. Susceptible latency can be Weibull or lognormal.
Exponential censoring is independent, with optional administrative follow-up.
`censoring_rate` is an exponential rate, not a target censoring percentage.

```python
truth = test.survival_probability([1, 3, 5])
cure_error = np.mean(np.abs(
    model[-1].predict_cure_probability(model[:-1].transform(test.X))
    - test.cure_probability
))
```

Run the independent benchmark from the repository root after installation:

```bash
python benchmarks/simulation.py --quick --output /tmp/cure-results.json
python benchmarks/simulation.py --repetitions 10 --output cure-results.json
python benchmarks/simulation.py --quick --models mixture promotion_time ggkm gradient_boosting
```

The full study varies sample size, censoring rate, follow-up, and latency
specification. It compares linear mixture and promotion-time models, retaining
every failed or nonconverged attempt alongside cure bias/RMSE, survival RMSE,
and marginal survival calibration error. Aggregate accuracy summaries use
converged fits and report their number. These simulations are controlled
experiments, not evidence of accuracy for every dataset or model family.
The optional `--models` selection includes GGKM and boosting, using bounded
training budgets. Every record saves its estimator settings, so budget-related
nonconvergence remains distinguishable from a raised fitting error. These
defaults are benchmark configurations rather than tuned model comparisons;
compare hyperparameters selected independently before drawing ranking claims.

A complete runnable example is
[examples/evaluation.py](../examples/evaluation.py).
