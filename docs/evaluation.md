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

`simulate_cure_survival` generates known cure probabilities, individual cure
labels, latent counts, latent event times, and analytic population survival.
It supports genuine binomial, Poisson, and negative-binomial counts. The default
`latent_distribution="binomial", K_bin=1` retains the original logistic
mixture-cure generator, including its seeded data stream.

For a subject with count `N`, the event is the first of `N` independent latent
event times; `N=0` gives an infinite event time. The population survival is the
count probability-generating function at the latent survival probability.
Thus the known survival is derived from the generating law, not fitted from
the simulated outcomes. See the formulas in [models](models.md).

The incidence predictor is linear under each family's link. Positive
`cure_coefficients` increase cure probability and have the opposite sign to
`incidence_coefficients`. The intercept is calibrated to a requested population
`cure_fraction`; the realized cohort's fraction remains random. Binomial
counts use `K_bin` trials; negative-binomial counts have conditional variance
`mean + phi * mean**2`, with `phi=0` giving the Poisson limit.

Each latent cause has unit-scale Weibull or lognormal latency. The latter
provides a departure from the finite generalized-gamma latency specification.
Exponential censoring is independent, with optional administrative follow-up.
`censoring_rate` is an exponential rate, not a target censoring percentage.

```python
truth = test.survival_probability([1, 3, 5])
cure_error = np.mean(np.abs(
    model[-1].predict_cure_probability(model[:-1].transform(test.X))
    - test.cure_probability
))

nb_truth = simulate_cure_survival(
    500, latent_distribution="negative_binomial", phi=0.5,
    cure_fraction=0.3, followup=5, random_state=76,
)
# nb_truth.latent_count, latent_mean, and incidence_predictor are simulation truth.
```

Run the independent benchmark from the repository root after installation:

```bash
python benchmarks/simulation.py --quick --output /tmp/cure-results.json
python benchmarks/simulation.py --repetitions 10 --output cure-results.json
python benchmarks/simulation.py --quick --models mixture promotion_time ggkm gradient_boosting
python benchmarks/simulation.py --sample-sizes 300 --repetitions 3 \
  --censoring-rates 0.1 --followups 5 --latencies weibull \
  --latent-distributions binomial poisson negative_binomial \
  --output scientific-study.json
```

The full study varies sample size, censoring rate, follow-up, latent count
family, cure fraction, and latency specification. All estimators in a repetition
receive the same independent training and test cohorts. GGKM and boosting fit
the generating count family; the linear pair supplies matched and deliberately
misspecified comparisons. A matching count/latency family alone does not prove
that a finite tree or kernel fit exactly represents the generating incidence
function, so the report separates these claims.

Each attempt retains seeds, estimator settings, convergence state, warnings,
fit errors, and observed-metric errors. Measures include cure bias/RMSE,
survival RMSE, marginal survival calibration error, and supported IPCW/AUC/Brier
evaluation. The marginal calibration measure compares mean predicted survival
with mean true survival and can conceal subgroup errors; it is not a complete
calibration assessment. Unsupported observed-data evaluation is reported without
dropping test subjects; known-truth errors remain available.

Primary accuracy summaries use converged fits, report their actual denominators,
and retain an `all_fitted` summary that also includes completed nonconverged
fits. Always inspect failure and nonconvergence rates: selecting only successful
fits can make a method appear more accurate. Reported Monte Carlo standard
errors are the standard deviation across available independent repetitions
divided by the square root of their number. They describe precision of the
simulation average, not confidence intervals for individual predictions.

An initial bounded study using the command above completed all 18 linear-model
fits without fit or observed-metric errors. At `n=300`, three repetitions, and
the specified Weibull/censoring/follow-up settings, matched Bernoulli mixture
fits had mean cure/survival RMSE of about 0.037/0.035; matched Poisson promotion
fits had about 0.024/0.024. These small development studies exercise the workflow;
they do not establish accuracy across datasets, tuning choices, or model families.
The [reference records](../benchmarks/results/scientific-reference.json) retain
the generating parameters, repetitions, diagnostics, and Monte Carlo errors.
The optional `--models` selection includes bounded GGKM and boosting runs;
budget-related nonconvergence is reported explicitly. Benchmark configurations
are not tuned model comparisons.

A complete runnable example is
[examples/evaluation.py](../examples/evaluation.py).
For observed clinical data, see the [GBSG2 walkthrough](real_data.md).
