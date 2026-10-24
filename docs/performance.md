# Performance and practical limits

Run the benchmark from the repository root after installing the package:

```bash
python benchmarks/performance.py --quick --output /tmp/performance.json
python benchmarks/performance.py --sample-sizes 100 250 500 1000 \
  --repetitions 3 --models mixture promotion_time ggkm gradient_boosting \
  --max-em-iter 20 --max-total-estimators 100 \
  --thread-limit 1 --timeout 300 --output performance-results.json
```

Each model/size/repetition runs in a fresh subprocess. The report stores fit
and prediction time, native peak process resident memory, model settings,
seeds, dependency/platform versions, fitted iteration/tree counts, convergence,
warnings, and failures or timeouts. Prediction measures risk, cure probability,
and survival for a fixed evaluation sample and time grid. Worker startup and
imports are excluded from fit time; they contribute to the process memory peak.

Linux/macOS use `getrusage`; Windows uses `GetProcessMemoryInfo`. These measure
the operating system's process high-water memory, including native NumPy arrays,
the interpreter, imported libraries, data, and estimator. The recorded baseline
is the high-water mark after imports/data preparation and before fitting.
Subtracting it gives a high-water increment, not memory allocated exclusively
by the estimator. Unsupported memory measurement is reported explicitly.

Timing summaries include all completed fits, including capped or nonconverged
ones, with their counts. Failed workers remain in attempts and failure rates.
Do not compare fit times without also comparing budgets and stopping reasons.
The benchmark is a workload measurement and does not rank statistical accuracy.

## Dense-kernel cost

GGKM retains a dense `n_samples × n_samples` float64 training Gram matrix.
The benchmark records its actual retained bytes and the estimate for one matrix:

| Training subjects | One float64 Gram matrix |
| ---: | ---: |
| 1,000 | 7.6 MiB |
| 5,000 | 190.7 MiB |
| 10,000 | 762.9 MiB |
| 20,000 | 3.0 GiB |

Fitting requires additional arrays and optimizer/kernel workspaces. Prediction
also constructs similarities to training subjects. These numbers are lower
bounds on matrix storage and do not specify a safe maximum dataset size.
Measure on the intended hardware before selecting GGKM for a large cohort.

## Initial measured workload

A development run on Linux x86_64, Python 3.12.12, NumPy 2.5.4, SciPy 1.18.1,
scikit-learn 1.9.1, and scikit-survival 0.28.0 used one numerical thread, five
features, two repetitions, at most 10 EM iterations, and a 30-tree boosting cap:

```bash
python benchmarks/performance.py --sample-sizes 100 250 500 --repetitions 2 \
  --max-em-iter 10 --max-total-estimators 30 --thread-limit 1 \
  --timeout 60 --output performance-scaling.json
```

| Estimator | Mean fit seconds, n=100 / 250 / 500 | Mean peak process MiB, n=100 / 250 / 500 |
| --- | --- | --- |
| MixtureCure | 0.141 / 0.138 / 0.184 | 154.8 / 154.9 / 154.9 |
| PromotionTimeCure | 0.217 / 0.287 / 0.362 | 154.8 / 155.1 / 154.9 |
| GGKM | 1.047 / 3.010 / 4.888 | 155.2 / 156.8 / 161.1 |
| GradientBoostingCure | 0.218 / 0.230 / 0.291 | 154.8 / 155.1 / 155.3 |

All 24 attempts completed without failure or timeout. All six mixture fits
converged; promotion and GGKM reached the 10-iteration limit, and boosting
reached 30 trees. These results compare these particular bounded workloads,
not time to convergence or equivalent model quality. The large common memory
baseline explains why small matrices barely change process peaks. Runtime,
threading, allocation behavior, and operating-system measurements vary between
machines; use the generated JSON for the actual environment and fit diagnostics.
The [reference records](../benchmarks/results/performance-reference.json) retain
the reported attempts and configuration; the local interpreter path is redacted.
