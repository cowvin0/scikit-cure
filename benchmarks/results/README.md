# Development reference results

These JSON files record small, actually executed studies from the `0.1.0a1`
development tree. They provide provenance for the numerical examples in the
user guide, not universal accuracy or scalability claims.

- `scientific-reference.json`: three repetitions at n=300 under each count
  family, comparing the two linear models with known generating truth.
- `performance-reference.json`: two repetitions at n=100/250/500 for all four
  core estimators with fixed training budgets and one numerical thread.

Each file includes its complete configuration, environment, summaries, and all
attempts, including capped fits. The absolute local Python executable path in
the performance record is replaced by its basename and marked as redacted;
measurements and other fields are retained. Timing and RSS values vary between
machines and executions. Use the commands in [evaluation](../../docs/evaluation.md)
and [performance](../../docs/performance.md) to generate new reports.
