# Quick start

## Symbolic regression demonstration

```bash
vhss demo --generations 40 --beam 24 --per-parent 64 --seed study-001 --json > demo.json
```

The output reports the recovered expression, train/validation/test error,
proposal count, unique evaluations, cache hits, hash attempts, and stop reason.

## Compare seed sources

```bash
vhss benchmark --json > benchmark.json
```

This compares the same evolutionary structure under SHA-256-derived and
conventional pseudo-random seeds, plus a random-program baseline.

## Python API

```python
from hashwave import HashEvolutionSearch, SearchConfig, Task

task = Task.from_function(
    "quadratic",
    lambda x: 3*x*x + 2*x + 1,
    train_xs=range(-6, 7),
    validation_xs=(-10, -8, 8, 10),
    test_xs=(-20, -15, 15, 20),
)
result = HashEvolutionSearch(task, SearchConfig(seed="example")).run()
print(result.best.source, result.test_mae)
```
