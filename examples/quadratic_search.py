from hashwave import HashEvolutionSearch, SearchConfig, Task


def target(x: int) -> int:
    return 3 * x * x + 2 * x + 1


task = Task.from_function(
    "quadratic",
    target,
    train_xs=range(-6, 7),
    validation_xs=(-10, -8, 8, 10),
    test_xs=(-20, -15, 15, 20),
)
result = HashEvolutionSearch(
    task,
    SearchConfig(seed="published-example", generations=50),
).run()
print(result.best.source)
print("test MAE:", result.test_mae)
