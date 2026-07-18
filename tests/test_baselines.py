from hashwave.baselines import BaselineConfig, BeamMutationSearch, SimulatedAnnealingSearch, StandardGeneticProgramming
from hashwave.search import Task


def tiny_task():
    return Task.from_function("2*x+1", lambda x:2*x+1, train_xs=range(-3,4), validation_xs=(-5,5), test_xs=(-8,8))


def test_baselines_run_and_return_finite_metrics():
    cfg=BaselineConfig(evaluations=200,population_size=12,tournament_size=3,seed="unit")
    for cls in (StandardGeneticProgramming, BeamMutationSearch, SimulatedAnnealingSearch):
        result=cls(tiny_task(),cfg).run()
        assert result.proposals > 0
        assert result.test_mae is not None
        assert result.test_mae >= 0


def test_standard_gp_is_deterministic():
    cfg=BaselineConfig(evaluations=250,population_size=12,tournament_size=3,seed="same")
    a=StandardGeneticProgramming(tiny_task(),cfg).run()
    b=StandardGeneticProgramming(tiny_task(),cfg).run()
    assert a.best.source == b.best.source
    assert a.test_mae == b.test_mae
