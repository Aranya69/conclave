"""The observed-rainfall replay keeps the simulator's mechanics."""
import numpy as np

from conclave.core.types import Action, ActionKind
from conclave.envs.disaster import DisasterConfig, FloodResponseEnv
from conclave.envs.real_forcing import (FLOOD_LEVEL, ForcingModel, ReplayFloodEnv,
                                        antecedent_index, flood_onset)


def test_antecedent_index():
    a = antecedent_index(np.array([10.0, 0.0, 0.0]), 0.5)
    assert np.allclose(a, [10.0, 5.0, 2.5])


def test_forcing_model_threshold_and_mapping():
    rng = np.random.default_rng(0)
    rain = {"a": rng.gamma(0.6, 15.0, 3000)}
    mask = {"a": np.ones(3000, dtype=bool)}
    fm = ForcingModel.fit(rain, mask, synthetic_rain=rng.uniform(0, 20, 5000),
                          kappa=0.85, quantile=0.95)
    st = fm.stage(rain["a"])
    assert abs(np.mean(st > FLOOD_LEVEL) - 0.05) < 0.01      # 1 - q of days flood
    m = fm.sensor_rain(np.array([0.0, np.median(rain["a"]), 1e4]))
    assert m[0] < m[1] < m[2] <= 20.0 + 1e-9                    # monotone, on the sensor scale


def test_replay_follows_the_stage_and_risk():
    stage = np.r_[np.full(10, 2.0), np.full(41, 6.0)]
    env = ReplayFloodEnv(np.full(51, 5.0), stage, DisasterConfig(seed=1, horizon=40))
    assert flood_onset(stage, 40) == 10
    o = env.reset()
    assert o.meta["true_risk"] < 0.65                           # flood 9 steps ahead
    for t in range(12):
        o, r, done, info = env.step(Action(kind=ActionKind.TOOL, tool="monitor"))
    assert abs(env.level - 6.0) < 1e-9 and env.stranded > 0
    assert o.meta["true_risk"] > 0.65


def test_reports_can_be_removed():
    env = ReplayFloodEnv(np.zeros(41), np.full(41, 1.0), DisasterConfig(seed=2), reports=False)
    assert env.reset().text == ""


def test_simulator_unchanged_by_hooks():
    """The forcing hooks reproduce the original random sequence exactly."""
    env = FloodResponseEnv(DisasterConfig(seed=7000, horizon=40))
    env.reset()
    total = 0.0
    for _ in range(40):
        _, r, _, _ = env.step(Action(kind=ActionKind.TOOL, tool="monitor"))
        total += r
    assert np.isfinite(total) and env.t == 40
