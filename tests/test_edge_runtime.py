"""The NumPy-only edge runtime must reproduce the scikit-learn agent exactly."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_runtime():
    spec = importlib.util.spec_from_file_location(
        "conclave_edge_runtime", ROOT / "conclave" / "edge" / "runtime.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    return rt


def test_murmurhash_matches_sklearn():
    from sklearn.utils import murmurhash3_32
    rt = _load_runtime()
    for s in ["", "a", "ab", "abc", "abcd", "flood rising", "café naïve", "x" * 37]:
        assert rt.murmurhash3_32(s.encode("utf-8"), 0) == murmurhash3_32(s, seed=0)


def test_runtime_reproduces_agent(tmp_path):
    from conclave.brain.ensemble import EnsembleBrain
    from conclave.brain.featurize import (BudgetedTextFeaturizer, FusedFeaturizer,
                                          TabularFeaturizer)
    from conclave.brain.uncertainty import UncertaintyGate
    from experiments.exp3_sequential import collect

    obs, lab = collect(25, seed0=500)
    fz = FusedFeaturizer(text=BudgetedTextFeaturizer(n_word_features=2 ** 10,
                                                     n_char_features=2 ** 10),
                         tabular=TabularFeaturizer()).fit(obs)
    brain = EnsembleBrain(weighting="uniform").fit(fz.transform(obs), lab)
    gate = UncertaintyGate(epistemic_max=0.3, aleatoric_max=0.7,
                           confidence_min=0.4, margin_min=0.05, allow_gather=True)
    rt = _load_runtime()
    path = tmp_path / "agent.npz"
    rt.export_agent(str(path), fz, brain, gate)
    edge = rt.EdgeAgent(str(path))

    test_obs, _ = collect(5, seed0=9000)
    reps = brain.report(fz.transform(test_obs))
    for o, rep in zip(test_obs, reps):
        e = edge.decide(o.features, o.text)
        assert e["action"] == str(rep.argmax)
        assert np.abs(e["posterior"] - rep.posterior).max() < 1e-10
        assert abs(e["stats"]["epistemic"] - rep.stats.epistemic) < 1e-10
        assert e["gate"] == gate(rep.stats).value
