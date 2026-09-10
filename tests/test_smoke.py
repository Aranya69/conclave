"""End-to-end smoke tests: every module must actually run."""
import numpy as np
import pytest

from conclave import (Agent, EnsembleBrain, EpisodicMemory, Observation,
                   ToolRegistry, UncertaintyGate, Goal, Planner)
from conclave.brain.featurize import (BudgetedTextFeaturizer, FusedFeaturizer,
                                   TabularFeaturizer)
from conclave.brain.uncertainty import decompose, risk_coverage_curve
from conclave.core.types import Action, ActionKind
from conclave.envs.disaster import (ACTIONS, DisasterConfig, FloodResponseEnv,
                                 expert_policy)
from conclave.planning.rules import ConstraintSet, require_confidence
from conclave.tools.base import tool


def test_uncertainty_decomposition():
    agree = np.array([[0.9, 0.1], [0.88, 0.12], [0.91, 0.09]])
    split = np.array([[0.98, 0.02], [0.02, 0.98], [0.5, 0.5]])
    a, s = decompose(agree), decompose(split)
    assert a.epistemic < s.epistemic, "disagreement must raise epistemic term"
    assert 0 <= a.total <= 1 and 0 <= s.total <= 1
    ambiguous = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    amb = decompose(ambiguous)
    assert amb.aleatoric > 0.9 and amb.epistemic < 1e-6


def test_gate_routes_on_disagreement():
    g = UncertaintyGate(epistemic_max=0.2, confidence_min=0.5, allow_gather=True)
    from conclave.brain.uncertainty import GateOutcome
    conf = decompose(np.array([[0.95, 0.05], [0.93, 0.07], [0.94, 0.06]]))
    assert g(conf) is GateOutcome.ACT
    split = decompose(np.array([[0.99, 0.01], [0.01, 0.99], [0.5, 0.5]]))
    assert g(split) in (GateOutcome.GATHER, GateOutcome.ESCALATE)


def test_tool_and_registry():
    @tool(description="Broadcast a flood warning to a zone")
    def issue_alert(zone: str, severity: int = 1) -> str:
        return f"alert:{zone}:{severity}"
    reg = ToolRegistry([issue_alert])
    assert "issue_alert" in reg
    assert reg.invoke("issue_alert", {"zone": "A", "severity": 2}) == "alert:A:2"
    with pytest.raises(ValueError):
        reg.invoke("issue_alert", {"severity": 2})       # missing required
    spec = reg.specs()[0]
    assert spec["name"] == "issue_alert" and "zone" in spec["parameters"]["properties"]


def test_planner_finds_ordered_plan():
    @tool(description="charge the radio", preconditions=[], effects=["radio_ready"])
    def charge_radio() -> str: return "ok"
    @tool(description="broadcast", preconditions=["radio_ready"], effects=["warned"])
    def broadcast() -> str: return "ok"
    reg = ToolRegistry([charge_radio, broadcast])
    p = Planner(reg).plan(state=[], goal=Goal.of("warned"))
    assert p.complete and p.steps == ["charge_radio", "broadcast"]
    assert not Planner(reg).plan(state=[], goal=Goal.of("impossible")).complete


def test_episodic_memory_prior():
    m = EpisodicMemory(capacity=50, k=3)
    for _ in range(10):
        m.add(np.array([1.0, 0.0, 0.0]), "evacuate_zone", reward=2.0)
        m.add(np.array([0.0, 1.0, 0.0]), "monitor", reward=1.0)
    p = m.prior(np.array([1.0, 0.02, 0.0]), ACTIONS)
    assert ACTIONS[int(np.argmax(p))] == "evacuate_zone"
    assert abs(p.sum() - 1.0) < 1e-6


def _train_agent(n_episodes=12, seed=0):
    rng = np.random.default_rng(seed)
    obs_list, labels = [], []
    for e in range(n_episodes):
        env = FloodResponseEnv(DisasterConfig(seed=seed * 100 + e, horizon=40))
        o = env.reset()
        for _ in range(40):
            a = expert_policy(o, rng)
            obs_list.append(o); labels.append(a)
            o, _, done, _ = env.step(Action(kind=ActionKind.TOOL, tool=a))
            if done: break
    fz = FusedFeaturizer(
        text=BudgetedTextFeaturizer(n_word_features=2**12, n_char_features=2**12),
        tabular=TabularFeaturizer(),
    ).fit(obs_list)
    X = fz.transform(obs_list)
    brain = EnsembleBrain().fit(X, np.array(labels))
    return brain, fz, obs_list, labels


def test_brain_learns_and_reports_uncertainty():
    brain, fz, obs, labels = _train_agent()
    X = fz.transform(obs)
    acc = (brain.predict(X) == np.array(labels)).mean()
    assert acc > 0.6, f"behaviour cloning too weak: {acc}"
    rep = brain.report(X[:5])
    assert len(rep) == 5
    assert all(0 <= r.stats.epistemic <= 1 for r in rep)
    assert brain.footprint_bytes()["_total"] > 0


def test_agent_runs_episode_and_is_bounded():
    brain, fz, _, _ = _train_agent()
    agent = Agent(brain=brain, featurizer=fz, registry=ToolRegistry(),
                  gate=UncertaintyGate(), memory=EpisodicMemory(capacity=200))
    env = FloodResponseEnv(DisasterConfig(seed=999, horizon=30))
    ep = agent.run(env, goal=Goal.of("district_safe"), max_steps=30)
    assert 0 < len(ep.steps) <= 30
    assert all(s.decision.latency_ms < 250 for s in ep.steps), "latency tail too long"
    d = ep.to_dict()
    assert d["n_steps"] == len(ep.steps) and "steps" in d


def test_constraints_force_escalation():
    brain, fz, _, _ = _train_agent()
    cs = ConstraintSet([require_confidence(1.01, ["open_sluice", "evacuate_zone",
                                                  "issue_alert", "monitor",
                                                  "deploy_rescue", "request_supplies"])])
    agent = Agent(brain=brain, featurizer=fz, constraints=cs,
                  gate=UncertaintyGate(confidence_min=0.0, epistemic_max=1.0,
                                       aleatoric_max=1.0))
    env = FloodResponseEnv(DisasterConfig(seed=3, horizon=10))
    ep = agent.run(env, max_steps=10)
    kinds = {s.decision.action.kind for s in ep.steps}
    assert ActionKind.ESCALATE in kinds, "unsatisfiable constraint must escalate"


def test_risk_coverage_monotone_ish():
    brain, fz, obs, labels = _train_agent()
    X = fz.transform(obs)
    M = brain.member_proba(X)
    y = np.searchsorted(brain.classes_, np.array(labels))
    rc = risk_coverage_curve(M, y)
    assert rc["accuracy"][0] >= rc["accuracy"][-1] - 0.05, \
        "confident subset should not be worse than the full set"
