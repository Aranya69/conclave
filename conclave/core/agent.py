"""The CONCLAVE agent loop.

One tick:

    observe -> featurise -> ensemble posterior + uncertainty
            -> episodic prior -> gate -> constraint check -> act

Every branch is a bounded amount of arithmetic, so the loop has a
measurable worst-case latency.  Nothing here reaches the network.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from conclave.brain.ensemble import BrainReport, EnsembleBrain
from conclave.brain.uncertainty import GateOutcome, UncertaintyGate
from conclave.core.registry import ToolRegistry
from conclave.core.types import (
    Action, ActionKind, Decision, Episode, Observation, Step,
)
from conclave.memory.episodic import EpisodicMemory
from conclave.planning.planner import Goal, Plan, Planner
from conclave.planning.rules import ConstraintSet


@dataclass
class AgentConfig:
    max_steps: int = 20
    memory_weight: float = 0.25        # blend weight for the episodic prior
    learn_online: bool = True          # write successful steps back to memory
    record_member_votes: bool = True


class Agent:
    """An autonomous agent whose cognitive core is an ML ensemble.

    Parameters
    ----------
    brain: fitted :class:`EnsembleBrain` whose classes are tool names (or
        any action label understood by ``action_fn``).
    featurizer: object with ``transform(list_of_observations)``.
    registry: available tools.
    gate: uncertainty gate; defaults to the brain's calibrated gate.
    """

    def __init__(
        self,
        brain: EnsembleBrain,
        featurizer: Any,
        registry: Optional[ToolRegistry] = None,
        gate: Optional[UncertaintyGate] = None,
        memory: Optional[EpisodicMemory] = None,
        planner: Optional[Planner] = None,
        constraints: Optional[ConstraintSet] = None,
        config: Optional[AgentConfig] = None,
        arg_builder: Optional[Callable[[str, Observation], Dict[str, Any]]] = None,
    ) -> None:
        self.brain = brain
        self.featurizer = featurizer
        self.registry = registry or ToolRegistry()
        self.gate = gate or brain.gate or UncertaintyGate()
        self.memory = memory
        self.planner = planner
        self.constraints = constraints or ConstraintSet()
        self.config = config or AgentConfig()
        self.arg_builder = arg_builder or (lambda tool, obs: {})
        self._call_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def _key(self, X) -> np.ndarray:
        """Dense retrieval key for episodic memory."""
        row = X[0]
        if hasattr(row, "toarray"):
            row = row.toarray().ravel()
        row = np.asarray(row, dtype=np.float32).ravel()
        if row.size > 512:                              # cheap random projection
            rng = np.random.default_rng(0)
            if not hasattr(self, "_proj") or self._proj.shape[0] != row.size:
                self._proj = rng.normal(0, 1 / np.sqrt(512), (row.size, 512)).astype(np.float32)
            row = row @ self._proj
        return row

    def decide(self, obs: Observation) -> Decision:
        """Produce a fully-audited decision for one observation."""
        t0 = time.perf_counter()
        X = self.featurizer.transform([obs])
        rep: BrainReport = self.brain.report(X)[0]
        posterior = rep.posterior.copy()
        labels = [str(l) for l in rep.labels]

        if self.memory is not None and len(self.memory) > 0:
            prior = self.memory.prior(self._key(X), labels)
            w = self.config.memory_weight
            posterior = (1 - w) * posterior + w * prior
            posterior = posterior / max(posterior.sum(), 1e-12)

        outcome = self.gate(rep.stats)
        kind = {
            GateOutcome.ACT: ActionKind.TOOL,
            GateOutcome.GATHER: ActionKind.GATHER,
            GateOutcome.ESCALATE: ActionKind.ESCALATE,
            GateOutcome.ABSTAIN: ActionKind.ABSTAIN,
        }[outcome]

        chosen = labels[int(np.argmax(posterior))]
        tool = chosen if kind is ActionKind.TOOL else None
        args = self.arg_builder(tool, obs) if tool else {}
        action = Action(kind=kind, tool=tool, args=args,
                        reason=f"gate={outcome.value}")

        ctx = {"confidence": float(posterior.max()),
               "call_counts": dict(self._call_counts)}
        viol = self.constraints.violations(action, obs, ctx)
        if viol:
            action = Action(kind=ActionKind.ESCALATE, reason="; ".join(viol))
            outcome = GateOutcome.ESCALATE

        lat = (time.perf_counter() - t0) * 1000.0
        return Decision(
            action=action,
            posterior={l: float(p) for l, p in zip(labels, posterior)},
            member_votes=(rep.member_mapping(self.brain.member_names_)
                          if self.config.record_member_votes else {}),
            confidence=float(posterior.max()),
            entropy=rep.stats.total,
            disagreement=rep.stats.epistemic,
            margin=rep.stats.margin,
            gate=outcome.value,
            latency_ms=lat,
        )

    # ------------------------------------------------------------------
    def step(self, obs: Observation, index: int = 0) -> Step:
        d = self.decide(obs)
        st = Step(index=index, observation=obs, decision=d)
        if d.action.kind is ActionKind.TOOL and d.action.tool in self.registry:
            try:
                st.result = self.registry.invoke(d.action.tool, d.action.args)
                self._call_counts[d.action.tool] = self._call_counts.get(d.action.tool, 0) + 1
            except Exception as exc:
                st.error = f"{type(exc).__name__}: {exc}"
        return st

    def run(self, env, goal: Optional[Goal] = None,
            max_steps: Optional[int] = None) -> Episode:
        """Drive a sequential environment to termination.

        ``env`` must expose ``reset() -> Observation`` and
        ``step(Action) -> (Observation, reward, done, info)``.
        """
        ep = Episode(goal=str(goal) if goal else None)
        self._call_counts.clear()
        obs = env.reset()
        n = max_steps or self.config.max_steps
        for i in range(n):
            d = self.decide(obs)
            nxt, reward, done, info = env.step(d.action)
            st = Step(index=i, observation=obs, decision=d,
                      reward=float(reward), result=info)
            ep.add(st)
            if (self.memory is not None and self.config.learn_online
                    and d.action.kind is ActionKind.TOOL and reward > 0):
                X = self.featurizer.transform([obs])
                self.memory.add(self._key(X), str(d.action.tool), float(reward))
            obs = nxt
            if done:
                break
        ep.finished = time.time()
        ep.success = bool(getattr(env, "succeeded", lambda: None)())
        return ep
