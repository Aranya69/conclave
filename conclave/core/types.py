"""Core data types for the CONCLAVE agent loop.

Everything the agent passes around is a frozen dataclass so that episodes
are trivially serialisable and experiments are reproducible.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class ActionKind(str, Enum):
    """What the agent decided to do this tick."""

    TOOL = "tool"            # invoke a registered tool
    ABSTAIN = "abstain"      # no tool is appropriate; do nothing
    GATHER = "gather"        # uncertainty too high; acquire more evidence
    ESCALATE = "escalate"    # hand control to a human operator
    TERMINATE = "terminate"  # goal reached or plan exhausted


@dataclass(frozen=True)
class Observation:
    """One reading of the world.

    ``features`` holds numeric/tabular signals (sensor values, counts).
    ``text`` holds any free-form report (a radio transcript, an SMS from
    a field responder).  Either may be empty; CONCLAVE featurises whichever
    modality is present.
    """

    features: Mapping[str, float] = field(default_factory=dict)
    text: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def is_empty(self) -> bool:
        return not self.features and not self.text.strip()


@dataclass(frozen=True)
class Action:
    """A concrete, executable act."""

    kind: ActionKind
    tool: Optional[str] = None
    args: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        if self.kind is ActionKind.TOOL:
            a = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
            return f"{self.tool}({a})"
        return f"<{self.kind.value}>"


@dataclass(frozen=True)
class Decision:
    """An action plus the full epistemic state that produced it.

    This is what makes CONCLAVE auditable: every decision carries the
    per-member votes, the aggregated posterior, and the uncertainty
    statistics that drove the gate.  A disaster-response operator can
    inspect exactly why the agent chose to escalate.
    """

    action: Action
    posterior: Mapping[str, float] = field(default_factory=dict)
    member_votes: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    confidence: float = 0.0
    entropy: float = 0.0
    disagreement: float = 0.0
    margin: float = 0.0
    gate: str = ""
    latency_ms: float = 0.0

    def top_k(self, k: int = 3) -> List[tuple]:
        return sorted(self.posterior.items(), key=lambda kv: -kv[1])[:k]


@dataclass
class Step:
    """One turn of the sense-decide-act loop."""

    index: int
    observation: Observation
    decision: Decision
    result: Any = None
    reward: float = 0.0
    error: Optional[str] = None


@dataclass
class Episode:
    """A complete agent run, retained for case-based memory and audit."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: Optional[str] = None
    steps: List[Step] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    success: Optional[bool] = None

    def add(self, step: Step) -> None:
        self.steps.append(step)

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def n_escalations(self) -> int:
        return sum(1 for s in self.steps
                   if s.decision.action.kind is ActionKind.ESCALATE)

    @property
    def duration_ms(self) -> float:
        end = self.finished if self.finished is not None else time.time()
        return (end - self.started) * 1000.0

    @property
    def decision_latencies(self) -> List[float]:
        return [s.decision.latency_ms for s in self.steps]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "success": self.success,
            "total_reward": self.total_reward,
            "n_steps": len(self.steps),
            "n_escalations": self.n_escalations,
            "duration_ms": self.duration_ms,
            "steps": [
                {
                    "index": s.index,
                    "action": str(s.decision.action),
                    "kind": s.decision.action.kind.value,
                    "confidence": s.decision.confidence,
                    "disagreement": s.decision.disagreement,
                    "gate": s.decision.gate,
                    "latency_ms": s.decision.latency_ms,
                    "reward": s.reward,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }
