"""CONCLAVE - agentic AI whose cognitive core is a committee, not a language model.

The name is the design. A conclave is a decision-making body deliberately
sealed off from outside communication, which must nevertheless reach a
binding decision -- by vote, among its own members, with no way to phone
anyone for help. That is exactly the situation of an autonomous agent
during a flood, when the cell towers are down.

So CONCLAVE replaces the LLM at the centre of the agent loop with a small,
deliberately heterogeneous committee of scikit-learn estimators. The vote
decides the action; the *spread* of the vote decides whether to act at all.

Design goals
------------
1. **Offline-first.**  No network, no API keys, no GPU.  The whole agent
   is a few hundred kilobytes of scikit-learn estimators.
2. **Bounded latency.**  Inference is a single forward pass through a
   fixed set of estimators, so worst-case decision time is a constant
   that can be measured and certified -- unlike autoregressive decoding.
3. **Uncertainty as a first-class control signal.**  Disagreement among
   committee members is cheap, well-calibrated epistemic uncertainty.
   CONCLAVE routes on it: act, gather more evidence, or escalate to a
   human.  A committee that cannot agree should not be acting.
"""

__version__ = "0.1.0"

from conclave.core.types import (
    Observation, Action, Decision, Episode, Step, ActionKind,
)
from conclave.core.agent import Agent
from conclave.core.registry import ToolRegistry
from conclave.brain.ensemble import EnsembleBrain
from conclave.brain.uncertainty import UncertaintyGate, GateOutcome
from conclave.memory.episodic import EpisodicMemory
from conclave.planning.planner import Planner, Goal
from conclave.tools.base import Tool

__all__ = [
    "Observation", "Action", "Decision", "Episode", "Step", "ActionKind",
    "Agent", "ToolRegistry", "EnsembleBrain", "UncertaintyGate",
    "GateOutcome", "EpisodicMemory", "Planner", "Goal", "Tool",
]
