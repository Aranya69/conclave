"""Deterministic symbolic planning over tool preconditions and effects.

The ensemble decides *what the situation is*; the planner decides *what
order to do things in*.  Separating the two is what lets CONCLAVE give a
worst-case decision-time bound: planning is forward search over a small,
finite tool set with an admissible heuristic, so it terminates in a
number of steps bounded by ``max_depth`` -- there is no analogue of an
LLM's unbounded reasoning chain.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from conclave.core.registry import ToolRegistry
from conclave.tools.base import Tool


@dataclass(frozen=True)
class Goal:
    """A goal is a set of predicates that must hold."""
    achieve: FrozenSet[str]
    avoid: FrozenSet[str] = frozenset()

    @staticmethod
    def of(*preds: str) -> "Goal":
        return Goal(achieve=frozenset(preds))

    def satisfied_by(self, state: FrozenSet[str]) -> bool:
        return self.achieve <= state and not (self.avoid & state)


@dataclass
class Plan:
    steps: List[str] = field(default_factory=list)
    cost: float = 0.0
    complete: bool = True

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)


class Planner:
    """Cost-optimal forward search (uniform-cost with a set-cover heuristic).

    Parameters
    ----------
    max_depth: hard bound on plan length -- guarantees termination.
    forbid_irreversible: if set, irreversible tools are excluded unless
        they are the only way to satisfy the goal, and are reported so a
        human can approve them.
    """

    def __init__(self, registry: ToolRegistry, max_depth: int = 8,
                 max_expansions: int = 5000,
                 forbid_irreversible: bool = False) -> None:
        self.registry = registry
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        self.forbid_irreversible = forbid_irreversible

    def _applicable(self, state: FrozenSet[str]) -> List[Tool]:
        out = []
        for t in self.registry:
            if self.forbid_irreversible and t.irreversible:
                continue
            if set(t.preconditions) <= set(state):
                out.append(t)
        return out

    @staticmethod
    def _h(state: FrozenSet[str], goal: Goal) -> float:
        """Admissible: at least one action per unmet predicate."""
        return float(len(goal.achieve - state))

    def plan(self, state: Sequence[str], goal: Goal) -> Plan:
        start = frozenset(state)
        if goal.satisfied_by(start):
            return Plan([], 0.0, True)

        counter = itertools.count()
        frontier: List[Tuple[float, int, float, FrozenSet[str], List[str]]] = []
        heapq.heappush(frontier, (self._h(start, goal), next(counter), 0.0, start, []))
        best: Dict[FrozenSet[str], float] = {start: 0.0}
        expansions = 0

        while frontier and expansions < self.max_expansions:
            _, _, g, state_i, path = heapq.heappop(frontier)
            expansions += 1
            if goal.satisfied_by(state_i):
                return Plan(path, g, True)
            if len(path) >= self.max_depth:
                continue
            for t in self._applicable(state_i):
                nxt = frozenset(set(state_i) | set(t.effects))
                if nxt == state_i:
                    continue
                ng = g + t.cost
                if ng < best.get(nxt, float("inf")):
                    best[nxt] = ng
                    heapq.heappush(
                        frontier,
                        (ng + self._h(nxt, goal), next(counter), ng, nxt, path + [t.name]),
                    )
        return Plan([], float("inf"), False)
