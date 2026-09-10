"""Hard safety constraints that sit above the learned policy.

No statistical model should be able to authorise an irreversible act on
its own.  Constraints are checked *after* the brain proposes and *before*
the tool executes; a violated constraint converts the action into an
escalation, which is recorded in the episode for audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence

from conclave.core.types import Action, ActionKind, Observation


@dataclass
class Constraint:
    name: str
    predicate: Callable[[Action, Observation, Mapping[str, Any]], bool]
    message: str = ""
    escalate: bool = True

    def check(self, action: Action, obs: Observation,
              ctx: Mapping[str, Any]) -> Optional[str]:
        try:
            ok = self.predicate(action, obs, ctx)
        except Exception as exc:                       # a broken rule fails closed
            return f"{self.name}: predicate error ({exc})"
        return None if ok else (self.message or f"{self.name} violated")


class ConstraintSet:
    def __init__(self, constraints: Optional[Sequence[Constraint]] = None) -> None:
        self.constraints: List[Constraint] = list(constraints or [])

    def add(self, c: Constraint) -> "ConstraintSet":
        self.constraints.append(c)
        return self

    def violations(self, action: Action, obs: Observation,
                   ctx: Mapping[str, Any]) -> List[str]:
        out = []
        for c in self.constraints:
            msg = c.check(action, obs, ctx)
            if msg:
                out.append(msg)
        return out


def require_confidence(threshold: float, tools: Sequence[str]) -> Constraint:
    """Named tools may only fire above a confidence floor."""
    def pred(a: Action, o: Observation, ctx: Mapping[str, Any]) -> bool:
        if a.kind is not ActionKind.TOOL or a.tool not in tools:
            return True
        return float(ctx.get("confidence", 0.0)) >= threshold
    return Constraint(
        name=f"confidence>={threshold}",
        predicate=pred,
        message=f"irreversible tool requires confidence >= {threshold}",
    )


def rate_limit(tool_name: str, max_calls: int) -> Constraint:
    """Cap how often a tool may fire within one episode."""
    def pred(a: Action, o: Observation, ctx: Mapping[str, Any]) -> bool:
        if a.kind is not ActionKind.TOOL or a.tool != tool_name:
            return True
        return int(ctx.get("call_counts", {}).get(tool_name, 0)) < max_calls
    return Constraint(name=f"rate_limit({tool_name},{max_calls})",
                      predicate=pred,
                      message=f"{tool_name} exceeded {max_calls} calls this episode")
