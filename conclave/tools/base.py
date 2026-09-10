"""Typed tools with declarative schemas.

A tool is anything the agent can do to the world: open a sluice gate,
broadcast an SMS alert, write to a local queue.  Schemas are plain dicts
in JSON-Schema style so the same specification feeds both the router's
features and runtime argument validation.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=lambda: {"properties": {}, "required": []})
    # Preconditions/effects let the symbolic planner reason about ordering.
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    cost: float = 1.0
    irreversible: bool = False

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}

    def validate(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        props = self.parameters.get("properties", {}) or {}
        req = self.parameters.get("required", []) or []
        missing = [r for r in req if r not in args]
        if missing:
            raise ValueError(f"{self.name}: missing required args {missing}")
        unknown = [k for k in args if k not in props] if props else []
        if unknown:
            raise ValueError(f"{self.name}: unknown args {unknown}")
        return dict(args)

    def __call__(self, **args) -> Any:
        return self.fn(**self.validate(args))


def tool(name: Optional[str] = None, description: str = "",
         parameters: Optional[Dict[str, Any]] = None, cost: float = 1.0,
         irreversible: bool = False, preconditions: Optional[List[str]] = None,
         effects: Optional[List[str]] = None):
    """Decorator turning a plain function into a :class:`Tool`.

    Parameter schema is inferred from the signature when not supplied.
    """
    def deco(fn: Callable[..., Any]) -> Tool:
        params = parameters
        if params is None:
            sig = inspect.signature(fn)
            props, req = {}, []
            for pname, p in sig.parameters.items():
                ann = p.annotation
                t = {int: "integer", float: "number", bool: "boolean",
                     str: "string"}.get(ann, "string")
                props[pname] = {"type": t, "description": ""}
                if p.default is inspect.Parameter.empty:
                    req.append(pname)
            params = {"properties": props, "required": req}
        return Tool(
            name=name or fn.__name__,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            fn=fn, parameters=params, cost=cost, irreversible=irreversible,
            preconditions=preconditions or [], effects=effects or [],
        )
    return deco
