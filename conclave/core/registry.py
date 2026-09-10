"""Registry of the agent's capabilities."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from conclave.tools.base import Tool


class ToolRegistry:
    """Holds tools and exposes their specs for routing.

    Tools may be added after the router is trained; the router scores
    (request, spec) pairs so new capabilities need no retraining.
    """

    def __init__(self, tools: Optional[Sequence[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool) -> "ToolRegistry":
        if t.name in self._tools:
            raise ValueError(f"tool {t.name!r} already registered")
        self._tools[t.name] = t
        return self

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}")
        return self._tools[name]

    def specs(self, subset: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        names = subset if subset is not None else list(self._tools)
        return [self._tools[n].spec() for n in names if n in self._tools]

    def names(self) -> List[str]:
        return list(self._tools)

    def invoke(self, name: str, args: Mapping[str, Any]) -> Any:
        return self.get(name)(**args)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools
