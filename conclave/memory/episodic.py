"""Case-based episodic memory -- the ensemble agent's answer to in-context learning.

An LLM agent adapts to a new situation by having examples pasted into its
prompt.  CONCLAVE instead keeps a bounded store of past (observation,
action, outcome) cases and retrieves the nearest ones at decision time.
Retrieved cases contribute a prior over actions that is blended with the
ensemble posterior.

This costs kilobytes and microseconds, works offline, and -- unlike a
context window -- degrades gracefully as it fills, because eviction is
by measured utility rather than recency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class Case:
    key: np.ndarray
    action: str
    reward: float
    meta: Dict[str, Any] = field(default_factory=dict)
    uses: int = 0


class EpisodicMemory:
    """Bounded k-NN store over decision keys.

    Parameters
    ----------
    capacity: maximum number of retained cases.
    k: neighbours consulted per query.
    metric: any metric accepted by ``sklearn.neighbors``.
    """

    def __init__(self, capacity: int = 2000, k: int = 5,
                 metric: str = "cosine") -> None:
        self.capacity = capacity
        self.k = k
        self.metric = metric
        self._cases: List[Case] = []
        self._index: Optional[NearestNeighbors] = None
        self._dirty = True

    # ------------------------------------------------------------------
    def add(self, key: np.ndarray, action: str, reward: float,
            meta: Optional[Dict[str, Any]] = None) -> None:
        self._cases.append(Case(np.asarray(key, dtype=np.float32).ravel(),
                                action, float(reward), meta or {}))
        self._dirty = True
        if len(self._cases) > self.capacity:
            self._evict()

    def _evict(self) -> None:
        """Drop the least useful case: low reward, rarely retrieved."""
        util = [c.reward + 0.1 * c.uses for c in self._cases]
        self._cases.pop(int(np.argmin(util)))

    def _rebuild(self) -> None:
        if not self._cases:
            self._index = None
            return
        M = np.vstack([c.key for c in self._cases])
        n = min(self.k, len(self._cases))
        self._index = NearestNeighbors(n_neighbors=n, metric=self.metric).fit(M)
        self._dirty = False

    # ------------------------------------------------------------------
    def retrieve(self, key: np.ndarray) -> List[Tuple[Case, float]]:
        """Nearest cases with similarity in [0, 1]."""
        if self._dirty:
            self._rebuild()
        if self._index is None:
            return []
        q = np.asarray(key, dtype=np.float32).reshape(1, -1)
        dist, idx = self._index.kneighbors(q)
        out = []
        for d, i in zip(dist[0], idx[0]):
            c = self._cases[int(i)]
            c.uses += 1
            out.append((c, float(max(0.0, 1.0 - d))))
        return out

    def prior(self, key: np.ndarray, actions: Sequence[str],
              temperature: float = 1.0) -> np.ndarray:
        """Similarity- and reward-weighted prior over ``actions``."""
        p = np.zeros(len(actions), dtype=np.float64)
        idx = {a: i for i, a in enumerate(actions)}
        for case, sim in self.retrieve(key):
            if case.action in idx and case.reward > 0:
                p[idx[case.action]] += sim * case.reward
        if p.sum() <= 0:
            return np.full(len(actions), 1.0 / max(len(actions), 1))
        p = p ** (1.0 / max(temperature, 1e-6))
        return p / p.sum()

    def __len__(self) -> int:
        return len(self._cases)

    def nbytes(self) -> int:
        return sum(c.key.nbytes for c in self._cases)
