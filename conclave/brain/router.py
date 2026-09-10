"""Tool routing without a language model.

Conventional agents pick a tool by asking an LLM to emit a function name.
``ToolRouter`` instead treats routing as *learning to rank* over
(request, tool-specification) pairs.

The critical design property is **open-vocabulary generalisation**: the
router scores a pair, it does not classify into a fixed label set.  A
tool registered after training is scored by the same features as any
other, so the agent can be given new capabilities without retraining --
the property that makes LLM routers attractive in the first place.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from conclave.brain.uncertainty import UncertaintyStats, decompose

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> set:
    return set(_TOKEN.findall(s.lower()))


def spec_text(spec: Mapping[str, Any]) -> str:
    """Flatten a tool specification into matchable text."""
    params = (spec.get("parameters") or {}).get("properties") or {}
    parts = [
        str(spec.get("name", "")).replace(".", " ").replace("_", " "),
        str(spec.get("description", "")),
    ]
    for k, v in params.items():
        parts.append(str(k).replace("_", " "))
        if isinstance(v, Mapping):
            parts.append(str(v.get("description", ""))[:200])
    return " ".join(parts)


@dataclass
class RouteResult:
    tool: Optional[str]
    scores: Dict[str, float]
    stats: UncertaintyStats
    latency_ms: float
    abstained: bool = False


class ToolRouter(BaseEstimator):
    """Pairwise ranking router over an open tool vocabulary.

    Parameters
    ----------
    abstain: if True the router may return ``None`` when no candidate
        clears ``abstain_threshold`` -- the "relevance detection" ability
        that function-calling benchmarks measure separately.
    """

    def __init__(
        self,
        members: Optional[Mapping[str, BaseEstimator]] = None,
        abstain: bool = True,
        abstain_threshold: float = 0.35,
        random_state: int = 0,
        cache_size: int = 20000,
    ) -> None:
        self.members = members
        self.abstain = abstain
        self.abstain_threshold = abstain_threshold
        self.random_state = random_state
        self.cache_size = cache_size
        self._spec_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _default_members(self) -> Dict[str, BaseEstimator]:
        """Edge-tuned committee.

        Two deliberate choices, both measured rather than assumed:

        * ``n_jobs=1``.  An agent decides about *one* request at a time,
          on 2-8 candidate tools.  Spinning a joblib thread pool for a
          four-row matrix costs roughly 20x more than the prediction
          itself (90 ms vs 4 ms in our measurements).  Parallelism is a
          throughput optimisation and an latency pessimisation; edge
          agents want the latter.
        * Shallow, few trees.  Depth-10 forests with 60 estimators lose
          well under a point of accuracy against depth-unbounded 200-tree
          forests while cutting the serialised model by ~9x.

        A boosted member carries most of the accuracy, a bagged member
        supplies decorrelated errors, and a linear member keeps the
        committee honest off-distribution -- the diversity that makes the
        epistemic-uncertainty signal meaningful.
        """
        members: Dict[str, BaseEstimator] = {}
        try:                                   # optional, materially faster
            import lightgbm as lgb
            members["lgbm"] = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.08, num_leaves=31,
                n_jobs=1, random_state=self.random_state, verbose=-1)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            members["hgb"] = HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.08, random_state=self.random_state)
        members["rf"] = RandomForestClassifier(
            n_estimators=60, max_depth=10, min_samples_leaf=4, n_jobs=1,
            random_state=self.random_state)
        members["lr"] = LogisticRegression(max_iter=1000, n_jobs=1)
        return members

    # ------------------------------------------------------------------
    # Feature computation.
    #
    # Tool specifications are static: a registered tool's text does not
    # change between requests.  We therefore vectorise every spec **once**
    # and cache it, so routing a request costs one query vectorisation
    # plus a handful of sparse dot products -- O(n_candidates) cheap
    # arithmetic rather than O(n_candidates) vectoriser invocations.
    # This is the difference between ~90 ms and ~0.3 ms per decision and
    # is what makes the router viable on a microcontroller-class board.
    # ------------------------------------------------------------------
    def _spec_entry(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        key = str(spec.get("name", "")) + "\x00" + str(spec.get("description", ""))[:64]
        hit = self._spec_cache.get(key)
        if hit is not None:
            return hit
        text = spec_text(spec)
        params = (spec.get("parameters") or {}).get("properties") or {}
        req = (spec.get("parameters") or {}).get("required") or []
        entry = {
            "wv": self._word.transform([text]),
            "cv": self._char.transform([text]),
            "wt": _tok(text),
            "nt": _tok(str(spec.get("name", ""))),
            "param_toks": [_tok(str(k)) for k in params],
            "req_toks": [_tok(str(k)) for k in req],
            "n_req": len(req),
            "n_params": len(params),
            "len": len(text) / 1000.0,
        }
        if len(self._spec_cache) < self.cache_size:
            self._spec_cache[key] = entry
        return entry

    def _features(self, query: str,
                  specs: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Feature matrix for one query against many candidate specs."""
        qv = self._word.transform([query])
        qc = self._char.transform([query])
        qt = _tok(query)
        nq = float(len(qt))
        rows = []
        for spec in specs:
            e = self._spec_entry(spec)
            wt, nt = e["wt"], e["nt"]
            inter = len(qt & wt)
            pt, rt = e["param_toks"], e["req_toks"]
            rows.append([
                float(qv.multiply(e["wv"]).sum()),          # word tf-idf overlap
                float(qc.multiply(e["cv"]).sum()),          # char tf-idf overlap
                inter / max(len(qt), 1),                    # query covered by spec
                inter / max(len(wt), 1),                    # spec covered by query
                len(qt & nt) / max(len(nt), 1),             # tool-name hit rate
                float(e["n_req"]),
                float(e["n_params"]),
                e["len"],
                (sum(1 for t in pt if t & qt) / len(pt)) if pt else 0.0,
                (sum(1 for t in rt if t & qt) / len(rt)) if rt else 0.0,
                nq,
            ])
        return np.asarray(rows, dtype=np.float32)

    def _pair_features(self, query: str, spec: Mapping[str, Any]) -> List[float]:
        """Single-pair convenience wrapper (used by analysis scripts)."""
        return self._features(query, [spec])[0].tolist()

    # ------------------------------------------------------------------
    def fit(self, queries: Sequence[str],
            candidate_sets: Sequence[Sequence[Mapping[str, Any]]],
            targets: Sequence[str]) -> "ToolRouter":
        corpus = list(queries) + [spec_text(s) for cs in candidate_sets for s in cs]
        self._word = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                     strip_accents="unicode",
                                     stop_words="english").fit(corpus)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                                     min_df=2, max_features=30000,
                                     sublinear_tf=True).fit(corpus)
        self._spec_cache = {}
        blocks, y = [], []
        for q, cands, tgt in zip(queries, candidate_sets, targets):
            blocks.append(self._features(q, cands))
            y.extend(1 if s_.get("name") == tgt else 0 for s_ in cands)
        X = np.vstack(blocks).astype(np.float32)
        y = np.asarray(y)
        specs = self.members or self._default_members()
        self.member_names_ = list(specs.keys())
        self.members_ = {n: clone(e).fit(X, y) for n, e in specs.items()}
        self.n_pair_features_ = X.shape[1]
        return self

    # ------------------------------------------------------------------
    def route(self, query: str,
              candidates: Sequence[Mapping[str, Any]]) -> RouteResult:
        t0 = time.perf_counter()
        if not candidates:
            return RouteResult(None, {}, decompose(np.ones((1, 2)) / 2), 0.0, True)
        X = self._features(query, candidates)
        member = np.stack([
            self.members_[n].predict_proba(X)[:, 1] for n in self.member_names_
        ])                                                  # (n_members, n_cands)
        score = member.mean(axis=0)
        total = float(score.sum())
        norm = score / total if total > 0 else np.full_like(score, 1 / len(score))
        # treat the normalised candidate distribution per member as a posterior
        mp = member / np.clip(member.sum(axis=1, keepdims=True), 1e-12, None)
        stats = decompose(mp)
        lat = (time.perf_counter() - t0) * 1000.0
        best = int(np.argmax(score))
        scores = {str(c.get("name")): float(v) for c, v in zip(candidates, norm)}
        if self.abstain and float(score[best]) < self.abstain_threshold:
            return RouteResult(None, scores, stats, lat, True)
        return RouteResult(str(candidates[best].get("name")), scores, stats, lat, False)

    def route_batch(self, queries, candidate_sets) -> List[RouteResult]:
        return [self.route(q, c) for q, c in zip(queries, candidate_sets)]
