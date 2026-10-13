"""The ensemble brain: a heterogeneous committee that decides what to do.

``EnsembleBrain`` replaces the language model at the centre of a
conventional agent.  It holds a set of *deliberately diverse* estimators
-- linear, probabilistic, and tree-based -- and aggregates them into a
posterior over actions together with an uncertainty decomposition.

Why heterogeneous rather than a single bagged forest?  Because epistemic
uncertainty is only meaningful if members can actually disagree for
structural reasons.  A random forest's trees share an inductive bias, so
their disagreement collapses off-distribution exactly when we need it
most.  Mixing model families keeps the disagreement signal alive.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from conclave.brain.uncertainty import (
    UncertaintyGate, UncertaintyStats, decompose,
)


def default_members(sparse_input: bool = True, seed: int = 0,
                    nonnegative: bool = True) -> Dict[str, BaseEstimator]:
    """A committee with genuinely different inductive biases.

    ``n_jobs=1`` throughout: an agent classifies one observation at a
    time, and joblib's thread-pool setup dominates prediction at that
    batch size.  Set ``n_jobs`` explicitly if you are scoring in bulk.

    The probabilistic member is chosen to match the data: ComplementNB is
    the strongest naive-Bayes variant for imbalanced non-negative counts,
    but it rejects negative inputs, which appear as soon as tabular
    features are standardised.  BernoulliNB binarises at zero and takes
    either, so the committee keeps a generative member in both regimes
    instead of silently degrading to two linear models.
    """
    if sparse_input:
        nb = ComplementNB(alpha=0.3) if nonnegative else BernoulliNB(alpha=0.3, binarize=0.0)
        return {
            "logreg": LogisticRegression(C=4.0, max_iter=400, n_jobs=1),
            "sgd_mh": SGDClassifier(loss="modified_huber", alpha=1e-5,
                                    max_iter=25, random_state=seed),
            "nb": nb,
        }
    return {
        "logreg": LogisticRegression(C=1.0, max_iter=500, n_jobs=1),
        "rf": RandomForestClassifier(n_estimators=100, max_depth=12,
                                     min_samples_leaf=2, n_jobs=1,
                                     random_state=seed),
        "et": ExtraTreesClassifier(n_estimators=100, max_depth=12,
                                   min_samples_leaf=2, n_jobs=1,
                                   random_state=seed),
    }


def _is_nonnegative(X) -> bool:
    """True if X contains no negative entries (sparse-safe)."""
    if hasattr(X, "data"):
        return X.data.size == 0 or float(X.data.min()) >= 0.0
    return float(np.min(X)) >= 0.0


@dataclass
class BrainReport:
    """Everything the brain knows about one decision."""

    posterior: np.ndarray
    member_probs: np.ndarray          # (n_members, n_classes)
    stats: UncertaintyStats
    labels: Sequence[Any]
    latency_ms: float = 0.0

    @property
    def argmax(self) -> Any:
        return self.labels[int(np.argmax(self.posterior))]

    def as_mapping(self) -> Dict[Any, float]:
        return {l: float(p) for l, p in zip(self.labels, self.posterior)}

    def member_mapping(self, names: Sequence[str]) -> Dict[str, Dict[Any, float]]:
        return {
            n: {l: float(p) for l, p in zip(self.labels, row)}
            for n, row in zip(names, self.member_probs)
        }


class EnsembleBrain(BaseEstimator, ClassifierMixin):
    """Soft-voting committee with optional learned member weights.

    Parameters
    ----------
    members:
        Mapping of name -> unfitted estimator.  Each must expose
        ``predict_proba`` after fitting (or ``decision_function``, which
        is softmax-normalised).
    weighting:
        ``"logloss"``  -- weight by inverse out-of-fold log loss. **Default.**
        ``"accuracy"`` -- weight members by out-of-fold accuracy.
        ``"uniform"``  -- plain average.

        The default is *not* uniform, and the reason is measured rather than
        theoretical. Uniform averaging assumes members are of comparable
        quality. When one is not -- naive Bayes on a task the linear members
        handle easily -- a uniform average drags the committee *below its own
        best member*: 95.0% against that member's 98.3% in our ablation
        (Experiment 6). Out-of-fold weighting detects the weak member and
        assigns it approximately zero weight, recovering 98.0%.

        Use ``"uniform"`` only when you have checked that members perform
        comparably -- or when the brain drives a closed control loop.  In the
        flood-response task (Experiment 7) uniform voting gave a significantly
        higher episode return than log-loss weighting (36.5 vs 33.5), although
        log-loss weighting is more accurate one decision at a time (98.0 vs
        96.3 %): the down-weighted member still decorrelates the committee's
        errors, and in a sequential task errors compound.
    gate:
        Uncertainty gate used by :meth:`decide`.  If ``None`` a default
        gate is constructed.
    """

    def __init__(
        self,
        members: Optional[Mapping[str, BaseEstimator]] = None,
        weighting: str = "logloss",
        gate: Optional[UncertaintyGate] = None,
        cv: int = 3,
        random_state: int = 0,
    ) -> None:
        self.members = members
        self.weighting = weighting
        self.gate = gate
        self.cv = cv
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _proba(self, est: BaseEstimator, X) -> np.ndarray:
        if hasattr(est, "predict_proba"):
            return np.asarray(est.predict_proba(X), dtype=np.float64)
        d = np.asarray(est.decision_function(X), dtype=np.float64)
        if d.ndim == 1:
            d = np.column_stack([-d, d])
        d = d - d.max(axis=1, keepdims=True)
        e = np.exp(d)
        return e / e.sum(axis=1, keepdims=True)

    def _align(self, est: BaseEstimator, P: np.ndarray) -> np.ndarray:
        """Map a member's class order onto the brain's global class order."""
        cls = list(getattr(est, "classes_", self.classes_))
        if list(cls) == list(self.classes_):
            return P
        out = np.zeros((P.shape[0], len(self.classes_)), dtype=np.float64)
        idx = {c: i for i, c in enumerate(self.classes_)}
        for j, c in enumerate(cls):
            if c in idx:
                out[:, idx[c]] = P[:, j]
        return out

    # ------------------------------------------------------------------
    def fit(self, X, y, sparse_input: Optional[bool] = None) -> "EnsembleBrain":
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if sparse_input is None:
            sparse_input = hasattr(X, "tocsr")
        specs = self.members or default_members(
            sparse_input, self.random_state, nonnegative=_is_nonnegative(X))
        self.member_names_ = list(specs.keys())
        self.members_: Dict[str, BaseEstimator] = {}
        self.fit_seconds_: Dict[str, float] = {}

        for name, est in specs.items():
            m = clone(est)
            t0 = time.perf_counter()
            m.fit(X, y)
            self.fit_seconds_[name] = time.perf_counter() - t0
            self.members_[name] = m

        self.weights_ = self._fit_weights(X, y, specs)
        return self

    def _fit_weights(self, X, y, specs) -> np.ndarray:
        n = len(self.member_names_)
        if self.weighting == "uniform":
            return np.ones(n) / n
        # out-of-fold scores
        from sklearn.metrics import accuracy_score, log_loss
        skf = StratifiedKFold(n_splits=max(self.cv, 2), shuffle=True,
                              random_state=self.random_state)
        scores = np.zeros(n)
        for tr, te in skf.split(np.zeros(len(y)), y):
            Xtr = X[tr]; Xte = X[te]
            for i, name in enumerate(self.member_names_):
                m = clone(specs[name]).fit(Xtr, y[tr])
                P = self._align(m, self._proba(m, Xte))
                if self.weighting == "accuracy":
                    scores[i] += accuracy_score(y[te], self.classes_[P.argmax(1)])
                else:
                    scores[i] += -log_loss(y[te], P, labels=list(self.classes_))
        scores /= skf.get_n_splits()
        s = scores - scores.min() + 1e-6
        return s / s.sum()

    # ------------------------------------------------------------------
    def member_proba(self, X) -> np.ndarray:
        """(n_members, n_samples, n_classes) posteriors."""
        return np.stack([
            self._align(self.members_[n], self._proba(self.members_[n], X))
            for n in self.member_names_
        ])

    def predict_proba(self, X) -> np.ndarray:
        M = self.member_proba(X)
        w = self.weights_.reshape(-1, 1, 1)
        P = (M * w).sum(axis=0)
        return P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    # ------------------------------------------------------------------
    def report(self, X) -> List[BrainReport]:
        """Full per-instance decision reports with uncertainty."""
        t0 = time.perf_counter()
        M = self.member_proba(X)
        w = self.weights_.reshape(-1, 1, 1)
        P = (M * w).sum(axis=0)
        P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
        elapsed = (time.perf_counter() - t0) * 1000.0
        n = P.shape[0]
        per = elapsed / max(n, 1)
        return [
            BrainReport(posterior=P[i], member_probs=M[:, i, :],
                        stats=decompose(M[:, i, :]),
                        labels=list(self.classes_), latency_ms=per)
            for i in range(n)
        ]

    def decide(self, X) -> BrainReport:
        """Convenience for a single instance."""
        return self.report(X)[0]

    # ------------------------------------------------------------------
    def calibrate_gate(self, X, y, target_precision: float = 0.90) -> UncertaintyGate:
        g = self.gate or UncertaintyGate()
        g.calibrate(self.member_proba(X), np.asarray(y),
                    target_precision=target_precision)
        self.gate = g
        return g

    def footprint_bytes(self) -> Dict[str, int]:
        """Serialised size of each member -- the edge deployment budget."""
        import io, joblib
        out = {}
        for n, m in self.members_.items():
            buf = io.BytesIO()
            joblib.dump(m, buf, compress=3)
            out[n] = buf.tell()
        out["_total"] = sum(v for k, v in out.items() if not k.startswith("_"))
        return out
