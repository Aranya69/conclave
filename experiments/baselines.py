"""Baseline agents used by the comparison experiments.

Kept in their own importable module so trained baselines can be pickled and
re-loaded in a fresh process (Experiment 7 measures each agent's peak RAM
that way).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from conclave.brain.ensemble import BrainReport
from conclave.brain.uncertainty import decompose


class HardVoteBrain(BaseEstimator, ClassifierMixin):
    """Majority vote over members (ties broken by mean probability).

    Hard voting keeps only each member's top choice, so the spread of the
    members' probabilities -- the signal CONCLAVE uses -- is discarded.
    The only uncertainty left is how split the vote is.
    """

    def __init__(self, members=None):
        self.members = members

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.member_names_ = list(self.members)
        self.members_ = {n: clone(e).fit(X, y) for n, e in self.members.items()}
        return self

    def member_proba(self, X):
        out = []
        for n in self.member_names_:
            m = self.members_[n]
            P = np.asarray(m.predict_proba(X), dtype=np.float64)
            idx = {c: i for i, c in enumerate(self.classes_)}
            Q = np.zeros((P.shape[0], len(self.classes_)))
            for j, c in enumerate(m.classes_):
                Q[:, idx[c]] = P[:, j]
            out.append(Q)
        return np.stack(out)

    def _votes(self, M):
        m, n, k = M.shape
        V = np.zeros((n, k))
        for j in range(m):
            V[np.arange(n), M[j].argmax(axis=1)] += 1.0
        V += 1e-6 * M.mean(axis=0)              # tie-break by mean probability
        return V / V.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        return self._votes(self.member_proba(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def report(self, X):
        M = self.member_proba(X)
        V = self._votes(M)
        return [BrainReport(posterior=V[i], member_probs=M[:, i, :],
                            stats=decompose(M[:, i, :]), labels=list(self.classes_))
                for i in range(V.shape[0])]
