"""Uncertainty decomposition and the gate that turns it into control flow.

This module is the conceptual centre of CONCLAVE.  A soft-voting ensemble
gives us, for free, a decomposition that a single language model cannot
provide without expensive sampling:

    total uncertainty  =  aleatoric  +  epistemic

* **Aleatoric** (``expected_entropy``): the average entropy of individual
  members.  Members agree that the situation is genuinely ambiguous --
  more data will not help.
* **Epistemic** (``mutual_information``, a.k.a. *disagreement*): the gap
  between the entropy of the mean posterior and the mean of member
  entropies.  Members disagree; the agent is outside its competence and
  should gather evidence or escalate.

This is the classical BALD decomposition (Houlsby et al., 2011) applied
as an agent control signal.  The practical payoff: an agent that knows
the difference between "this is a hard call" and "I do not know", and
routes them to different behaviours.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

_EPS = 1e-12


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in nats, numerically safe."""
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def normalized_entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Entropy scaled to [0, 1] by log(n_classes)."""
    k = np.asarray(p).shape[axis]
    return entropy(p, axis=axis) / max(np.log(max(k, 2)), _EPS)


@dataclass(frozen=True)
class UncertaintyStats:
    """Full epistemic picture for one decision."""

    confidence: float       # max posterior probability
    margin: float           # top-1 minus top-2 probability
    total: float            # normalized entropy of the mean posterior
    aleatoric: float        # normalized mean of member entropies
    epistemic: float        # normalized mutual information (disagreement)
    vote_entropy: float     # normalized entropy of hard-vote distribution
    n_members: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "confidence": self.confidence, "margin": self.margin,
            "total": self.total, "aleatoric": self.aleatoric,
            "epistemic": self.epistemic, "vote_entropy": self.vote_entropy,
        }


def decompose(member_probs: np.ndarray) -> UncertaintyStats:
    """Decompose ensemble uncertainty.

    Parameters
    ----------
    member_probs : array, shape (n_members, n_classes)
        Posterior from each ensemble member for a *single* instance.
    """
    P = np.atleast_2d(np.asarray(member_probs, dtype=np.float64))
    P = P / np.clip(P.sum(axis=1, keepdims=True), _EPS, None)
    m, k = P.shape
    mean = P.mean(axis=0)

    H_mean = entropy(mean)                       # total
    mean_H = float(entropy(P, axis=1).mean())    # aleatoric
    mi = max(H_mean - mean_H, 0.0)               # epistemic (BALD)
    logk = max(np.log(max(k, 2)), _EPS)

    srt = np.sort(mean)[::-1]
    margin = float(srt[0] - srt[1]) if k > 1 else 1.0

    hard = np.bincount(P.argmax(axis=1), minlength=k).astype(np.float64)
    hard /= max(hard.sum(), _EPS)

    return UncertaintyStats(
        confidence=float(mean.max()),
        margin=margin,
        total=float(H_mean / logk),
        aleatoric=float(mean_H / logk),
        epistemic=float(mi / logk),
        vote_entropy=float(entropy(hard) / logk),
        n_members=m,
    )


def decompose_batch(member_probs: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorised :func:`decompose` over many instances at once.

    Shape ``(n_members, n_samples, n_classes)`` in, arrays of length
    ``n_samples`` out.  Calibration and risk-coverage analysis run over
    tens of thousands of instances, where a per-instance Python loop
    dominates the cost; this keeps those paths O(1) in interpreter
    overhead.  Results agree with :func:`decompose` to floating-point
    tolerance.
    """
    P = np.asarray(member_probs, dtype=np.float64)
    if P.ndim != 3:
        raise ValueError("expected (n_members, n_samples, n_classes)")
    P = P / np.clip(P.sum(axis=2, keepdims=True), _EPS, None)
    m, n, k = P.shape
    mean = P.mean(axis=0)                                  # (n, k)
    logk = max(np.log(max(k, 2)), _EPS)

    H_mean = entropy(mean, axis=1)                          # total
    mean_H = entropy(P, axis=2).mean(axis=0)                # aleatoric
    mi = np.maximum(H_mean - mean_H, 0.0)                   # epistemic

    srt = np.sort(mean, axis=1)[:, ::-1]
    margin = (srt[:, 0] - srt[:, 1]) if k > 1 else np.ones(n)

    hard = P.argmax(axis=2)                                 # (m, n)
    counts = np.zeros((n, k))
    for j in range(m):
        counts[np.arange(n), hard[j]] += 1.0
    counts /= np.clip(counts.sum(axis=1, keepdims=True), _EPS, None)

    return {
        "confidence": mean.max(axis=1),
        "margin": margin,
        "total": H_mean / logk,
        "aleatoric": mean_H / logk,
        "epistemic": mi / logk,
        "vote_entropy": entropy(counts, axis=1) / logk,
    }


class GateOutcome(str, Enum):
    ACT = "act"
    GATHER = "gather"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"


@dataclass
class UncertaintyGate:
    """Maps an :class:`UncertaintyStats` to a control decision.

    The ordering of checks encodes a safety policy:

    1. High **epistemic** uncertainty -> the agent is out of distribution.
       Escalate to a human if we cannot cheaply gather more evidence.
    2. High **aleatoric** uncertainty -> genuinely ambiguous.  More sensing
       will not help; abstain rather than guess.
    3. Low confidence or thin margin -> gather more evidence.
    4. Otherwise act.

    Thresholds are calibrated on held-out data by :meth:`calibrate`, not
    hand-tuned, which keeps the framework honest.
    """

    epistemic_max: float = 0.25
    aleatoric_max: float = 0.60
    confidence_min: float = 0.50
    margin_min: float = 0.10
    allow_gather: bool = True

    def __call__(self, s: UncertaintyStats) -> GateOutcome:
        if s.epistemic > self.epistemic_max:
            return GateOutcome.GATHER if self.allow_gather else GateOutcome.ESCALATE
        if s.aleatoric > self.aleatoric_max:
            return GateOutcome.ABSTAIN
        if s.confidence < self.confidence_min or s.margin < self.margin_min:
            return GateOutcome.GATHER if self.allow_gather else GateOutcome.ESCALATE
        return GateOutcome.ACT

    # ------------------------------------------------------------------
    def calibrate(
        self,
        member_probs: np.ndarray,
        y_true: np.ndarray,
        target_precision: float = 0.90,
        grid: int = 24,
    ) -> "UncertaintyGate":
        """Pick thresholds so that *acted-on* decisions hit a precision target.

        This is selective prediction: we choose the largest coverage whose
        accuracy-on-accepted meets ``target_precision``.  Reporting the
        resulting risk-coverage curve is far more informative than a
        single accuracy number for a safety-critical agent.

        Parameters
        ----------
        member_probs : array, shape (n_members, n_samples, n_classes)
        y_true : array, shape (n_samples,)
        """
        M = np.asarray(member_probs, dtype=np.float64)
        if M.ndim != 3:
            raise ValueError("member_probs must be (n_members, n_samples, n_classes)")
        b = decompose_batch(M)
        pred = M.mean(axis=0).argmax(axis=1)
        correct = (pred == np.asarray(y_true)).astype(float)
        conf, epi = b["confidence"], b["epistemic"]

        best = None
        for c_thr in np.linspace(0.0, 0.95, grid):
            keep_c = conf >= c_thr
            for e_thr in np.linspace(0.02, 0.80, grid):
                acc_mask = keep_c & (epi <= e_thr)
                n_acc = int(acc_mask.sum())
                if n_acc < max(0.05 * len(conf), 1):
                    continue
                prec = float(correct[acc_mask].sum() / n_acc)
                cov = n_acc / len(conf)
                if prec >= target_precision and (best is None or cov > best[0]):
                    best = (cov, prec, float(c_thr), float(e_thr))
        if best is not None:
            _, _, c_thr, e_thr = best
            self.confidence_min = float(c_thr)
            self.epistemic_max = float(e_thr)
        self.calibration_ = {
            "target_precision": target_precision,
            "achieved": None if best is None else
                        {"coverage": best[0], "precision": best[1]},
            "confidence_min": self.confidence_min,
            "epistemic_max": self.epistemic_max,
        }
        return self


def risk_coverage_curve(
    member_probs: np.ndarray, y_true: np.ndarray, n_points: int = 25
) -> Dict[str, List[float]]:
    """Selective-prediction curve driven by ensemble confidence.

    Returns coverage (fraction acted on) against accuracy on that subset.
    A well-behaved agent shows accuracy rising monotonically as coverage
    falls -- evidence that its uncertainty estimates are meaningful.
    """
    M = np.asarray(member_probs, dtype=np.float64)
    mean = M.mean(axis=0)
    pred = mean.argmax(axis=1)
    correct = (pred == np.asarray(y_true)).astype(float)
    score = decompose_batch(M)["confidence"]
    order = np.argsort(-score)
    cov, acc = [], []
    n = len(order)
    for q in np.linspace(0.05, 1.0, n_points):
        k = max(int(q * n), 1)
        sel = order[:k]
        cov.append(k / n)
        acc.append(float(correct[sel].mean()))
    return {"coverage": cov, "accuracy": acc,
            "aurc": float(np.trapezoid(acc, cov)) if hasattr(np, "trapezoid")
                    else float(np.trapz(acc, cov))}
