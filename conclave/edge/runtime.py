"""NumPy-only edge runtime for a trained CONCLAVE agent.

Training a CONCLAVE agent needs scikit-learn and SciPy.  *Running* one does
not.  Once the committee is fitted, every member of the default committee
(logistic regression, SGD with modified-Huber loss, naive Bayes) is a linear
function of the hashed TF-IDF features, so a decision is a handful of hash
computations, two sparse-times-dense products and a softmax.

``export_agent`` (run on the training machine, where scikit-learn is
available) writes the fitted featuriser, committee and gate into one
``.npz`` file of plain arrays.  ``EdgeAgent`` (run on the board) loads that
file and reproduces the full agent's decision -- same features, same member
posteriors, same uncertainty decomposition, same gate -- using only NumPy and
the Python standard library.

This file deliberately imports nothing from the rest of the package, so it
can be copied to a device on its own:  ``runtime.py`` + ``agent.npz`` is the
entire deployment.  Removing SciPy and scikit-learn from the process is what
brings peak memory from ~217 MB down to the size of the interpreter plus
NumPy (Experiment 8).
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12
_TOKEN = re.compile(r"(?u)\b\w\w+\b")          # scikit-learn's default token_pattern
_WHITE = re.compile(r"\s\s+")


# --------------------------------------------------------------- hashing
def murmurhash3_32(key: bytes, seed: int = 0) -> int:
    """MurmurHash3 x86 32-bit, signed -- identical to scikit-learn's hasher."""
    c1, c2 = 0xCC9E2D51, 0x1B873593
    length = len(key)
    h1 = seed & 0xFFFFFFFF
    end = length & ~3
    for i in range(0, end, 4):
        k1 = key[i] | (key[i + 1] << 8) | (key[i + 2] << 16) | (key[i + 3] << 24)
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF
    k1 = 0
    tail = length & 3
    if tail == 3:
        k1 ^= key[end + 2] << 16
    if tail >= 2:
        k1 ^= key[end + 1] << 8
    if tail >= 1:
        k1 ^= key[end]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= h1 >> 16
    return h1 - 0x100000000 if h1 & 0x80000000 else h1


def _bucket(token: str, n: int) -> int:
    h = murmurhash3_32(token.encode("utf-8"), 0)
    if h == -2147483648:                         # abs(-2**31) is undefined
        return (2147483647 - (n - 1)) % n
    return abs(h) % n


def _strip_accents(s: str) -> str:
    try:
        s.encode("ASCII", errors="strict")
        return s
    except UnicodeEncodeError:
        nf = unicodedata.normalize("NFKD", s)
        return "".join(c for c in nf if not unicodedata.combining(c))


def word_ngrams(text: str, ngram: Tuple[int, int], strip_accents: bool) -> List[str]:
    doc = text.lower()
    if strip_accents:
        doc = _strip_accents(doc)
    toks = _TOKEN.findall(doc)
    lo, hi = ngram
    out = list(toks) if lo == 1 else []
    for n in range(max(lo, 2), min(hi, len(toks)) + 1):
        out.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return out


def char_wb_ngrams(text: str, ngram: Tuple[int, int]) -> List[str]:
    doc = _WHITE.sub(" ", text.lower())
    lo, hi = ngram
    out: List[str] = []
    for w in doc.split():
        w = " " + w + " "
        L = len(w)
        for n in range(lo, hi + 1):
            off = 0
            out.append(w[off:off + n])
            while off + n < L:
                off += 1
                out.append(w[off:off + n])
            if off == 0:                         # a short word is counted once
                break
    return out


# ---------------------------------------------------------------- export
def export_agent(path: str, featurizer, brain, gate=None) -> Dict[str, Any]:
    """Write a fitted FusedFeaturizer + EnsembleBrain (+ gate) to ``path``.

    Only linear members are supported (the default committee).  Weights of
    hashed columns that never occurred in training are exactly zero for the
    discriminative members and a per-class constant for naive Bayes, so only
    the columns seen in training are stored.
    """
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

    text, tab = featurizer.text, featurizer.tabular
    n_tab = len(tab.keys_) if tab is not None else 0
    nw = text.n_word_features if text is not None else 0
    nc = text.n_char_features if text is not None else 0
    D = n_tab + nw + nc
    classes = [str(c) for c in brain.classes_]
    K = len(classes)
    if K < 3:
        raise NotImplementedError("the edge runtime exports multi-class agents "
                                  "(three or more actions)")

    # columns that carry any discriminative weight
    seen = np.zeros(D, dtype=bool)
    seen[:n_tab] = True
    kinds = []
    for name in brain.member_names_:
        m = brain.members_[name]
        if isinstance(m, (LogisticRegression, SGDClassifier)):
            seen |= np.any(m.coef_ != 0, axis=0)
        elif isinstance(m, (BernoulliNB, ComplementNB, MultinomialNB)):
            # a column never active in training has the same log-probability
            # pattern as every other such column, so one vector covers them all
            seen |= np.any(m.feature_count_ > 0, axis=0)
        else:
            raise TypeError(f"member {name!r} ({type(m).__name__}) is not linear; "
                            "the edge runtime supports linear members only")
    cols = np.flatnonzero(seen)
    colmap = np.full(D, -1, dtype=np.int32)
    colmap[cols] = np.arange(cols.size, dtype=np.int32)

    arrays: Dict[str, np.ndarray] = {"colmap": colmap}
    for i, name in enumerate(brain.member_names_):
        m = brain.members_[name]
        order = [list(map(str, m.classes_)).index(c) for c in classes]
        if isinstance(m, (LogisticRegression, SGDClassifier)):
            coef, icpt = m.coef_[order], m.intercept_[order]
            unseen = np.flatnonzero(~seen)
            if unseen.size and np.any(coef[:, unseen] != 0):
                raise ValueError(f"{name}: non-zero weight on an unseen column")
            arrays[f"m{i}_W"] = coef[:, cols].astype(np.float64)
            arrays[f"m{i}_b"] = icpt.astype(np.float64)
            kind = ("logreg" if isinstance(m, LogisticRegression) else
                    f"sgd_{m.loss}")
        elif isinstance(m, BernoulliNB):
            fl = m.feature_log_prob_[order]
            neg = np.log(1.0 - np.exp(fl))
            delta = fl - neg
            unseen_idx = np.flatnonzero(~seen)
            arrays[f"m{i}_W"] = delta[:, cols]
            arrays[f"m{i}_u"] = (delta[:, unseen_idx[0]] if unseen_idx.size
                                 else np.zeros(K))
            arrays[f"m{i}_b"] = m.class_log_prior_[order] + neg.sum(axis=1)
            arrays[f"m{i}_thr"] = np.array([m.binarize if m.binarize is not None
                                            else -np.inf])
            kind = "bernoulli_nb"
        else:                                           # Complement / Multinomial
            fl = m.feature_log_prob_[order]
            unseen_idx = np.flatnonzero(~seen)
            arrays[f"m{i}_W"] = fl[:, cols]
            arrays[f"m{i}_u"] = (fl[:, unseen_idx[0]] if unseen_idx.size
                                 else np.zeros(K))
            prior = m.class_log_prior_[order]
            if isinstance(m, ComplementNB) and K > 1:
                prior = np.zeros(K)                     # ComplementNB ignores it
            arrays[f"m{i}_b"] = prior
            kind = "complement_nb" if isinstance(m, ComplementNB) else "multinomial_nb"
        kinds.append(kind)

    if text is not None:
        arrays["idf_w"] = text._tw.idf_.astype(np.float64)
        arrays["idf_c"] = text._tc.idf_.astype(np.float64)
    if tab is not None:
        arrays["tab_mean"] = np.asarray(tab.mean_, dtype=np.float32)
        arrays["tab_scale"] = np.asarray(tab.scale_, dtype=np.float32)
    arrays["weights"] = np.asarray(brain.weights_, dtype=np.float64)

    g = gate
    spec = {
        "version": 1,
        "classes": classes,
        "member_kinds": kinds,
        "member_names": list(brain.member_names_),
        "tab_keys": list(tab.keys_) if tab is not None else [],
        "tab_fill": float(tab.fill) if tab is not None else 0.0,
        "n_word": int(nw), "n_char": int(nc),
        "word_ngram": list(text.word_ngrams) if text is not None else [1, 1],
        "char_ngram": list(text.char_ngrams) if text is not None else [3, 3],
        "strip_accents": True,
        "sublinear_tf": True,
        "gate": None if g is None else {
            "epistemic_max": g.epistemic_max, "aleatoric_max": g.aleatoric_max,
            "confidence_min": g.confidence_min, "margin_min": g.margin_min,
            "allow_gather": bool(g.allow_gather)},
    }
    arrays["spec"] = np.frombuffer(json.dumps(spec).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **arrays)
    return {"columns_kept": int(cols.size), "columns_total": int(D), "spec": spec}


# --------------------------------------------------------------- runtime
class EdgeAgent:
    """Loads an exported agent and decides with NumPy alone."""

    def __init__(self, path: str, cache_size: int = 50000) -> None:
        z = np.load(path)
        self.spec = json.loads(bytes(z["spec"]).decode("utf-8"))
        s = self.spec
        self.classes: List[str] = s["classes"]
        self.K = len(self.classes)
        self.colmap = z["colmap"]
        self.n_tab = len(s["tab_keys"])
        self.nw, self.nc = s["n_word"], s["n_char"]
        self.keys = s["tab_keys"]
        self.fill = s["tab_fill"]
        self.tab_mean = z["tab_mean"] if self.n_tab else None
        self.tab_scale = z["tab_scale"] if self.n_tab else None
        self.idf_w = z["idf_w"] if self.nw else None
        self.idf_c = z["idf_c"] if self.nc else None
        self.weights = z["weights"]
        self.members = []
        for i, kind in enumerate(s["member_kinds"]):
            m = {"kind": kind, "W": z[f"m{i}_W"], "b": z[f"m{i}_b"]}
            if f"m{i}_u" in z.files:
                m["u"] = z[f"m{i}_u"]
            if f"m{i}_thr" in z.files:
                m["thr"] = float(z[f"m{i}_thr"][0])
            self.members.append(m)
        self.gate = s.get("gate")
        self._cache_w: Dict[str, int] = {}
        self._cache_c: Dict[str, int] = {}
        self._cache_size = cache_size

    # ------------------------------------------------------------- features
    def _hashed(self, grams: Sequence[str], n: int, cache: Dict[str, int]) -> Dict[int, float]:
        counts: Dict[int, float] = {}
        for g in grams:
            j = cache.get(g)
            if j is None:
                j = _bucket(g, n)
                if len(cache) < self._cache_size:
                    cache[g] = j
            counts[j] = counts.get(j, 0.0) + 1.0
        return counts

    @staticmethod
    def _tfidf(counts: Dict[int, float], idf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not counts:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        val = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        val = (np.log(val) + 1.0) * idf[idx]
        nrm = math.sqrt(float(val @ val))
        if nrm > 0:
            val = val / nrm
        return idx, val

    def features(self, features: Mapping[str, float], text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Sparse feature vector as (column indices, values)."""
        idx_parts, val_parts = [], []
        if self.n_tab:
            raw = np.array([float(features.get(k, self.fill)) for k in self.keys],
                           dtype=np.float32)
            t = ((raw - self.tab_mean) / self.tab_scale).astype(np.float32)
            nz = np.flatnonzero(t)
            idx_parts.append(nz.astype(np.int64))
            val_parts.append(t[nz].astype(np.float64))
        if self.nw:
            s = self.spec
            wi, wv = self._tfidf(self._hashed(
                word_ngrams(text, tuple(s["word_ngram"]), s["strip_accents"]),
                self.nw, self._cache_w), self.idf_w)
            ci, cv = self._tfidf(self._hashed(
                char_wb_ngrams(text, tuple(s["char_ngram"])),
                self.nc, self._cache_c), self.idf_c)
            idx_parts += [wi + self.n_tab, ci + self.n_tab + self.nw]
            val_parts += [wv, cv]
        idx = np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.int64)
        val = np.concatenate(val_parts) if val_parts else np.zeros(0)
        return idx, val

    # --------------------------------------------------------------- members
    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def member_proba(self, idx: np.ndarray, val: np.ndarray) -> np.ndarray:
        local = self.colmap[idx]
        seen = local >= 0
        li, lv = local[seen], val[seen]
        n_unseen = int(np.count_nonzero(~seen))
        out = np.empty((len(self.members), self.K))
        for r, m in enumerate(self.members):
            kind = m["kind"]
            if kind == "bernoulli_nb":
                on = lv > m["thr"]
                n_unseen_on = int(np.count_nonzero(val[~seen] > m["thr"]))
                jll = m["b"] + m["W"][:, li[on]].sum(axis=1) + n_unseen_on * m["u"]
                out[r] = self._softmax(jll)
            elif kind in ("complement_nb", "multinomial_nb"):
                jll = m["b"] + m["W"][:, li] @ lv + float(val[~seen].sum()) * m["u"]
                out[r] = self._softmax(jll)
            else:
                z = m["b"] + m["W"][:, li] @ lv
                if kind == "logreg":
                    out[r] = self._softmax(z)
                elif kind == "sgd_modified_huber":
                    p = (np.clip(z, -1.0, 1.0) + 1.0) / 2.0
                    s = p.sum()
                    out[r] = p / s if s > 0 else np.full(self.K, 1.0 / self.K)
                elif kind == "sgd_log_loss":
                    p = 1.0 / (1.0 + np.exp(-z))
                    s = p.sum()
                    out[r] = p / s if s > 0 else np.full(self.K, 1.0 / self.K)
                else:
                    raise ValueError(f"unsupported member kind {kind}")
        return out

    # -------------------------------------------------------------- decision
    @staticmethod
    def _entropy(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, _EPS, 1.0)
        return -(p * np.log(p)).sum(axis=-1)

    def decide(self, features: Mapping[str, float], text: str = "") -> Dict[str, Any]:
        idx, val = self.features(features, text)
        M = self.member_proba(idx, val)
        P = (M * self.weights[:, None]).sum(axis=0)
        P = P / max(P.sum(), _EPS)
        mean = M.mean(axis=0)                      # uncertainty uses the plain mean
        logk = max(math.log(max(self.K, 2)), _EPS)
        H_mean = float(self._entropy(mean))
        mean_H = float(self._entropy(M).mean())
        epi = max(H_mean - mean_H, 0.0) / logk
        ale = mean_H / logk
        srt = np.sort(mean)[::-1]
        stats = {"confidence": float(mean.max()),
                 "margin": float(srt[0] - srt[1]) if self.K > 1 else 1.0,
                 "total": H_mean / logk, "aleatoric": ale, "epistemic": epi}
        outcome = "act"
        g = self.gate
        if g is not None:
            if epi > g["epistemic_max"]:
                outcome = "gather" if g["allow_gather"] else "escalate"
            elif ale > g["aleatoric_max"]:
                outcome = "abstain"
            elif stats["confidence"] < g["confidence_min"] or stats["margin"] < g["margin_min"]:
                outcome = "gather" if g["allow_gather"] else "escalate"
        return {"action": self.classes[int(np.argmax(P))], "posterior": P,
                "member_probs": M, "stats": stats, "gate": outcome}
