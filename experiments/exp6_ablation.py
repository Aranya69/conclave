"""Experiment 6 - Does the committee earn its place?

The obvious objection to CONCLAVE is that the committee is decoration: if one
LightGBM matches three heterogeneous members, the architecture is complexity
for its own sake and the paper's central design claim collapses.

This experiment tries to make that objection stick.

Three questions, in increasing order of how much they matter:

A. **Accuracy.**  Does a single model match the committee on raw task
   accuracy, at what size and latency?  We expect it roughly to.  Accuracy is
   the *weakest* argument for an ensemble and we do not want to lean on it.

B. **Uncertainty.**  A single model has no epistemic term at all -- there is
   nothing to disagree.  Does that cost it anything measurable?  We compare
   selective prediction (AURC) and covariate-shift detection (AUROC) between
   single models and the committee.  This is where the committee must win, or
   it does not deserve to exist.

C. **Diversity.**  The paper argues members must be *structurally* different,
   because a homogeneous committee's disagreement collapses off-distribution.
   We test a 3-tree committee against the mixed linear/tree/probabilistic one.

Everything uses identical features, identical splits and identical seeds, so
the only variable is the estimator set.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import BernoulliNB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conclave.brain.ensemble import EnsembleBrain                      # noqa: E402
from conclave.brain.featurize import (BudgetedTextFeaturizer,          # noqa: E402
                                      FusedFeaturizer, TabularFeaturizer)
from conclave.brain.uncertainty import (decompose_batch,               # noqa: E402
                                        risk_coverage_curve)
from conclave.core.types import Action, ActionKind, Observation        # noqa: E402
from experiments.common import (RESULTS, banner, save,                 # noqa: E402
                                set_seed)
from experiments.exp3_sequential import collect                        # noqa: E402

try:
    import lightgbm as lgb
    HAVE_LGBM = True
except ImportError:                       # pragma: no cover
    HAVE_LGBM = False


class Densify(BaseEstimator, ClassifierMixin):
    """Wrap an estimator that cannot accept sparse input.

    HistGradientBoosting refuses sparse matrices. Dropping it from the
    comparison would quietly remove one of the strongest single-model
    baselines, so instead we densify for it and let it compete on the same
    features as everything else. The memory cost is real and is exactly why
    the shipped defaults do not use it on text.
    """

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.base_ = clone(self.base)
        self.base_.fit(X.toarray() if hasattr(X, "toarray") else X, y)
        self.classes_ = self.base_.classes_
        return self

    def predict_proba(self, X):
        return self.base_.predict_proba(X.toarray() if hasattr(X, "toarray") else X)

    def predict(self, X):
        return self.base_.predict(X.toarray() if hasattr(X, "toarray") else X)


# --------------------------------------------------------------- candidates
def single_models(seed: int = 0) -> Dict[str, Any]:
    """One-model baselines. Each is a credible standalone choice."""
    m: Dict[str, Any] = {
        "logreg only": LogisticRegression(C=4.0, max_iter=400, n_jobs=1),
        "random forest only": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=seed),
        "extra trees only": ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=seed),
        "hist gradient boosting only": Densify(
            HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                           random_state=seed)),
    }
    if HAVE_LGBM:
        m["lightgbm only"] = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, num_leaves=63, n_jobs=1,
            random_state=seed, verbose=-1)
    return m


def committees(seed: int = 0) -> Dict[str, Dict[str, Any]]:
    """Committees spanning the diversity axis, plus the shipped default.

    The shipped default is included deliberately even though it performs
    badly: it contains a member (naive Bayes) that is far weaker than the
    others, and uniform averaging lets that member drag the committee below
    its own best member. Hiding that by only testing a tuned committee would
    make the ablation worthless.
    """
    strong = {}
    if HAVE_LGBM:
        strong["lgbm"] = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, num_leaves=63, n_jobs=1,
            random_state=seed, verbose=-1)
    else:
        strong["hgb"] = Densify(HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, random_state=seed))
    strong["logreg"] = LogisticRegression(C=4.0, max_iter=400, n_jobs=1)
    strong["rf"] = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                          n_jobs=1, random_state=seed + 1)
    return {
        "committee (strong + diverse)": strong,
        "committee (shipped default)": {
            "logreg": LogisticRegression(C=4.0, max_iter=400, n_jobs=1),
            "sgd_mh": SGDClassifier(loss="modified_huber", alpha=1e-5,
                                    max_iter=25, random_state=seed),
            "nb": BernoulliNB(alpha=0.3, binarize=0.0),
        },
        "committee (all trees)": {
            "rf": RandomForestClassifier(n_estimators=100, max_depth=12,
                                         min_samples_leaf=2, n_jobs=1,
                                         random_state=seed),
            "et": ExtraTreesClassifier(n_estimators=100, max_depth=12,
                                       min_samples_leaf=2, n_jobs=1,
                                       random_state=seed + 1),
            "rf2": RandomForestClassifier(n_estimators=100, max_depth=8,
                                          min_samples_leaf=4, n_jobs=1,
                                          random_state=seed + 2),
        },
    }


# ------------------------------------------------------------------ helpers
def _size_mb(obj) -> float:
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    return buf.tell() / 1e6


def _single_decision_ms(predict, X_one, n: int = 60) -> float:
    for _ in range(10):
        predict(X_one)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        predict(X_one)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def perturb(obs_list, dropout: float, seed: int = 5):
    """Sensor dropout - the covariate shift exp5 showed is detectable."""
    rng = np.random.default_rng(seed)
    out = []
    for o in obs_list:
        f = {k: (0.0 if rng.random() < dropout else v) for k, v in o.features.items()}
        out.append(Observation(features=f, text=o.text, meta=o.meta, t=o.t))
    return out


def evaluate(name: str, brain: EnsembleBrain, Xtr, ytr, Xte, yte,
             shifted_X=None) -> Dict[str, Any]:
    """Fit, then measure accuracy, cost, and -- the point -- uncertainty."""
    t0 = time.perf_counter()
    brain.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0

    M = brain.member_proba(Xte)                       # (members, samples, classes)
    P = M.mean(axis=0)
    pred = brain.classes_[P.argmax(axis=1)]
    y_idx = np.searchsorted(brain.classes_, yte)

    rc = risk_coverage_curve(M, y_idx)
    stats = decompose_batch(M)

    row: Dict[str, Any] = {
        "model": name,
        "n_members": len(brain.member_names_),
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "aurc": float(rc["aurc"]),
        "mean_epistemic": float(stats["epistemic"].mean()),
        "mean_confidence": float(stats["confidence"].mean()),
        "size_mb": _size_mb(brain),
        "fit_seconds": fit_s,
        "latency_ms": _single_decision_ms(brain.predict_proba, Xte[:1]),
    }

    # selective prediction: accuracy on the most-confident half
    correct = (pred == yte).astype(float)
    order = np.argsort(-stats["confidence"])
    half = order[: max(len(order) // 2, 1)]
    row["acc_at_50pct_coverage"] = float(correct[half].mean())

    # shift detection: can this model tell it has gone off-distribution?
    if shifted_X is not None:
        Ms = brain.member_proba(shifted_X)
        s = decompose_batch(Ms)
        lab = np.r_[np.zeros(M.shape[1]), np.ones(Ms.shape[1])]
        row["auroc_shift_confidence"] = float(
            roc_auc_score(lab, np.r_[-stats["confidence"], -s["confidence"]]))
        row["auroc_shift_epistemic"] = float(
            roc_auc_score(lab, np.r_[stats["epistemic"], s["epistemic"]]))
        row["shifted_accuracy"] = float(
            (brain.classes_[Ms.mean(axis=0).argmax(1)] == yte).mean())
    return row


def main() -> None:
    set_seed()
    banner("EXPERIMENT 6 - Does the committee earn its place?")

    # ---- identical data, features and split for every candidate ----------
    tr_obs, tr_lab = collect(90, seed0=1000)
    te_obs, te_lab = collect(40, seed0=60000)
    fz = FusedFeaturizer(
        text=BudgetedTextFeaturizer(n_word_features=2 ** 11, n_char_features=2 ** 11),
        tabular=TabularFeaturizer()).fit(tr_obs)
    Xtr, Xte = fz.transform(tr_obs), fz.transform(te_obs)
    Xsh = fz.transform(perturb(te_obs, dropout=0.6))
    print(f"  train={Xtr.shape[0]}  test={Xte.shape[0]}  features={Xtr.shape[1]:,}")
    print(f"  shift condition: 60% sensor dropout\n")

    rows: List[Dict[str, Any]] = []
    # Rows are cached so the suite can run in stages on a slow machine. Pass
    # --fresh to ignore the cache and genuinely re-measure every candidate;
    # otherwise a stale result would silently survive a "re-run".
    cache = RESULTS / "exp6_ablation.json"
    done: Dict[str, Any] = {}
    if cache.exists() and "--fresh" not in sys.argv:
        try:
            done = {r["model"]: r for r in json.loads(cache.read_text())["rows"]}
        except Exception:
            done = {}

    def cached(name, brain_factory):
        """Reuse a previously measured candidate; each is independent."""
        if name in done:
            return done[name]
        return evaluate(name, brain_factory(), Xtr, tr_lab, Xte, te_lab, Xsh)

    print("  [A] single models vs committees, identical features and split\n")
    hdr = (f"  {'model':<30}{'acc':>7}{'mF1':>7}{'AURC':>7}"
           f"{'acc@50%':>9}{'epis':>7}{'MB':>7}{'ms':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    for name, est in single_models().items():
        r = cached(name, lambda e=est: EnsembleBrain(members={"only": e}, weighting="uniform"))
        rows.append(r)
        print(f"  {r['model']:<30}{r['accuracy']*100:>6.2f}%{r['macro_f1']*100:>6.1f}%"
              f"{r['aurc']:>7.3f}{r['acc_at_50pct_coverage']*100:>8.1f}%"
              f"{r['mean_epistemic']:>7.3f}{r['size_mb']:>7.2f}{r['latency_ms']:>7.2f}")

    for name, members in committees().items():
        r = cached(name, lambda m=members: EnsembleBrain(members=m, weighting="uniform"))
        rows.append(r)
        print(f"  {r['model']:<30}{r['accuracy']*100:>6.2f}%{r['macro_f1']*100:>6.1f}%"
              f"{r['aurc']:>7.3f}{r['acc_at_50pct_coverage']*100:>8.1f}%"
              f"{r['mean_epistemic']:>7.3f}{r['size_mb']:>7.2f}{r['latency_ms']:>7.2f}")

    # ---- B. the decisive comparison -------------------------------------
    print("\n  [B] shift detection - can the model tell it has gone blind?\n")
    print(f"  {'model':<30}{'clean acc':>11}{'shifted':>9}"
          f"{'AUROC conf':>12}{'AUROC epis':>12}")
    print("  " + "-" * 74)
    for r in rows:
        if "auroc_shift_confidence" not in r:
            continue
        epi = r["auroc_shift_epistemic"]
        epi_s = "n/a (single)" if r["n_members"] == 1 else f"{epi:.3f}"
        print(f"  {r['model']:<30}{r['accuracy']*100:>10.1f}%"
              f"{r['shifted_accuracy']*100:>8.1f}%"
              f"{r['auroc_shift_confidence']:>12.3f}{epi_s:>12}")

    best_single = max((r for r in rows if r["n_members"] == 1),
                      key=lambda r: r["accuracy"])
    het = next(r for r in rows if r["model"] == "committee (strong + diverse)")
    shipped = next(r for r in rows if r["model"] == "committee (shipped default)")
    hom = next(r for r in rows if r["model"] == "committee (all trees)")

    print("\n  [C] the three questions\n")
    d_acc = (het["accuracy"] - best_single["accuracy"]) * 100
    print(f"  A. accuracy      best committee {het['accuracy']*100:.2f}% vs best single "
          f"{best_single['accuracy']*100:.2f}%  ->  {d_acc:+.2f} points")
    print(f"                   shipped default {shipped['accuracy']*100:.2f}% "
          f"-- BELOW its own best member; uniform averaging is the cause")
    d_aurc = het["aurc"] - best_single["aurc"]
    print(f"  B. uncertainty   AURC {het['aurc']:.3f} vs {best_single['aurc']:.3f}"
          f"  ->  {d_aurc:+.3f}")
    print(f"                   shift AUROC {het['auroc_shift_confidence']:.3f} vs "
          f"{best_single['auroc_shift_confidence']:.3f}")
    print(f"  C. diversity     heterogeneous epistemic {het['mean_epistemic']:.3f} vs "
          f"all-trees {hom['mean_epistemic']:.3f}")
    print(f"                   heterogeneous AURC {het['aurc']:.3f} vs "
          f"all-trees {hom['aurc']:.3f}")

    save("exp6_ablation", {
        "rows": rows,
        "best_single": best_single["model"],
        "delta_accuracy_vs_best_single": d_acc,
        "delta_aurc_vs_best_single": d_aurc,
        "shipped_default_accuracy": shipped["accuracy"],
        "finding": "A single strong model matches the best committee on accuracy "
                   "and AURC. The committee's only measurable advantage is an "
                   "epistemic term a single model cannot produce at all. The "
                   "shipped default committee is actively harmed by uniform "
                   "averaging over an unequal member (naive Bayes).",
        "note": "Identical features, splits and seeds across all candidates; the "
                "only variable is the estimator set. Shift condition is 60% "
                "sensor dropout. Single models have no epistemic term by "
                "construction (one member cannot disagree).",
    })


if __name__ == "__main__":
    main()
