"""Experiment 2 - Disaster field-report triage (HumAID).

Question
--------
The disaster case study needs a perception task with *real, human*
labels.  HumAID (Alam et al., ICWSM 2021) is 77k crisis tweets from 19
disasters annotated into 10 humanitarian categories -- exactly the
"incoming field report, which response channel does it belong to"
decision an offline triage agent must make.

We use the authors' official train/dev/test split, so our numbers sit
beside the published transformer baselines on the same data.

Honest comparison
-----------------
The transformer baselines were trained and evaluated by their authors on
this split; we quote their reported weighted-F1 rather than re-running
them, and we state the cost side (parameters, footprint) that motivates
the trade.  The claim is *not* that an ensemble beats RoBERTa.  It is
that it lands a few points behind while fitting in a fraction of the
memory and running without a GPU or a network.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conclave.brain.ensemble import EnsembleBrain                # noqa: E402
from conclave.brain.featurize import BudgetedTextFeaturizer      # noqa: E402
from conclave.brain.uncertainty import (UncertaintyGate,         # noqa: E402
                                     risk_coverage_curve)
from experiments.common import (DATA, RESULTS, banner, save,     # noqa: E402
                                set_seed)

# Reported by the dataset authors, Table 7 of Alam et al. (ICWSM 2021),
# row "All" -- the combined 10-class dataset, which is the split we use.
# NOTE: 0.781 appears in that table as the *Average* over per-event and
# event-type rows, NOT as the combined-dataset result.  Quoting it as the
# comparable baseline would overstate the gap; the correct figure is 0.760.
PUBLISHED = {
    "RoBERTa (125M)":    0.760,
    "BERT (110M)":       0.758,
    "DistilBERT (66M)":  0.758,
    "XLM-RoBERTa":       0.758,
    "FastText":          0.744,
    "SVM (classical)":   0.731,
    "Random Forest":     0.707,
}


def main() -> None:
    set_seed()
    banner("EXPERIMENT 2 - Disaster field-report triage on HumAID")

    tr = pd.read_parquet(DATA / "humaid" / "train.parquet")
    va = pd.read_parquet(DATA / "humaid" / "validation.parquet")
    te = pd.read_parquet(DATA / "humaid" / "test.parquet")
    print(f"  train={len(tr):,}  val={len(va):,}  test={len(te):,}  "
          f"classes={tr.class_label.nunique()}")

    out: Dict[str, Any] = {"published_baselines": PUBLISHED, "budgets": []}
    prev = RESULTS / "exp2_humaid.json"
    if prev.exists():                      # allow incremental accumulation
        import json as _j
        old = _j.loads(prev.read_text())
        out["budgets"] = [b for b in old.get("budgets", [])]

    # The feature budget is the deployment knob: how much hashed space can
    # the target board spare?  Sweeping it shows the accuracy/RAM frontier.
    bits_list = ([int(a) for a in sys.argv[1:]] if len(sys.argv) > 1
                 else [12, 14, 16])
    for bits in bits_list:
        n = 2 ** bits
        fz = BudgetedTextFeaturizer(n_word_features=n, n_char_features=n)
        t0 = time.perf_counter()
        Xtr = fz.fit(tr.tweet_text.tolist()).transform(tr.tweet_text.tolist())
        Xva = fz.transform(va.tweet_text.tolist())
        Xte = fz.transform(te.tweet_text.tolist())
        vec_s = time.perf_counter() - t0

        brain = EnsembleBrain(weighting="uniform")
        t0 = time.perf_counter()
        brain.fit(Xtr, tr.class_label.values)
        fit_s = time.perf_counter() - t0

        pred = brain.predict(Xte)
        y = te.class_label.values
        acc = accuracy_score(y, pred)
        wf1 = f1_score(y, pred, average="weighted")
        mf1 = f1_score(y, pred, average="macro")

        # single-decision latency (the number an edge agent experiences)
        one = te.tweet_text.iloc[:1].tolist()
        lat = []
        for _ in range(60):
            t0 = time.perf_counter()
            brain.predict_proba(fz.transform(one))
            lat.append((time.perf_counter() - t0) * 1000)

        buf = io.BytesIO(); joblib.dump({"fz": fz, "brain": brain}, buf, compress=3)
        size_mb = buf.tell() / 1e6

        # selective prediction on the official validation split
        gate = UncertaintyGate()
        gate.calibrate(brain.member_proba(Xva),
                       np.searchsorted(brain.classes_, va.class_label.values),
                       target_precision=0.90)
        rc = risk_coverage_curve(brain.member_proba(Xte),
                                 np.searchsorted(brain.classes_, y))

        rec = {
            "hash_bits": bits, "n_features": 2 * n,
            "accuracy": float(acc), "weighted_f1": float(wf1),
            "macro_f1": float(mf1),
            "median_latency_ms": float(np.median(lat)),
            "p99_latency_ms": float(np.percentile(lat, 99)),
            "model_mb": float(size_mb),
            "vectorize_seconds": vec_s, "fit_seconds": fit_s,
            "members": {k: float(v) for k, v in brain.fit_seconds_.items()},
            "gate_calibration": gate.calibration_,
            "aurc": rc["aurc"],
        }
        out["budgets"].append(rec)
        print(f"  2^{bits:<2d} feats  acc={acc*100:5.2f}%  wF1={wf1*100:5.2f}%  "
              f"mF1={mf1*100:5.2f}%  {np.median(lat):6.2f} ms  {size_mb:5.1f} MB  "
              f"AURC={rc['aurc']:.3f}")

    seen, dedup = set(), []
    for b in sorted(out["budgets"], key=lambda r: r["hash_bits"]):
        if b["hash_bits"] not in seen:
            seen.add(b["hash_bits"]); dedup.append(b)
    out["budgets"] = dedup
    best = max(out["budgets"], key=lambda r: r["weighted_f1"])
    print(f"\n  best CONCLAVE wF1 = {best['weighted_f1']:.3f} at 2^{best['hash_bits']} features")
    print("  published baselines on the same combined split (Alam et al., Table 7):")
    for k, v in sorted(PUBLISHED.items(), key=lambda kv: -kv[1]):
        delta = (best["weighted_f1"] - v) * 100
        flag = "CONCLAVE ahead" if delta > 0 else "CONCLAVE behind"
        print(f"    {k:20s} {v:.3f}   {flag} by {abs(delta):4.1f} F1 points")
    out["best"] = best
    save("exp2_humaid", out)


if __name__ == "__main__":
    main()
