"""Experiment 1 - Tool routing and abstention without a language model.

Question
--------
Agentic frameworks use an LLM to pick which tool to call.  Can a
pairwise-ranking ensemble do the same job on the standard function-calling
benchmark, at edge-feasible cost?

Protocol
--------
* Data: Berkeley Function Calling Leaderboard v3 (BFCL), the benchmark
  used by the small-language-model agent literature.
* Task A - **routing**: given a request and 2+ candidate tool specs,
  select the correct tool.  Metric: top-1 accuracy.
* Task B - **abstention**: given a request, decide whether *any*
  registered tool applies (BFCL "irrelevance" / relevance detection).
  Metric: accuracy / F1 / AUC.
* Splits are by *request*, never by pair, so no request leaks across the
  boundary.  We additionally report an **unseen-tool** split in which
  every tool in the test set was absent from training, which measures
  open-vocabulary generalisation.

Honest scope
------------
We measure tool *selection* and *abstention*, not full argument-level AST
match.  BFCL's headline accuracy includes argument filling; our numbers
are therefore NOT directly comparable to a leaderboard AST score and are
never presented as such.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conclave.brain.router import ToolRouter, spec_text          # noqa: E402
from conclave.brain.uncertainty import decompose                 # noqa: E402
from experiments.common import DATA, banner, save, set_seed   # noqa: E402

BFCL = DATA / "bfcl"


def load_pairs(name: str) -> List[Dict[str, Any]]:
    qs = [json.loads(l) for l in open(BFCL / f"{name}.json") if l.strip()]
    gt = {}
    ap = BFCL / f"ans_{name}.json"
    if ap.exists():
        for l in open(ap):
            if l.strip():
                d = json.loads(l)
                gt[d["id"]] = list(d["ground_truth"][0].keys())[0]
    out = []
    for r in qs:
        t = gt.get(r["id"])
        if t is None:
            continue
        turns = r["question"][0]
        q = " ".join(m["content"] for m in turns if m.get("role") == "user")
        funcs = r.get("function") or []
        if not q.strip() or t not in [f.get("name") for f in funcs]:
            continue
        out.append({"q": q, "funcs": funcs, "target": t})
    return out


def load_irrelevance(name: str) -> List[Dict[str, Any]]:
    out = []
    for l in open(BFCL / f"{name}.json"):
        if not l.strip():
            continue
        r = json.loads(l)
        turns = r["question"][0]
        q = " ".join(m["content"] for m in turns if m.get("role") == "user")
        funcs = r.get("function") or []
        if q.strip() and funcs:
            out.append({"q": q, "funcs": funcs, "target": None})
    return out


def eval_routing(train, test, label: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    router = ToolRouter(abstain=False).fit(
        [d["q"] for d in train], [d["funcs"] for d in train],
        [d["target"] for d in train])
    fit_s = time.perf_counter() - t0

    lat, correct = [], []
    for d in test:
        r = router.route(d["q"], d["funcs"])
        lat.append(r.latency_ms)
        correct.append(int(r.tool == d["target"]))

    # lexical control: pure TF-IDF argmax, no learning
    base = []
    for d in test:
        X = np.asarray([router._pair_features(d["q"], s) for s in d["funcs"]])
        base.append(int(d["funcs"][int(np.argmax(X[:, 0]))]["name"] == d["target"]))

    res = {
        "split": label, "n_train": len(train), "n_test": len(test),
        "avg_candidates": float(np.mean([len(d["funcs"]) for d in test])),
        "top1_accuracy": float(np.mean(correct)),
        "tfidf_only_baseline": float(np.mean(base)),
        "mean_latency_ms": float(np.mean(lat)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
        "fit_seconds": fit_s,
    }
    print(f"  {label:34s} n={len(test):5d} cand={res['avg_candidates']:4.1f}  "
          f"top1={res['top1_accuracy']*100:6.2f}%  (tfidf-only {res['tfidf_only_baseline']*100:5.2f}%)  "
          f"{res['mean_latency_ms']:6.3f} ms  p99={res['p99_latency_ms']:.3f} ms")
    return res


def unseen_tool_split(data, seed=0):
    """Hold out entire tools, so no test tool was ever seen in training."""
    rng = np.random.default_rng(seed)
    tools = sorted({d["target"] for d in data})
    rng.shuffle(tools)
    held = set(tools[: max(1, len(tools) // 3)])
    train = [d for d in data if d["target"] not in held
             and not any(f["name"] in held for f in d["funcs"])]
    test = [d for d in data if d["target"] in held]
    return train, test, len(held)


def main() -> None:
    set_seed()
    banner("EXPERIMENT 1 - Tool routing and abstention without an LLM")
    out: Dict[str, Any] = {"routing": [], "abstention": {}}

    print("\n[A] Routing accuracy (random request-level split)\n")
    for name in ["BFCL_v3_multiple", "BFCL_v3_live_multiple"]:
        data = [d for d in load_pairs(name) if len(d["funcs"]) > 1]
        if len(data) < 40:
            continue
        tr, te = train_test_split(data, test_size=0.3, random_state=42)
        out["routing"].append(eval_routing(tr, te, name))

    print("\n[B] Open-vocabulary generalisation (tools unseen in training)\n")
    for name in ["BFCL_v3_live_multiple"]:
        data = [d for d in load_pairs(name) if len(d["funcs"]) > 1]
        tr, te, n_held = unseen_tool_split(data)
        if len(tr) > 50 and len(te) > 30:
            r = eval_routing(tr, te, f"{name} [unseen tools n={n_held}]")
            r["held_out_tools"] = n_held
            out["routing"].append(r)

    print("\n[C] Abstention - should any tool fire at all?\n")
    irrel = load_irrelevance("BFCL_v3_irrelevance") + load_irrelevance("BFCL_v3_live_irrelevance")
    rel = ([d for d in load_pairs("BFCL_v3_multiple")]
           + [d for d in load_pairs("BFCL_v3_live_multiple")]
           + [d for d in load_pairs("BFCL_v3_live_simple")])
    X, y = [], []
    router = ToolRouter(abstain=False).fit(
        [d["q"] for d in rel], [d["funcs"] for d in rel], [d["target"] for d in rel])
    for d, lab in [(d, 1) for d in irrel] + [(d, 0) for d in rel]:
        F = np.asarray([router._pair_features(d["q"], s) for s in d["funcs"]],
                       dtype=np.float32)
        sims = F[:, 0]; covs = F[:, 2]; pars = F[:, 8]
        X.append([sims.max(), sims.mean(), sims.max() - sims.mean(),
                  sims.std() if len(sims) > 1 else 0.0,
                  covs.max(), covs.mean(), pars.max(), pars.mean(),
                  float(len(d["funcs"])), F[0, 10]])
        y.append(lab)
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    members = [RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=0),
               ExtraTreesClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=0),
               LogisticRegression(max_iter=1000)]
    for m in members:
        m.fit(Xtr, ytr)
    t0 = time.perf_counter()
    P = np.mean([m.predict_proba(Xte)[:, 1] for m in members], axis=0)
    lat = (time.perf_counter() - t0) * 1000 / len(yte)
    pred = (P > 0.5).astype(int)
    out["abstention"] = {
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(yte, pred)),
        "f1": float(f1_score(yte, pred)),
        "auc": float(roc_auc_score(yte, P)),
        "mean_latency_ms": float(lat),
    }
    a = out["abstention"]
    print(f"  irrelevance detection  n={a['n']}  acc={a['accuracy']*100:.2f}%  "
          f"F1={a['f1']*100:.2f}%  AUC={a['auc']*100:.2f}%  {a['mean_latency_ms']:.3f} ms")

    save("exp1_tool_routing", out)


if __name__ == "__main__":
    main()
