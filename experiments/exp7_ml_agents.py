"""Experiment 7 - CONCLAVE against other machine-learning-based agents.

Question
--------
Every agent in this comparison runs fully offline on the same CPU, so none
of them is hurt by a network outage.  What separates them is (i) how well
they act in closed loop, (ii) what they cost on low-cost hardware, and
(iii) whether they can tell when they should not be trusted.

Agents (identical training data, features, splits and seeds)
------------------------------------------------------------
CBR (k-NN)        case-based reasoning: the action of the nearest past cases,
                  the reasoning style of multi-agent CBR flood warning systems
Decision tree     a single CART tree
Naive Bayes       Bernoulli naive Bayes
Logistic reg.     multinomial logistic regression
MLP               a one-hidden-layer neural network
Random forest     bagged decision trees
LightGBM          gradient-boosted decision trees
Hard-vote ens.    MLP + random forest + k-NN, majority vote: the aggregation
                  used by ensemble flood-warning frameworks
CONCLAVE          heterogeneous soft-voting committee with uncertainty
                  decomposition (the configuration of Experiment 3)

Three parts
-----------
A. Flood response (the agentic task): imitation accuracy, closed-loop return
   and casualties with the network up (0 % offline) and down (73 % offline),
   decision latency, model size, peak RAM, estimated latency on boards.
B. Tool routing on BFCL v3: single-model rankers against the committee.
C. Crisis triage on HumAID: single text classifiers against the committee.

Run:  python experiments/exp7_ml_agents.py           (all parts)
      python experiments/exp7_ml_agents.py A         (one part)
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conclave.brain.ensemble import BrainReport, EnsembleBrain          # noqa: E402
from conclave.brain.featurize import BudgetedTextFeaturizer            # noqa: E402
from conclave.brain.uncertainty import decompose, decompose_batch      # noqa: E402
from conclave.edge.profile import PROFILES                             # noqa: E402
from experiments.common import DATA, RESULTS, banner, save, set_seed   # noqa: E402
from experiments.exp3_sequential import (HORIZON, N_EVAL_EPISODES,     # noqa: E402
                                         collect, run_policy,
                                         train_conclave)
from experiments.exp6_ablation import perturb                          # noqa: E402
from experiments.baselines import HardVoteBrain                         # noqa: E402

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAVE_LGBM = True
except ImportError:                       # pragma: no cover
    HAVE_LGBM = False

OUTAGE_LEVELS = [0.0, 0.5]               # 0 % and ~73 % of steps offline


# --------------------------------------------------------------- the agents
def flood_agents(seed: int = 0) -> Dict[str, Any]:
    """Unfitted brains, in the order they appear in the paper's table."""
    single = {
        "CBR (k-NN)": KNeighborsClassifier(n_neighbors=5, metric="cosine",
                                           algorithm="brute"),
        "Decision tree": DecisionTreeClassifier(min_samples_leaf=2, random_state=seed),
        "Naive Bayes": BernoulliNB(alpha=0.3, binarize=0.0),
        "Logistic regression": LogisticRegression(C=4.0, max_iter=400),
        "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=60, random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                                n_jobs=1, random_state=seed),
    }
    if HAVE_LGBM:
        single["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, num_leaves=63, n_jobs=1,
            random_state=seed, verbose=-1)
    out: Dict[str, Any] = {k: EnsembleBrain(members={"only": v}, weighting="uniform")
                           for k, v in single.items()}
    # ablation: the same committee, members weighted by out-of-fold log loss
    out["CONCLAVE, log-loss weights"] = EnsembleBrain(weighting="logloss")
    out["Hard-vote ensemble"] = HardVoteBrain(members={
        "mlp": MLPClassifier(hidden_layer_sizes=(64,), max_iter=60, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=1,
                                     random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
    })
    return out


def _size_kb(obj) -> float:
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    return buf.tell() / 1e3


_RSS_PROBE = r"""
import sys, resource, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[2])
blob = joblib.load(sys.argv[1])
fz, brain, obs = blob["fz"], blob["brain"], blob["obs"]
for o in obs:
    brain.report(fz.transform([o]))
hwm = None
for line in open("/proc/self/status"):
    if line.startswith("VmHWM:"):
        hwm = int(line.split()[1]) / 1024.0
print(hwm if hwm is not None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
"""


def peak_rss_mb(fz, brain, obs) -> float:
    """Peak resident memory of a fresh process that loads and runs the agent.

    Read from /proc/self/status (VmHWM): ru_maxrss would include the parent's
    resident set, which Linux carries over to the child across fork/exec.
    """
    root = str(Path(__file__).resolve().parents[1])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "agent.joblib"
        joblib.dump({"fz": fz, "brain": brain, "obs": obs[:200]}, p, compress=3)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        r = subprocess.run([sys.executable, "-c", _RSS_PROBE, str(p), root],
                           capture_output=True, text=True, env=env)
        try:
            return float(r.stdout.strip().splitlines()[-1])
        except Exception:
            print(r.stderr[-800:])
            return float("nan")


def paired_ci(a, b, n_boot: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def part_a(only: List[str], cache: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the flood-response agents.

    Each agent is evaluated independently and cached, so the part can be run
    one agent at a time (``A logistic`` etc.) on a machine that cannot hold a
    long job; ``A summary`` then computes the paired tests from the cache.
    """
    banner("EXPERIMENT 7A - Flood response: CONCLAVE vs other ML agents")
    flood = dict(cache.get("flood", {}))
    rows = {r["agent"]: r for r in flood.get("rows", [])}
    returns = dict(flood.get("returns", {}))
    names = list(flood_agents()) + ["CONCLAVE (ours)"]
    todo = [n for n in names if any(o.lower() in n.lower() for o in only)] \
        if only else names

    if only and only[0].lower() == "rss":
        sel = [n for n in names if not only[1:] or any(o.lower() in n.lower() for o in only[1:])]
        te_obs, _ = collect(60, seed0=60000)
        brain_c, fz, _ = train_conclave()
        cache_dir = Path(tempfile.gettempdir()) / "exp7_cache"
        cache_dir.mkdir(exist_ok=True)
        corpus = None
        for name in sel:
            mfile = cache_dir / (name.replace(" ", "_").replace("(", "").replace(")", "") + ".joblib")
            if name == "CONCLAVE (ours)":
                b = brain_c
            elif mfile.exists():
                b = joblib.load(mfile)
            else:
                if corpus is None:
                    o_, l_ = collect(160, seed0=1000)
                    corpus = (fz.transform(o_), l_)
                b = flood_agents()[name].fit(*corpus)
                joblib.dump(b, mfile)
            rss = peak_rss_mb(fz, b, te_obs)
            if name in rows:
                rows[name]["peak_rss_mb"] = rss
                rows[name]["fits_512mb"] = bool(rss < PROFILES["rpi_zero2w"].ram_mb)
            if name in flood.get("units", {}) and "static" in flood["units"][name]:
                flood["units"][name]["static"]["peak_rss_mb"] = rss
            print(f"  {name:<22s} peak RSS {rss:6.1f} MB", flush=True)
        todo = []
    if todo:
        tr_obs, tr_lab = collect(160, seed0=1000)          # same corpus as Exp. 3
        te_obs, te_lab = collect(60, seed0=60000)          # held-out decisions
        brain_c, fz, _ = train_conclave()                  # identical to Exp. 3
        Xtr, Xte = fz.transform(tr_obs), fz.transform(te_obs)
        Xsh = fz.transform(perturb(te_obs, dropout=0.6))   # failed-sensor shift
        flood["n_train_decisions"] = int(Xtr.shape[0])
        flood["n_test_decisions"] = int(Xte.shape[0])
        print(f"  train={Xtr.shape[0]:,} decisions  test={Xte.shape[0]:,}  "
              f"features={Xtr.shape[1]:,}\n")
    zero, pi4 = PROFILES["rpi_zero2w"], PROFILES["rpi4b_4gb"]

    units = flood.setdefault("units", {})
    # EXP7_BUDGET_S lets a machine with a short job limit run the part in
    # several invocations; by default everything runs in one go.
    budget_end = time.perf_counter() + float(os.environ.get("EXP7_BUDGET_S", "inf"))
    cache_dir = Path(tempfile.gettempdir()) / "exp7_cache"
    cache_dir.mkdir(exist_ok=True)
    for name in todo:
        if name in rows and name in returns:
            continue                                   # already complete
        mfile = cache_dir / (name.replace(" ", "_").replace("(", "").replace(")", "") + ".joblib")
        if name == "CONCLAVE (ours)":
            b = brain_c
        elif mfile.exists():
            b = joblib.load(mfile)
        else:
            t0 = time.perf_counter()
            b = flood_agents()[name].fit(Xtr, tr_lab)
            joblib.dump(b, mfile)
            print(f"  fitted {name:<22s} {time.perf_counter() - t0:6.1f} s", flush=True)
        u = units.setdefault(name, {})
        if "static" not in u:
            M = b.member_proba(Xte)
            pred = b.predict(Xte)
            Ms = b.member_proba(Xsh)
            lab = np.r_[np.zeros(M.shape[1]), np.ones(Ms.shape[1])]
            s_, ss = decompose_batch(M), decompose_batch(Ms)
            conf = np.r_[-b.predict_proba(Xte).max(1), -b.predict_proba(Xsh).max(1)]
            u["static"] = {
                "agent": name,
                "n_members": int(M.shape[0]),
                "accuracy": float(accuracy_score(te_lab, pred)),
                "macro_f1": float(f1_score(te_lab, pred, average="macro")),
                "auroc_shift_confidence": float(roc_auc_score(lab, conf)),
                "auroc_shift_epistemic": (float(roc_auc_score(
                    lab, np.r_[s_["epistemic"], ss["epistemic"]]))
                    if M.shape[0] > 1 else None),
                "size_kb": _size_kb({"fz": fz, "brain": b}),
                "peak_rss_mb": peak_rss_mb(fz, b, te_obs),
            }
        chunk = int(os.environ.get("EXP7_CHUNK", "40"))
        for op in OUTAGE_LEVELS:
            for start in range(0, N_EVAL_EPISODES, chunk):
                key = f"{op}:{start}"
                if key in u:
                    continue
                if time.perf_counter() > budget_end:
                    break
                r = run_policy("conclave", op, min(chunk, N_EVAL_EPISODES - start),
                               b, fz, None, seed0=7000 + start)
                u[key] = {"returns": r["returns"],
                          "mean_casualties": r["mean_casualties"],
                          "mean_false_alarms": r["mean_false_alarms"],
                          "offline_fraction": r["measured_offline_fraction"],
                          "median_decision_ms": r["median_decision_ms"],
                          "n": r["n_episodes"]}
                print(f"  {name:<22s} unit {key:>8s} done", flush=True)
        keys = [f"{op}:{st}" for op in OUTAGE_LEVELS
                for st in range(0, N_EVAL_EPISODES, chunk)]
        missing = [k for k in keys if k not in u]
        if missing:
            print(f"  {name}: {len(missing)} unit(s) left -- run again", flush=True)
            continue
        row = dict(u["static"])
        returns[name] = {}
        for op in OUTAGE_LEVELS:
            parts = [u[f"{op}:{st}"] for st in range(0, N_EVAL_EPISODES, chunk)]
            rets = [x for p_ in parts for x in p_["returns"]]
            w = np.array([p_["n"] for p_ in parts], dtype=float)
            key = "offline0" if op == 0.0 else "offline73"
            row[f"return_{key}"] = float(np.mean(rets))
            row[f"sem_{key}"] = float(np.std(rets) / np.sqrt(len(rets)))
            row[f"casualties_{key}"] = float(np.average([p_["mean_casualties"] for p_ in parts], weights=w))
            row[f"false_alarms_{key}"] = float(np.average([p_["mean_false_alarms"] for p_ in parts], weights=w))
            row[f"offline_fraction_{key}"] = float(np.average([p_["offline_fraction"] for p_ in parts], weights=w))
            if op == 0.0:
                row["median_decision_ms"] = float(np.median([p_["median_decision_ms"] for p_ in parts]))
            returns[name][str(op)] = rets
        row["est_ms_pi_zero2w"] = row["median_decision_ms"] / zero.scalar_score
        row["est_ms_pi4"] = row["median_decision_ms"] / pi4.scalar_score
        row["fits_512mb"] = bool(row["peak_rss_mb"] < zero.ram_mb)
        row["evaluated_in_chunks_of"] = chunk
        rows[name] = row
        print(f"  {name:<22s} acc {row['accuracy']*100:5.1f}%  "
              f"ret {row['return_offline0']:6.1f}/{row['return_offline73']:6.1f}  "
              f"cas {row['casualties_offline0']:5.1f}  "
              f"{row['median_decision_ms']:5.2f} ms  {row['size_kb']:8.1f} KB  "
              f"RSS {row['peak_rss_mb']:5.0f} MB", flush=True)

    flood["rows"] = [rows[n] for n in names if n in rows]
    flood["returns"] = returns
    # CONCLAVE against every other agent, paired over identical episode seeds
    if "CONCLAVE (ours)" in returns:
        tests = []
        ours = returns["CONCLAVE (ours)"]
        for name in names:
            if name == "CONCLAVE (ours)" or name not in returns:
                continue
            for op in OUTAGE_LEVELS:
                m, lo, hi = paired_ci(ours[str(op)], returns[name][str(op)])
                tests.append({"vs": name, "outage_prob": op, "delta_mean": m,
                              "ci_low": lo, "ci_high": hi,
                              "significant": bool(lo > 0 or hi < 0)})
        flood["paired_tests"] = tests
        others = [r for r in flood["rows"] if r["agent"] != "CONCLAVE (ours)"]
        if others:
            flood["best_other_agent"] = max(others, key=lambda r: r["return_offline0"])["agent"]
    ref = json.loads((RESULTS / "exp3_sequential.json").read_text())
    flood["reference_agents_exp3"] = [
        r for r in ref["rows"] if r["policy"] in
        ("cloud_oracle", "cloud_oracle_cache", "static_rules", "random")
        and r["outage_prob"] in OUTAGE_LEVELS]
    flood["shift"] = "60% sensor dropout"
    flood["board_scaling"] = {"pi_zero2w": zero.scalar_score, "pi4": pi4.scalar_score}
    return flood


# ------------------------------------------------------------ B. routing
def part_b(out: Dict[str, Any]) -> Dict[str, Any]:
    banner("EXPERIMENT 7B - Tool routing: single-model rankers vs committee")
    prev = {(r["split"], r["ranker"]): r for r in out.get("routing", {}).get("rows", [])}
    budget_end = time.perf_counter() + float(os.environ.get("EXP7_BUDGET_S", "inf"))
    from sklearn.model_selection import train_test_split
    from conclave.brain.router import ToolRouter
    from experiments.exp1_tool_routing import load_pairs, unseen_tool_split

    def rankers():
        out = {"Logistic regression": {"lr": LogisticRegression(max_iter=1000, n_jobs=1)},
               "Random forest": {"rf": RandomForestClassifier(
                   n_estimators=60, max_depth=10, min_samples_leaf=4, n_jobs=1,
                   random_state=0)}}
        if HAVE_LGBM:
            out["LightGBM"] = {"lgbm": lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.08, num_leaves=31, n_jobs=1,
                random_state=0, verbose=-1)}
        out["CONCLAVE (ours)"] = None          # the shipped committee
        return out

    splits = []
    for name in ["BFCL_v3_multiple", "BFCL_v3_live_multiple"]:
        data = [d for d in load_pairs(name) if len(d["funcs"]) > 1]
        tr, te = train_test_split(data, test_size=0.3, random_state=42)
        splits.append((name, tr, te))
    data = [d for d in load_pairs("BFCL_v3_live_multiple") if len(d["funcs"]) > 1]
    tr, te, n_held = unseen_tool_split(data)
    splits.append((f"BFCL_v3_live_multiple [unseen tools n={n_held}]", tr, te))

    rows = []
    for label, tr, te in splits:
        for rname, members in rankers().items():
            if (label, rname) in prev:
                rows.append(prev[(label, rname)])
                continue
            if time.perf_counter() > budget_end:
                print(f"  out of time before {label} / {rname} -- run again", flush=True)
                continue
            router = ToolRouter(members=members, abstain=False).fit(
                [d["q"] for d in tr], [d["funcs"] for d in tr], [d["target"] for d in tr])
            lat, ok = [], []
            for d in te:
                r = router.route(d["q"], d["funcs"])
                lat.append(r.latency_ms)
                ok.append(int(r.tool == d["target"]))
            rows.append({"split": label, "ranker": rname, "n_test": len(te),
                         "top1_accuracy": float(np.mean(ok)),
                         "mean_latency_ms": float(np.mean(lat)),
                         "size_kb": _size_kb(router.members_)})
            print(f"  {label[:34]:34s} {rname:22s} top1={np.mean(ok)*100:6.2f}%  "
                  f"{np.mean(lat):5.2f} ms", flush=True)
            out["routing"] = {"rows": rows}
            save("exp7_ml_agents", out)
    return {"rows": rows}


# ------------------------------------------------------------- C. HumAID
def part_c(out: Dict[str, Any], only: List[str], bits: int = 16) -> Dict[str, Any]:
    banner("EXPERIMENT 7C - Crisis triage: single classifiers vs committee")
    prev = {r["model"]: r for r in out.get("humaid", {}).get("rows", [])}
    import pandas as pd
    tr = pd.read_parquet(DATA / "humaid" / "train.parquet")
    te = pd.read_parquet(DATA / "humaid" / "test.parquet")
    n = 2 ** bits
    fz = BudgetedTextFeaturizer(n_word_features=n, n_char_features=n)
    Xtr = fz.fit(tr.tweet_text.tolist()).transform(tr.tweet_text.tolist())
    Xte = fz.transform(te.tweet_text.tolist())
    y, yt = tr.class_label.values, te.class_label.values
    cands = {
        "Naive Bayes": EnsembleBrain(members={"only": ComplementNB(alpha=0.3)},
                                     weighting="uniform"),
        "SGD (modified Huber)": EnsembleBrain(members={"only": SGDClassifier(
            loss="modified_huber", alpha=1e-5, max_iter=25, random_state=0)},
            weighting="uniform"),
        "Logistic regression": EnsembleBrain(members={"only": LogisticRegression(
            C=4.0, max_iter=400, n_jobs=1)}, weighting="uniform"),
        "CONCLAVE (ours)": EnsembleBrain(weighting="uniform"),   # as in Exp. 2
    }
    rows = []
    one = te.tweet_text.iloc[:1].tolist()
    for name, b in cands.items():
        if name in prev or (only and not any(o.lower() in name.lower() for o in only)):
            if name in prev:
                rows.append(prev[name])
            continue
        t0 = time.perf_counter()
        b.fit(Xtr, y)
        fit_s = time.perf_counter() - t0
        pred = b.predict(Xte)
        lat = []
        for _ in range(60):
            t1 = time.perf_counter()
            b.predict_proba(fz.transform(one))
            lat.append((time.perf_counter() - t1) * 1000)
        rows.append({"model": name, "hash_bits": bits,
                     "weighted_f1": float(f1_score(yt, pred, average="weighted")),
                     "macro_f1": float(f1_score(yt, pred, average="macro")),
                     "accuracy": float(accuracy_score(yt, pred)),
                     "median_latency_ms": float(np.median(lat)),
                     "fit_seconds": fit_s,
                     "size_mb": _size_kb({"fz": fz, "brain": b}) / 1e3})
        r = rows[-1]
        print(f"  {name:22s} wF1={r['weighted_f1']*100:5.2f}  mF1={r['macro_f1']*100:5.2f}  "
              f"{r['median_latency_ms']:5.2f} ms  {r['size_mb']:5.1f} MB  fit {fit_s:5.1f}s",
              flush=True)
        out["humaid"] = {"rows": rows + [prev[k] for k in prev if k not in {x["model"] for x in rows}]}
        save("exp7_ml_agents", out)
    order = list(cands)
    allrows = {r["model"]: r for r in list(prev.values()) + rows}
    return {"rows": [allrows[k] for k in order if k in allrows]}


def main() -> None:
    set_seed()
    args = sys.argv[1:]
    parts = {args[0].upper()} if args else {"A", "B", "C"}
    only = [a for a in args[1:]]
    path = RESULTS / "exp7_ml_agents.json"
    out: Dict[str, Any] = {}
    if path.exists():
        try:
            out = json.loads(path.read_text())
            out.pop("_meta", None)
        except Exception:
            out = {}
    if "A" in parts:
        out["flood"] = part_a(only, out)
        save("exp7_ml_agents", out)          # checkpoint after every agent batch
    if "B" in parts:
        out["routing"] = part_b(out)
    if "C" in parts:
        out["humaid"] = part_c(out, only)
    save("exp7_ml_agents", out)


if __name__ == "__main__":
    main()
