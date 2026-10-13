"""Every numeric claim in the paper, checked against results/*.json.

If an experiment is re-run and a number moves, this test fails and the
paper must be updated.  Tolerances are tight (0.05 absolute on
percentages, 0.5 on returns) -- enough for float formatting, not enough
to hide a real change.
"""
import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(name):
    p = RESULTS / f"{name}.json"
    if not p.exists():
        pytest.skip(f"{name}.json not generated yet - run experiments/")
    return json.loads(p.read_text())


def close(a, b, tol=0.05):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- exp1
def test_routing_claims():
    d = load("exp1_tool_routing")
    by = {r["split"]: r for r in d["routing"]}
    m = by["BFCL_v3_multiple"]
    lm = by["BFCL_v3_live_multiple"]
    un = next(r for r in d["routing"] if "unseen" in r["split"])

    assert close(m["top1_accuracy"] * 100, 98.33), "abstract/Table II: multiple = 98.3%"
    assert close(m["tfidf_only_baseline"] * 100, 90.00)
    assert close(lm["top1_accuracy"] * 100, 90.82), "abstract: live_multiple = 90.8%"
    assert close(lm["tfidf_only_baseline"] * 100, 79.43)
    assert close(un["top1_accuracy"] * 100, 78.90), "abstract: unseen tools = 78.9%"
    assert close(un["tfidf_only_baseline"] * 100, 56.65)

    # "3.99 ms per decision (p99 7.09 ms)"
    assert close(lm["mean_latency_ms"], 3.99, tol=0.5)
    assert close(lm["p99_latency_ms"], 7.09, tol=1.0)

    # learning contributes "11.4 points ... and 22.2 points"
    assert close((lm["top1_accuracy"] - lm["tfidf_only_baseline"]) * 100, 11.4, tol=0.1)
    assert close((un["top1_accuracy"] - un["tfidf_only_baseline"]) * 100, 22.2, tol=0.1)

    a = d["abstention"]
    assert close(a["accuracy"] * 100, 84.28), "Table II: abstention acc 84.3%"
    assert close(a["auc"], 0.9212, tol=0.001), "abstract: 0.921 AUC"
    assert a["mean_latency_ms"] < 0.5


# ---------------------------------------------------------------- exp2
def test_humaid_claims():
    d = load("exp2_humaid")
    by = {b["hash_bits"]: b for b in d["budgets"]}
    assert set(by) >= {12, 14, 16}, "paper reports three feature budgets"

    assert close(by[16]["weighted_f1"] * 100, 74.25), "abstract: 74.3% wF1"
    assert close(by[14]["weighted_f1"] * 100, 72.56)
    assert close(by[12]["weighted_f1"] * 100, 71.32), "abstract: 71.3% wF1"
    assert close(by[16]["macro_f1"] * 100, 71.60)
    assert close(by[12]["macro_f1"] * 100, 68.52)

    assert close(by[16]["model_mb"], 36.4, tol=0.1), "abstract: 36 MB artefact"
    assert close(by[12]["model_mb"], 2.6, tol=0.1), "abstract: 2.6 MB artefact"

    pb = d["published_baselines"]
    roberta = pb["RoBERTa (125M)"]
    # Alam et al. Table 7, row "All" -- NOT the 0.781 column average, which is
    # taken over per-event rows and is not comparable to this split.
    assert close(roberta, 0.760, tol=0.001)
    assert close((roberta - by[16]["weighted_f1"]) * 100, 1.75, tol=0.06), \
        "paper claims a 1.8-point gap to RoBERTa"
    # CONCLAVE must beat the classical baselines the authors report
    assert by[16]["weighted_f1"] > pb["SVM (classical)"]
    assert by[16]["weighted_f1"] > pb["Random Forest"]
    assert by[12]["weighted_f1"] > pb["Random Forest"], \
        "the 2.6 MB config should still edge past their random forest"
    assert close(by[16]["weighted_f1"], pb["FastText"], tol=0.002), \
        "paper claims parity with FastText"
    # "a 14x smaller artefact"  (500 MB / 36.4 MB)
    assert 13 <= 500 / by[16]["model_mb"] <= 15


# ---------------------------------------------------------------- exp3
def test_availability_claims():
    d = load("exp3_sequential")
    comp = {c["outage_prob"]: c for c in d["paired_comparisons"]}

    # abstract: "indistinguishable ... (delta = 0.6, 95% CI [-0.7, 1.9])"
    c0 = comp[0.0]
    assert close(c0["delta_mean"], 0.6, tol=0.1)
    assert close(c0["ci_low"], -0.7, tol=0.15) and close(c0["ci_high"], 1.9, tol=0.15)
    assert c0["verdict"] == "no difference", "the fair-comparison check must hold"

    # every non-zero outage level must favour CONCLAVE significantly
    for op, c in comp.items():
        if op > 0:
            assert c["ci_low"] > 0, f"outage {op} should significantly favour CONCLAVE"
    assert d["significant_crossover_outage_prob"] == 0.1

    rows = {(r["policy"], r["outage_prob"]): r for r in d["rows"]}
    conclave_hi = rows[("conclave", 0.5)]
    cloud_hi = rows[("cloud_oracle", 0.5)]
    # abstract: "-127 mean return into +34 when 73% of steps are offline"
    assert close(cloud_hi["mean_return"], -127.3, tol=0.5)
    assert close(conclave_hi["mean_return"], 33.5, tol=0.5)
    assert close(conclave_hi["measured_offline_fraction"] * 100, 73.1, tol=0.5)

    # "casualties rise from 53.7 to 276.9 ... CONCLAVE stays between 41.9 and 50.0"
    assert close(rows[("cloud_oracle", 0.0)]["mean_casualties"], 53.7, tol=0.5)
    assert close(cloud_hi["mean_casualties"], 276.9, tol=0.5)
    conclave_cas = [r["mean_casualties"] for (p, _), r in rows.items() if p == "conclave"]
    assert 41.0 <= min(conclave_cas) and max(conclave_cas) <= 50.5

    # the cache baseline "recovers roughly half the loss but no more"
    cache = rows[("cloud_oracle_cache", 0.5)]["mean_return"]
    assert cloud_hi["mean_return"] < cache < 0


# ---------------------------------------------------------------- exp4
def test_edge_claims():
    d = load("exp4_edge")
    # "serialises to 232 KB"
    assert close(d["model_bytes"] / 1000, 232.8, tol=1.0)

    by = {p["key"]: p for p in d["profiles"] if "measured" in p}
    # Pi Zero 2 W projection. The mean is stable across runs; p99 is a tail
    # statistic and moves several ms between runs even on an idle machine, so
    # it gets a looser bound. Tightening it is how this test became flaky.
    assert by["rpi_zero2w"]["estimated"]["est_mean_ms"] < 25.0
    assert by["rpi_zero2w"]["estimated"]["est_p99_ms"] < 35.0
    # Table VI: "22.6 ms (p99 30.6 ms) even on a Raspberry Pi Zero 2 W ... 56 mJ"
    assert close(by["rpi_zero2w"]["estimated"]["est_mean_ms"], 22.59, tol=0.05)
    assert close(by["rpi_zero2w"]["estimated"]["est_energy_mJ"], 56.5, tol=0.1)
    # timings must come from a quiet machine, or the paper's numbers are noise
    for k, p in by.items():
        assert p["measured"]["trials"] >= 3, f"{k}: too few timing trials"
        assert p["measured"]["noise_ratio"] < 1.5, (
            f"{k}: noise ratio {p['measured']['noise_ratio']} - the machine was "
            "busy; re-run exp4 on an idle system before quoting these numbers")
    # every profile must fit its RAM budget, or the claim of edge-feasibility fails
    for k, p in by.items():
        assert p["estimated"]["fits_in_ram"], f"{k} does not fit its RAM ceiling"


# ---------------------------------------------------------------- exp5
def test_uncertainty_claims():
    d = load("exp5_uncertainty")
    sp = {round(r["coverage"], 2): r for r in d["selective_prediction"]}
    # "96.3% at full coverage to 100% at 50% coverage; random stays ~96%"
    assert close(sp[1.0]["confidence"] * 100, 96.29)
    assert close(sp[0.5]["confidence"] * 100, 100.0)
    assert 95.5 <= sp[0.5]["random"] * 100 <= 96.5
    assert close(d["risk_coverage"]["aurc"], 0.946, tol=0.002)

    ood = {(r["shift"], r["magnitude"]): r for r in d["ood_detection"]}
    nt = ood[("novel_text", 1.0)]
    assert close(nt["auroc_epistemic"], 0.955, tol=0.002), "abstract: AUROC 0.955"
    assert close(nt["accuracy_under_shift"] * 100, 67.2, tol=0.2)
    dr = ood[("sensor_dropout", 0.8)]
    assert close(dr["auroc_epistemic"], 0.820, tol=0.002)
    assert close(dr["accuracy_under_shift"] * 100, 26.6, tol=0.2)

    # NEGATIVE RESULT 1: confidence matches or beats the epistemic term
    assert nt["auroc_confidence"] >= nt["auroc_epistemic"], \
        "paper claims plain confidence is not beaten by the decomposition"

    # NEGATIVE RESULT 2: random oversight beats uncertainty-triggered oversight
    hb = {(r["sensor_dropout"], r["budget"]): r for r in d["human_attention_budget"]}
    r = hb[(0.6, 10)]
    assert close(r["none"], -14.7, tol=0.5)
    assert close(r["random"], 2.9, tol=0.5)
    assert close(r["epistemic"], -11.1, tol=0.5)
    assert close(r["confidence"], -10.3, tol=0.5)
    assert r["random"] > r["epistemic"], \
        "the reported negative result must still hold"


# ---------------------------------------------------------------- exp6
def test_ablation_claims():
    """Ablation of the committee (Section IV-F, Table XI).

    If any of these start passing 'better' than asserted, the paper's statement
    of what the committee does and does not add has become false and the
    section must be rewritten -- not the assertion.
    """
    d = load("exp6_ablation")
    by = {r["model"]: r for r in d["rows"]}

    singles = [r for r in d["rows"] if r["n_members"] == 1]
    best_single = max(singles, key=lambda r: r["accuracy"])
    best_comm = by["committee (strong + diverse)"]

    # "the best single model exactly matches the best committee"
    assert close(best_single["accuracy"], best_comm["accuracy"], tol=0.001)
    # "on AURC the single model is marginally ahead"
    assert best_single["aurc"] >= best_comm["aurc"] - 1e-6

    # a single model cannot produce an epistemic term at all
    for r in singles:
        assert r["mean_epistemic"] == 0.0, f"{r['model']} should have no epistemic term"

    # diversity is load-bearing: homogeneous committees lose the signal
    trees = by["committee (all trees)"]
    assert trees["mean_epistemic"] < best_comm["mean_epistemic"]
    assert trees["auroc_shift_epistemic"] < 0.60, "all-tree committee near chance"
    assert best_comm["auroc_shift_epistemic"] > 0.75

    # the shipped default is dragged BELOW its own best member by uniform averaging
    shipped = by["committee (shipped default)"]
    assert shipped["accuracy"] < best_single["accuracy"]
    assert close(shipped["accuracy"] * 100, 95.0, tol=0.5)


# ---------------------------------------------------------------- exp7
def test_ml_agent_comparison_claims():
    """Section IV-E, Tables IX and X: CONCLAVE against other ML-based agents."""
    d = load("exp7_ml_agents")
    fl = d["flood"]
    rows = {r["agent"]: r for r in fl["rows"]}
    ours = rows["CONCLAVE (ours)"]
    others = [r for a, r in rows.items()
              if a not in ("CONCLAVE (ours)", "CONCLAVE, log-loss weights")]
    assert len(others) == 8, "the paper compares against eight other ML agents"

    # identical to Experiment 3 (Table VII)
    assert close(ours["return_offline0"], 36.5, tol=0.05)
    assert close(ours["return_offline73"], 33.5, tol=0.05)
    assert close(ours["casualties_offline0"], 41.9, tol=0.05)
    assert close(ours["casualties_offline73"], 50.0, tol=0.05)
    assert close(ours["accuracy"] * 100, 96.3, tol=0.05)

    # "the highest return of all agents ... and the fewest casualties"
    for r in others:
        assert ours["return_offline0"] > r["return_offline0"], r["agent"]
        assert ours["return_offline73"] > r["return_offline73"], r["agent"]
        assert ours["casualties_offline0"] < r["casualties_offline0"], r["agent"]
        assert ours["casualties_offline73"] < r["casualties_offline73"], r["agent"]

    # paired tests: significant against every agent with the network up, and
    # against all but the decision tree with 73 % offline
    tests = {(t["vs"], t["outage_prob"]): t for t in fl["paired_tests"]}
    for r in others:
        assert tests[(r["agent"], 0.0)]["significant"], r["agent"]
        if r["agent"] == "Decision tree":
            t = tests[(r["agent"], 0.5)]
            assert not t["significant"]
            assert close(t["delta_mean"], 1.9, tol=0.06)
            assert close(t["ci_low"], -0.2, tol=0.06) and close(t["ci_high"], 3.9, tol=0.06)
        else:
            assert tests[(r["agent"], 0.5)]["significant"], r["agent"]

    # "one-step accuracy does not predict closed-loop quality"
    for a, acc in [("Decision tree", 98.7), ("LightGBM", 98.6), ("Logistic regression", 98.4)]:
        assert close(rows[a]["accuracy"] * 100, acc, tol=0.05), a
        assert rows[a]["accuracy"] > ours["accuracy"]
        gap = rows[a]["casualties_offline73"] - ours["casualties_offline73"]
        assert 12.5 <= gap <= 19.0, f"{a}: '13 to 19 more casualties'"
    assert close(rows["Hard-vote ensemble"]["return_offline73"], 24.5, tol=0.05)
    assert close(rows["Naive Bayes"]["return_offline73"], -69.0, tol=0.05)

    # low-cost hardware: "2 to 6x slower and 7 to 150x larger"
    for a in ["CBR (k-NN)", "Random forest", "Hard-vote ensemble"]:
        assert 1.7 <= rows[a]["median_decision_ms"] / ours["median_decision_ms"] <= 6.5, a
        assert 7.0 <= rows[a]["size_kb"] / ours["size_kb"] <= 150.0, a
    assert close(ours["median_decision_ms"], 2.45, tol=0.01)
    assert close(ours["est_ms_pi_zero2w"], 22.3, tol=0.05)
    assert close(ours["size_kb"], 249, tol=1.0)
    assert close(ours["peak_rss_mb"], 184, tol=1.0)
    assert all(r["fits_512mb"] for r in fl["rows"])

    # cloud agent reference (Table IX)
    ref = {(r["policy"], r["outage_prob"]): r for r in fl["reference_agents_exp3"]}
    assert close(ref[("cloud_oracle", 0.5)]["mean_return"], -127.3, tol=0.05)
    assert close(ref[("cloud_oracle_cache", 0.5)]["mean_return"], -55.9, tol=0.05)

    # ablation: log-loss weighting is more accurate one step at a time but
    # worse in closed loop (Section IV-F)
    ll = rows["CONCLAVE, log-loss weights"]
    assert close(ll["accuracy"] * 100, 98.0, tol=0.05)
    assert close(ll["return_offline0"], 33.5, tol=0.05)
    assert close(ll["return_offline73"], 30.9, tol=0.05)
    assert close(ll["casualties_offline0"], 56.9, tol=0.05)
    for op in (0.0, 0.5):
        assert tests[("CONCLAVE, log-loss weights", op)]["significant"]

    # Table X: single rankers / classifiers against the committee
    rt = {(r["split"].split(" [")[0] + ("|unseen" if "unseen" in r["split"] else ""),
           r["ranker"]): r["top1_accuracy"] * 100 for r in d["routing"]["rows"]}
    m, lm, un = "BFCL_v3_multiple", "BFCL_v3_live_multiple", "BFCL_v3_live_multiple|unseen"
    assert close(rt[(m, "Logistic regression")], 98.33) and close(rt[(m, "CONCLAVE (ours)")], 98.33)
    assert close(rt[(lm, "CONCLAVE (ours)")], 90.82)
    gaps = [rt[(lm, "CONCLAVE (ours)")] - rt[(lm, k)]
            for k in ("Logistic regression", "Random forest", "LightGBM")]
    assert close(min(gaps), 1.9, tol=0.06) and close(max(gaps), 4.1, tol=0.06)
    assert close(rt[(un, "Logistic regression")], 84.10)
    assert close(rt[(un, "Logistic regression")] - rt[(un, "CONCLAVE (ours)")], 5.2, tol=0.06)
    hm = {r["model"]: r["weighted_f1"] * 100 for r in d["humaid"]["rows"]}
    assert close(hm["Logistic regression"], 74.73) and close(hm["CONCLAVE (ours)"], 74.27)
    assert close(hm["SGD (modified Huber)"], 72.66) and close(hm["Naive Bayes"], 66.17)


# ---------------------------------------------------------------- exp8
def test_edge_runtime_claims():
    """Section IV-D, Table VIII: the NumPy-only runtime."""
    d = load("exp8_edge_runtime")
    eq = d["equivalence"]
    assert eq["n"] == 2400 and eq["action_agreement"] == 1.0 and eq["gate_agreement"] == 1.0
    assert eq["max_abs_posterior_diff"] < 1e-12, "'largest difference ... 8e-14'"
    assert close(d["export"]["npz_kb"], 113, tol=1.0)
    assert d["export"]["columns_kept"] == 2486 and d["export"]["columns_total"] == 16396

    full, rt = d["full_stack"], d["runtime"]
    assert close(full["peak_rss_mb"], 185.7, tol=0.1)
    assert close(rt["peak_rss_mb"], 32.2, tol=0.1)
    assert not rt["sklearn_loaded"] and not rt["scipy_loaded"]
    assert "error" not in d["runtime_256mb_cap"], "runtime must run under a 256 MB cap"
    assert "error" not in d["full_stack_512mb_cap"]
    assert close(full["median_ms"], 2.72, tol=0.01) and close(rt["median_ms"], 0.19, tol=0.01)
    assert close(full["p99_ms"], 4.18, tol=0.01) and close(rt["p99_ms"], 0.30, tol=0.01)
    zero = d["boards"]["rpi_zero2w"]
    assert close(zero["runtime_est_median_ms"], 1.71, tol=0.01)
    assert close(zero["full_est_median_ms"], 24.7, tol=0.05)
    assert close(d["boards"]["rpi4b_4gb"]["runtime_est_median_ms"], 0.67, tol=0.01)

    # the closed-loop agent of Exp. 3/7 behaves identically inside the simulator
    a7 = d["exp7_agent"]
    exp7 = {r["agent"]: r for r in load("exp7_ml_agents")["flood"]["rows"]}["CONCLAVE (ours)"]
    assert a7["action_agreement"] == 1.0
    assert abs(a7["closed_loop"]["0.0"]["mean_return"] - exp7["return_offline0"]) < 1e-9
    assert abs(a7["closed_loop"]["0.5"]["mean_return"] - exp7["return_offline73"]) < 1e-9
    assert close(a7["npz_kb"], 128, tol=1.0)
    assert close(a7["runtime"]["peak_rss_mb"], 31.4, tol=0.1)
    assert close(a7["est_ms_pi_zero2w"], 1.69, tol=0.01)
