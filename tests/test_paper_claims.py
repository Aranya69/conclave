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
    # "~2100x smaller than a quantised 0.5B SLM's weights"
    ratio = d["slm_reference"]["Qwen2.5-0.5B (int8)"]["weights_mb"] * 1e6 / d["model_bytes"]
    assert 2000 <= ratio <= 2200

    by = {p["key"]: p for p in d["profiles"] if "measured" in p}
    # Pi Zero 2 W projection. The mean is stable across runs; p99 is a tail
    # statistic and moves several ms between runs even on an idle machine, so
    # it gets a looser bound. Tightening it is how this test became flaky.
    assert by["rpi_zero2w"]["estimated"]["est_mean_ms"] < 25.0
    assert by["rpi_zero2w"]["estimated"]["est_p99_ms"] < 35.0
    # "peak resident memory is 217 MB"
    assert close(by["host"]["measured"]["peak_rss_mb"], 217.0, tol=3.0)
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
