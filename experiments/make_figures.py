"""Regenerate every figure in the paper from results/*.json."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import RESULTS, ROOT

FIG = ROOT / "paper" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"conclave": "#1b6ca8", "cloud_oracle": "#c0392b",
     "cloud_oracle_cache": "#e67e22", "static_rules": "#7f8c8d",
     "random": "#bdc3c7"}
LBL = {"conclave": "CONCLAVE (local ensemble)", "cloud_oracle": "Cloud oracle agent",
       "cloud_oracle_cache": "Cloud oracle + cache", "static_rules": "Static threshold rules",
       "random": "Random policy"}


def load(n):
    p = RESULTS / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_availability():
    d = load("exp3_sequential")
    if not d: return
    rows = d["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for pol in ["conclave", "cloud_oracle", "cloud_oracle_cache", "static_rules"]:
        r = sorted([x for x in rows if x["policy"] == pol], key=lambda z: z["outage_prob"])
        x = [z["measured_offline_fraction"] * 100 for z in r]
        y = [z["mean_return"] for z in r]
        e = [z["sem_return"] for z in r]
        ax[0].errorbar(x, y, yerr=e, marker="o", ms=3.5, lw=1.5, capsize=2,
                       color=C[pol], label=LBL[pol])
        ax[1].plot(x, [z["mean_casualties"] for z in r], marker="o", ms=3.5,
                   lw=1.5, color=C[pol], label=LBL[pol])
    ax[0].set_xlabel("steps with no connectivity (%)"); ax[0].set_ylabel("mean episode return")
    ax[0].axhline(0, color="k", lw=0.6, ls=":")
    ax[1].set_xlabel("steps with no connectivity (%)"); ax[1].set_ylabel("mean casualties")
    ax[0].legend(fontsize=6.6, loc="lower left", frameon=False)
    fig.suptitle("Agent performance degrades with connectivity, unless the brain is local",
                 fontsize=9.5, y=1.03)
    fig.savefig(FIG / "fig1_availability.pdf"); fig.savefig(FIG / "fig1_availability.png")
    plt.close(fig); print("fig1_availability")


def fig_frontier():
    d = load("exp2_humaid")
    if not d: return
    b = sorted(d["budgets"], key=lambda r: r["model_mb"])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot([r["model_mb"] for r in b], [r["weighted_f1"] * 100 for r in b],
            marker="o", color="#1b6ca8", lw=1.6, label="CONCLAVE (hashed features)")
    for r in b:
        ax.annotate(f"$2^{{{r['hash_bits']}}}$", (r["model_mb"], r["weighted_f1"] * 100),
                    textcoords="offset points", xytext=(5, -9), fontsize=7)
    pb = d["published_baselines"]
    styles = {"RoBERTa (125M)": ("#c0392b", "-"), "FastText": ("#7f8c8d", "--"),
              "SVM (classical)": ("#95a5a6", ":"), "Random Forest": ("#b2babb", ":")}
    for k, (col, ls) in styles.items():
        if k not in pb:
            continue
        ax.axhline(pb[k] * 100, color=col, ls=ls, lw=1.1)
        ax.annotate(f"{k.split(' (')[0]} {pb[k]*100:.1f}",
                    (b[-1]["model_mb"] * 0.30, pb[k] * 100 + 0.18),
                    fontsize=6.2, color=col, ha="left")
    ax.set_title("HumAID: accuracy vs footprint\n(dashed = baselines reported by dataset authors)",
                 fontsize=8.5)
    ax.set_xscale("log"); ax.set_xlabel("serialised model size (MB, log scale)")
    ax.set_ylabel("weighted F1 (%)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.margins(y=0.14)
    fig.savefig(FIG / "fig2_frontier.pdf"); fig.savefig(FIG / "fig2_frontier.png")
    plt.close(fig); print("fig2_frontier")


def fig_uncertainty():
    d = load("exp5_uncertainty")
    if not d: return
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    rc = d.get("risk_coverage")
    sp = d.get("selective_prediction", [])
    if sp:
        cov = [r["coverage"] * 100 for r in sp]
        ax[0].plot(cov, [r["random"] * 100 for r in sp], marker="s", ms=3.5,
                   color="#bdc3c7", lw=1.4, label="random rejection")
        ax[0].plot(cov, [r["confidence"] * 100 for r in sp], marker="o", ms=3.5,
                   color="#1b6ca8", lw=1.6, label="ensemble confidence")
        ax[0].plot(cov, [r["gate"] * 100 for r in sp], marker="^", ms=3.5,
                   color="#16a085", lw=1.4, ls="--", label="confidence - epistemic")
        ax[0].set_xlabel("coverage: decisions acted on (%)")
        ax[0].set_ylabel("accuracy on acted-on decisions (%)")
        ax[0].legend(fontsize=7, frameon=False, loc="lower left")
        ax[0].set_title("Selective prediction", fontsize=9)
    ood = d.get("ood_detection", [])
    if ood:
        names = [f"{r['shift'].replace('sensor_','').replace('_',' ')}\n"
                 f"x{r['magnitude']:g}" for r in ood]
        acc = [r["accuracy_under_shift"] * 100 for r in ood]
        auc = [r["auroc_epistemic"] for r in ood]
        a2 = ax[1].twinx(); a2.grid(False)
        ax[1].bar(range(len(ood)), acc, color="#d5dbdb", width=0.62,
                  label="accuracy under shift")
        a2.plot(range(len(ood)), auc, marker="o", ms=4, color="#c0392b", lw=1.6,
                label="AUROC (epistemic)")
        a2.axhline(0.5, color="#c0392b", ls=":", lw=0.9)
        ax[1].set_xticks(range(len(ood))); ax[1].set_xticklabels(names, fontsize=5.6)
        ax[1].set_ylabel("accuracy (%)"); a2.set_ylabel("shift-detection AUROC")
        a2.set_ylim(0.4, 1.02)
        ax[1].set_title("Uncertainty rises as accuracy falls", fontsize=9)
        h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
        ax[1].legend(h1 + h2, l1 + l2, fontsize=6.6, frameon=False, loc="lower left")
    fig.savefig(FIG / "fig3_uncertainty.pdf"); fig.savefig(FIG / "fig3_uncertainty.png")
    plt.close(fig); print("fig3_uncertainty")


def fig_edge():
    d = load("exp4_edge")
    if not d: return
    profs = [p for p in d["profiles"] if "measured" in p]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    names = [p["device"].replace(" (", "\n(") for p in profs]
    mean = [p["estimated"]["est_mean_ms"] for p in profs]
    p99 = [p["estimated"]["est_p99_ms"] for p in profs]
    xs = np.arange(len(profs))
    ax[0].bar(xs - 0.19, mean, 0.38, label="estimated mean", color="#1b6ca8")
    ax[0].bar(xs + 0.19, p99, 0.38, label="estimated p99", color="#5dade2")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(names, fontsize=6)
    ax[0].set_ylabel("latency per decision (ms)")
    ax[0].set_title("Estimated on-device decision latency", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].axhline(1000, color="#c0392b", ls="--", lw=1.0)
    ax[0].annotate("1 s", (len(profs) - 0.6, 1050), fontsize=6.5, color="#c0392b")

    agent_mb = d["model_bytes"] / 1e6
    labels = ["CONCLAVE agent"] + list(d["slm_reference"])
    sizes = [agent_mb] + [v["weights_mb"] for v in d["slm_reference"].values()]
    cols = ["#1b6ca8"] + ["#c0392b"] * len(d["slm_reference"])
    ax[1].barh(range(len(sizes)), sizes, color=cols, height=0.6)
    ax[1].set_yticks(range(len(sizes)))
    ax[1].set_yticklabels([l.replace(" (", "\n(") for l in labels], fontsize=6.3)
    ax[1].set_xscale("log"); ax[1].set_xlabel("stored parameters / model (MB, log)")
    ax[1].set_title("Deployment footprint", fontsize=9)
    for i, s in enumerate(sizes):
        ax[1].annotate(f"{s:.2f} MB" if s < 10 else f"{s:.0f} MB",
                       (s, i), textcoords="offset points", xytext=(4, -2.5), fontsize=6.3)
    ax[1].invert_yaxis()
    fig.savefig(FIG / "fig4_edge.pdf"); fig.savefig(FIG / "fig4_edge.png")
    plt.close(fig); print("fig4_edge")


def fig_routing():
    d = load("exp1_tool_routing")
    if not d: return
    rows = d["routing"]
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    names = [r["split"].replace("BFCL_v3_", "").replace("_", " ") for r in rows]
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r["tfidf_only_baseline"] * 100 for r in rows], 0.4,
           label="lexical similarity only", color="#bdc3c7")
    ax.bar(xs + 0.2, [r["top1_accuracy"] * 100 for r in rows], 0.4,
           label="CONCLAVE ranking ensemble", color="#1b6ca8")
    for i, r in enumerate(rows):
        ax.annotate(f"{r['mean_latency_ms']:.1f} ms",
                    (i + 0.2, r["top1_accuracy"] * 100 + 1.2),
                    ha="center", fontsize=6.2)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=6, rotation=12)
    ax.set_ylabel("top-1 tool selection accuracy (%)"); ax.set_ylim(0, 108)
    ax.set_title("Tool routing on BFCL v3 without a language model", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(FIG / "fig5_routing.pdf"); fig.savefig(FIG / "fig5_routing.png")
    plt.close(fig); print("fig5_routing")


if __name__ == "__main__":
    fig_availability(); fig_frontier(); fig_uncertainty(); fig_edge(); fig_routing()
    print(f"\nfigures -> {FIG}")
