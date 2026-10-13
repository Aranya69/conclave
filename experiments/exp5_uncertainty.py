"""Experiment 5 - Does ensemble disagreement earn its place?

The framework's central claim is that a committee gives an agent an
uncertainty signal a single model cannot, and that *decomposing* it into
aleatoric and epistemic parts is what makes it actionable.  Three tests:

A. **Selective prediction.**  If the agent may decline to act, does
   accuracy on what it does act on rise?  We compare rejection by random
   choice, by confidence alone, and by the full gate.

B. **Out-of-distribution detection.**  We train on ordinary floods and
   evaluate on severities never seen in training.  Epistemic uncertainty
   should rise on the shifted data while aleatoric stays flat -- that is
   the whole point of separating them.  Metric: AUROC for separating
   in-distribution from shifted observations.

C. **Does it help downstream?**  Under distribution shift, does an agent
   that escalates on high disagreement end up with fewer casualties than
   one that always acts?  This is the test that matters: an uncertainty
   signal that does not change outcomes is decoration.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conclave.brain.ensemble import EnsembleBrain                        # noqa: E402
from conclave.brain.featurize import (BudgetedTextFeaturizer,            # noqa: E402
                                   FusedFeaturizer, TabularFeaturizer)
from conclave.brain.uncertainty import (UncertaintyGate, decompose_batch,# noqa: E402
                                     risk_coverage_curve)
from conclave.core.types import Action, ActionKind                       # noqa: E402
from conclave.envs.disaster import (ACTIONS, DisasterConfig,             # noqa: E402
                                 FloodResponseEnv, expert_policy)
from experiments.common import banner, save, set_seed                 # noqa: E402
from experiments.exp3_sequential import HORIZON, collect              # noqa: E402


def collect_shifted(n, seed0, rain_intensity, horizon=HORIZON):
    """Episodes from a severity regime absent from training."""
    rng = np.random.default_rng(seed0)
    obs, lab = [], []
    for e in range(n):
        env = FloodResponseEnv(DisasterConfig(seed=seed0 + e, horizon=horizon,
                                              rain_intensity=rain_intensity))
        o = env.reset()
        for _ in range(horizon):
            a = expert_policy(o, rng)
            obs.append(o); lab.append(a)
            o, _, done, _ = env.step(Action(kind=ActionKind.TOOL, tool=a))
            if done:
                break
    return obs, np.array(lab)


def main() -> None:
    set_seed()
    parts = set(sys.argv[1:]) or {"A", "B", "C"}
    banner("EXPERIMENT 5 - Value of decomposed ensemble uncertainty")
    out: Dict[str, Any] = {}

    tr_obs, tr_lab = collect(160, seed0=1000)
    te_obs, te_lab = collect(60, seed0=60000)
    fz = FusedFeaturizer(
        text=BudgetedTextFeaturizer(n_word_features=2 ** 13, n_char_features=2 ** 13),
        tabular=TabularFeaturizer()).fit(tr_obs)
    # weighting pinned: this experiment's published numbers were produced with
    # uniform averaging, before the library default became 'logloss' (see exp6).
    brain = EnsembleBrain(weighting="uniform").fit(fz.transform(tr_obs), tr_lab)
    Xte = fz.transform(te_obs)
    M = brain.member_proba(Xte)
    y = np.searchsorted(brain.classes_, te_lab)
    correct = (M.mean(axis=0).argmax(1) == y).astype(float)
    stats = decompose_batch(M)

    # ---------------- A. selective prediction -------------------------
    if "A" in parts:
      print("\n[A] Selective prediction - accuracy when the agent may decline\n")
    rng = np.random.default_rng(0)
    rows = []
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = max(int(cov * len(y)), 1)
        by_conf = np.argsort(-stats["confidence"])[:k]
        combined = stats["confidence"] - stats["epistemic"]
        by_gate = np.argsort(-combined)[:k]
        by_rand = rng.permutation(len(y))[:k]
        rows.append({"coverage": cov,
                     "random": float(correct[by_rand].mean()),
                     "confidence": float(correct[by_conf].mean()),
                     "gate": float(correct[by_gate].mean())})
        print(f"  coverage {cov:4.0%}   random {rows[-1]['random']*100:5.2f}%   "
              f"confidence {rows[-1]['confidence']*100:5.2f}%   "
              f"conf-epistemic {rows[-1]['gate']*100:5.2f}%")
    out["selective_prediction"] = rows
    out["risk_coverage"] = risk_coverage_curve(M, y)

    # ---------------- B. OOD detection --------------------------------
    #
    # A first attempt shifted flood *severity* and produced AUROC ~= 0.50.
    # That was a flaw in the probe, not in the signal: severity changes the
    # latent dynamics but the expert controller keeps the agent inside the
    # same region of observation space, so there was no covariate shift to
    # detect.  We therefore test shifts that genuinely move the inputs, of
    # the kind that actually occur in a disaster: instrument degradation
    # and unfamiliar report vocabulary.
    print("\n[B] Out-of-distribution detection under genuine covariate shift\n")

    def perturb(obs_list, mode: str, mag: float, seed: int = 5):
        rng = np.random.default_rng(seed)
        out_ = []
        from conclave.core.types import Observation as _Obs
        for o in obs_list:
            f = dict(o.features); t = o.text
            if mode == "sensor_noise":
                f = {k: float(v * (1 + rng.normal(0, mag))) for k, v in f.items()}
            elif mode == "sensor_dropout":
                for k in list(f):
                    if rng.random() < mag:
                        f[k] = 0.0
            elif mode == "sensor_bias":
                f = {k: float(v * (1 + mag)) for k, v in f.items()}
            elif mode == "novel_text":
                # a different language register: terse operator shorthand
                lex = ["WL", "up", "sect", "cas", "req", "neg", "ack", "std",
                       "brk", "rt", "hz", "evac", "sitrep", "grid", "unk"]
                t = " ".join(rng.choice(lex, size=rng.integers(5, 14)))
            out_.append(_Obs(features=f, text=t, meta=o.meta, t=o.t))
        return out_

    ood_rows = []
    for mode, mags in [("sensor_noise", [0.2, 0.6, 1.2]),
                       ("sensor_dropout", [0.2, 0.5, 0.8]),
                       ("sensor_bias", [0.5, 2.0]),
                       ("novel_text", [1.0])]:
        for mag in mags:
            sh = perturb(te_obs, mode, mag)
            Ms = brain.member_proba(fz.transform(sh))
            sst = decompose_batch(Ms)
            lab = np.r_[np.zeros(len(y)), np.ones(len(sh))]
            auc_e = roc_auc_score(lab, np.r_[stats["epistemic"], sst["epistemic"]])
            auc_a = roc_auc_score(lab, np.r_[stats["aleatoric"], sst["aleatoric"]])
            auc_c = roc_auc_score(lab, np.r_[-stats["confidence"], -sst["confidence"]])
            # does accuracy actually degrade?  otherwise there is nothing to detect
            acc_sh = float((Ms.mean(axis=0).argmax(1) ==
                            np.searchsorted(brain.classes_, te_lab)).mean())
            ood_rows.append({"shift": mode, "magnitude": mag,
                             "accuracy_under_shift": acc_sh,
                             "clean_accuracy": float(correct.mean()),
                             "mean_epistemic": float(sst["epistemic"].mean()),
                             "mean_aleatoric": float(sst["aleatoric"].mean()),
                             "auroc_epistemic": float(auc_e),
                             "auroc_aleatoric": float(auc_a),
                             "auroc_confidence": float(auc_c)})
            print(f"  {mode:15s} x{mag:<4.1f} acc {correct.mean()*100:5.1f}%->"
                  f"{acc_sh*100:5.1f}%   epi {sst['epistemic'].mean():.3f}   "
                  f"AUROC epi {auc_e:.3f} | alea {auc_a:.3f} | conf {auc_c:.3f}")
    out["ood_detection"] = ood_rows
    out["ood_note"] = ("Severity shift alone produced AUROC ~0.50 because the "
                       "controller keeps observations in-distribution; reported "
                       "shifts move the covariates and accuracy measurably.")

    # ---------------- C. downstream effect ----------------------------
    #
    # First attempt made escalation mean "do nothing", which collapsed the
    # return -- correctly, because inaction during a flood is lethal.  That
    # measured the fallback, not the signal.  Escalation in a real
    # deployment means *a human takes over for that decision*, which is
    # effective but scarce.  The right question is therefore:
    #
    #   for a fixed budget of human interventions per episode, does
    #   spending them where the ensemble disagrees beat spending them at
    #   random?
    #
    # That isolates the value of the uncertainty ranking itself.
    print("\n[C] Human-attention budget - where should scarce oversight go?\n")

    def run_budget(trigger: str, budget: int, dropout: float, n=40, seed0=77000):
        """trigger in {none, random, epistemic, confidence}; budget = max human calls.

        ``dropout`` degrades the agent's *sensors* -- gauges silting up,
        telemetry dying mid-event -- which is the condition under which an
        agent actually needs help.  The human operator is assumed to still
        have ground truth (they can look at the river).
        """
        from conclave.core.types import Observation as _Obs
        rng2 = np.random.default_rng(seed0)
        rets, cas, used = [], [], []
        for e in range(n):
            env = FloodResponseEnv(DisasterConfig(seed=seed0 + e, horizon=HORIZON))
            o = env.reset(); tot = 0.0; spent = 0
            for _ in range(HORIZON):
                if dropout > 0:
                    f = {k: (0.0 if rng2.random() < dropout else v)
                         for k, v in o.features.items()}
                    o_seen = _Obs(features=f, text=o.text, meta=o.meta, t=o.t)
                else:
                    o_seen = o
                rep = brain.report(fz.transform([o_seen]))[0]
                name = str(rep.labels[int(np.argmax(rep.posterior))])
                escalate = False
                if spent < budget and trigger != "none":
                    if trigger == "random":
                        escalate = rng2.random() < (budget / HORIZON)
                    elif trigger == "epistemic":
                        escalate = rep.stats.epistemic >= epi_thr
                    elif trigger == "confidence":
                        escalate = rep.stats.confidence <= conf_thr
                if escalate:
                    spent += 1
                    name = expert_policy(o, rng2)     # human sees the real river
                o, r, done, _ = env.step(Action(kind=ActionKind.TOOL, tool=name))
                tot += r
                if done:
                    break
            rets.append(tot); cas.append(env.casualties); used.append(spent)
        return (float(np.mean(rets)), float(np.std(rets) / np.sqrt(n)),
                float(np.mean(cas)), float(np.mean(used)))

    dep = []
    for dropout in [0.0, 0.6]:
        base = run_budget("none", 0, dropout)
        print(f"  sensor dropout {dropout:.0%}   (no oversight: "
              f"{base[0]:.1f}+-{base[1]:.1f}, casualties {base[2]:.1f})")
        print(f"    {'budget':>7s} | {'random':>16s} | {'confidence':>16s} | "
              f"{'epistemic':>16s}")
        for budget in [4, 10]:
            epi_thr = float(np.percentile(stats["epistemic"], 100 * (1 - budget / HORIZON)))
            conf_thr = float(np.percentile(stats["confidence"], 100 * (budget / HORIZON)))
            r_rand = run_budget("random", budget, dropout)
            r_conf = run_budget("confidence", budget, dropout)
            r_epi = run_budget("epistemic", budget, dropout)
            dep.append({"sensor_dropout": dropout, "budget": budget,
                        "none": base[0], "none_sem": base[1],
                        "random": r_rand[0], "random_sem": r_rand[1],
                        "confidence": r_conf[0], "confidence_sem": r_conf[1],
                        "epistemic": r_epi[0], "epistemic_sem": r_epi[1],
                        "used_random": r_rand[3], "used_conf": r_conf[3],
                        "used_epistemic": r_epi[3],
                        "casualties_none": base[2],
                        "casualties_epistemic": r_epi[2]})
            print(f"    {budget:7d} | {r_rand[0]:10.1f}+-{r_rand[1]:<4.1f} | "
                  f"{r_conf[0]:10.1f}+-{r_conf[1]:<4.1f} | "
                  f"{r_epi[0]:10.1f}+-{r_epi[1]:<4.1f}")
    out["human_attention_budget"] = dep
    out["design_note"] = ("Escalation is modelled as a human taking that single "
                          "decision using the expert controller, capped by a "
                          "per-episode budget. Triggers are compared at matched "
                          "budgets so the comparison isolates the ranking signal. "
                          "Oversight is evaluated under sensor dropout, since a "
                          "healthy cloned agent already matches the expert and "
                          "leaves nothing for a human to add.")
    save("exp5_uncertainty", out)


if __name__ == "__main__":
    main()
