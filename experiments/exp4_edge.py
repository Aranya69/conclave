"""Experiment 4 - Resource cost under emulated edge constraints.

We separate what we *measured* from what we *estimated*, because
conflating the two is how resource claims get (correctly) rejected.

Measured on the host, with real limits applied
    - single-decision latency, mean / p95 / p99 / max
    - peak resident memory under an enforced RLIMIT_AS ceiling
    - serialised model size
    - whether the agent runs at all inside each board's RAM budget

Estimated (clearly labelled, with the scaling factor stated)
    - on-device latency = host latency / relative single-core score
    - energy per decision = estimated latency x board TDP

The tail matters more than the mean: an autonomous responder that is
usually fast and occasionally slow is unsafe, so p99 is reported
everywhere.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conclave.brain.ensemble import EnsembleBrain                        # noqa: E402
from conclave.brain.featurize import (BudgetedTextFeaturizer,            # noqa: E402
                                   FusedFeaturizer, TabularFeaturizer)
from conclave.edge.profile import (MEMORY_LIMIT_SUPPORTED, PROFILES,     # noqa: E402
                                constrain, cpu_calibration_score,
                                host_description, profile_call)
from experiments.common import banner, save, set_seed                 # noqa: E402
from experiments.exp3_sequential import collect                       # noqa: E402

# Published reference points for what an SLM agent needs, for context.
SLM_REFERENCE = {
    "TinyLlama-1.1B (int8)":  {"params_b": 1.1, "weights_mb": 1100},
    "Qwen2.5-0.5B (int8)":    {"params_b": 0.5, "weights_mb": 500},
    "RoBERTa-base (fp32)":    {"params_b": 0.125, "weights_mb": 500},
}


def build_agent(hash_bits: int = 13):
    obs, lab = collect(120, seed0=4242)
    fz = FusedFeaturizer(
        text=BudgetedTextFeaturizer(n_word_features=2 ** hash_bits,
                                    n_char_features=2 ** hash_bits),
        tabular=TabularFeaturizer(),
    ).fit(obs)
    brain = EnsembleBrain().fit(fz.transform(obs), lab)
    return brain, fz, obs


def main() -> None:
    set_seed()
    banner("EXPERIMENT 4 - Resource cost under emulated edge constraints")

    host = host_description()
    print(f"  host: {host['processor']} | {host['logical_cores']} cores | "
          f"{host['total_ram_gb']} GB | {host['platform']}")
    score = cpu_calibration_score(0.5)
    print(f"  host calibration score: {score:.1f} (arbitrary units)")
    if not MEMORY_LIMIT_SUPPORTED:
        print("  NOTE: this platform (likely Windows) cannot enforce a process "
              "RAM ceiling.\n        Thread caps still apply and latency numbers "
              "are valid, but the\n        \"fits in RAM\" column is advisory, "
              "not an enforced limit.")
    print()

    brain, fz, obs = build_agent()
    probe = obs[:1]

    buf = io.BytesIO(); joblib.dump({"fz": fz, "brain": brain}, buf, compress=3)
    model_bytes = buf.tell()

    def one_decision():
        X = fz.transform(probe)
        brain.report(X)

    out: Dict[str, Any] = {"host": host, "host_calibration_score": score,
                           "model_bytes": model_bytes,
                           "memory_limit_enforceable": MEMORY_LIMIT_SUPPORTED,
                           "slm_reference": SLM_REFERENCE, "profiles": []}

    print(f"  serialised agent (featuriser + ensemble): {model_bytes/1e6:.2f} MB")
    print(f"  ensemble members: {brain.footprint_bytes()}\n")

    if MEMORY_LIMIT_SUPPORTED:
        print("  MEASURED on host with the board's core count and RAM ceiling "
              "enforced")
    else:
        print("  MEASURED on host with the board's core count applied "
              "(RAM ceiling NOT enforced on this platform)")
    print(f"  {'board':<28s}{'RAM cap':>9s}{'mean':>9s}{'p95':>9s}{'p99':>9s}"
          f"{'max':>9s}{'peakRSS':>10s}{'fits':>7s}{'noise':>7s}")
    print("  " + "-" * 97)

    for key in ["host", "jetson_nano", "rpi4b_4gb", "rpi3bp", "rpi_zero2w"]:
        dev = PROFILES[key]
        try:
            # RLIMIT_AS below current usage would kill the process; we
            # enforce only where it is a genuine constraint we can honour.
            enforce = dev.ram_mb >= 1024 and MEMORY_LIMIT_SUPPORTED
            with constrain(dev, enforce_memory=enforce):
                m = profile_call(one_decision, label=dev.name, n=120, warmup=25,
                                 model_bytes=model_bytes, trials=5)
            est = m.estimate_on(dev)
            fits = "yes" if (dev.ram_mb == 0 or m.peak_rss_mb < dev.ram_mb) else "NO"
            print(f"  {dev.name:<28s}{(str(dev.ram_mb)+'MB' if dev.ram_mb else '-'):>9s}"
                  f"{m.mean_ms:>8.2f}m{m.p95_ms:>8.2f}m{m.p99_ms:>8.2f}m"
                  f"{m.max_ms:>8.2f}m{m.peak_rss_mb:>9.1f}M{fits:>7s}"
                  f"{m.noise_ratio:>7.2f}")
            out["profiles"].append({"device": dev.name, "key": key,
                                    "measured": m.as_dict(), "estimated": est,
                                    "ram_enforced": enforce})
        except MemoryError as exc:
            print(f"  {dev.name:<28s}  MemoryError under enforced cap: {exc}")
            out["profiles"].append({"device": dev.name, "key": key,
                                    "error": "MemoryError"})

    print("\n  ESTIMATED on-device latency (host latency / relative core score)")
    print(f"  {'board':<28s}{'factor':>8s}{'est mean':>11s}{'est p99':>11s}"
          f"{'est energy':>13s}")
    print("  " + "-" * 72)
    for p in out["profiles"]:
        if "estimated" not in p:
            continue
        e = p["estimated"]
        print(f"  {p['device']:<28s}{e['scaling_factor']:>8.2f}"
              f"{e['est_mean_ms']:>10.2f}m{e['est_p99_ms']:>10.2f}m"
              f"{e['est_energy_mJ']:>11.2f}mJ")

    print("\n  Context: weight storage alone for the smallest agent-capable LMs")
    for k, v in SLM_REFERENCE.items():
        ratio = v["weights_mb"] * 1e6 / model_bytes
        print(f"    {k:<26s} {v['weights_mb']:>6d} MB   "
              f"{ratio:>7.0f}x the CONCLAVE agent")

    noisy = [p["measured"]["noise_ratio"] for p in out["profiles"]
             if "measured" in p]
    if noisy and max(noisy) > 1.5:
        print(f"\n  WARNING: noise ratio up to {max(noisy):.2f} -- another process "
              f"was competing for CPU.\n           Latency figures are upper "
              f"bounds; re-run on an idle machine before quoting them.")
    else:
        print(f"\n  timing stable across trials (max noise ratio "
              f"{max(noisy) if noisy else 1.0:.2f})")

    save("exp4_edge", out)


if __name__ == "__main__":
    main()
