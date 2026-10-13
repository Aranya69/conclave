"""Experiment 8 - The NumPy-only edge runtime.

Experiment 4 found that the agent itself is 232 KB but the Python process
running it peaks at ~217 MB, because SciPy and scikit-learn are loaded only
to evaluate a few linear functions.  ``conclave/edge/runtime.py`` removes
them: the fitted committee is exported to plain arrays and evaluated with
NumPy alone.

This experiment checks three things:

1. **Equivalence.**  On 2,400 held-out flood observations, does the runtime
   reproduce the scikit-learn agent's action, posterior, uncertainty and gate
   decision?
2. **Memory.**  Peak resident memory of a fresh process running each stack,
   and whether the runtime starts and runs under an *enforced* address-space
   cap equal to a Raspberry Pi Zero 2 W's 512 MB of RAM (and a tighter
   256 MB cap).
3. **Latency.**  Single-decision latency of both stacks on the same host,
   with the same estimated scaling to the boards as Experiment 4.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conclave.brain.uncertainty import UncertaintyGate                 # noqa: E402
from conclave.edge.profile import PROFILES                             # noqa: E402
from conclave.edge.runtime import EdgeAgent, export_agent              # noqa: E402
from experiments.common import banner, save, set_seed                 # noqa: E402
from experiments.exp3_sequential import collect                       # noqa: E402
from experiments.exp4_edge import build_agent                         # noqa: E402
from experiments.exp3_sequential import train_conclave                # noqa: E402

RUNTIME = ROOT / "conclave" / "edge" / "runtime.py"
EXPORT = ROOT / "deploy" / "edge_agent_flood.npz"

# A fresh interpreter that never imports SciPy or scikit-learn.  The cap is
# applied before NumPy is imported, so it governs the whole process.
_RUNTIME_PROBE = r"""
import sys, time, json, statistics, resource
cap_mb = int(sys.argv[3])
if cap_mb > 0:
    lim = cap_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
def _vmhwm_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
import importlib.util
spec = importlib.util.spec_from_file_location("conclave_edge_runtime", sys.argv[1])
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
agent = rt.EdgeAgent(sys.argv[2])
obs = json.load(open(sys.argv[4]))
for o in obs[:25]:
    agent.decide(o["f"], o["t"])
med = []
for trial in range(5):
    ts = []
    for o in obs[:120]:
        t0 = time.perf_counter(); agent.decide(o["f"], o["t"])
        ts.append((time.perf_counter() - t0) * 1000.0)
    med.append((statistics.median(ts), sorted(ts)[int(0.99 * len(ts))]))
for o in obs:
    agent.decide(o["f"], o["t"])
best = min(med)
print(json.dumps({"median_ms": best[0], "p99_ms": best[1],
                  "peak_rss_mb": _vmhwm_mb(),
                  "sklearn_loaded": "sklearn" in sys.modules,
                  "scipy_loaded": "scipy" in sys.modules}))
"""

_FULL_PROBE = r"""
import sys, time, json, statistics, resource, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[3])
def _vmhwm_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
import joblib
from conclave.core.types import Observation
blob = joblib.load(sys.argv[1]); fz, brain = blob["fz"], blob["brain"]
obs = [Observation(features=o["f"], text=o["t"]) for o in json.load(open(sys.argv[2]))]
for o in obs[:25]:
    brain.report(fz.transform([o]))
med = []
for trial in range(5):
    ts = []
    for o in obs[:120]:
        t0 = time.perf_counter(); brain.report(fz.transform([o]))
        ts.append((time.perf_counter() - t0) * 1000.0)
    med.append((statistics.median(ts), sorted(ts)[int(0.99 * len(ts))]))
for o in obs:
    brain.report(fz.transform([o]))
best = min(med)
print(json.dumps({"median_ms": best[0], "p99_ms": best[1],
                  "peak_rss_mb": _vmhwm_mb()}))
"""


def _run(code: str, *args: str) -> Dict[str, Any]:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-c", code, *args],
                       capture_output=True, text=True, env=env)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": (r.stderr or r.stdout)[-600:]}


def main() -> None:
    set_seed()
    banner("EXPERIMENT 8 - NumPy-only edge runtime")

    # the 232 KB agent of Experiment 4, plus a gate so the gate is checked too
    brain, fz, _ = build_agent()
    gate = UncertaintyGate(epistemic_max=0.35, aleatoric_max=0.80,
                           confidence_min=0.25, margin_min=0.02, allow_gather=False)
    EXPORT.parent.mkdir(exist_ok=True)
    info = export_agent(str(EXPORT), fz, brain, gate)
    npz_kb = EXPORT.stat().st_size / 1e3
    print(f"  exported {EXPORT.relative_to(ROOT)}: {npz_kb:.1f} KB, "
          f"{info['columns_kept']:,} of {info['columns_total']:,} feature columns kept")

    # ---- 1. equivalence ------------------------------------------------
    te_obs, _ = collect(60, seed0=60000)
    edge = EdgeAgent(str(EXPORT))
    reps = brain.report(fz.transform(te_obs))
    same_action = same_gate = 0
    d_post = d_epi = d_conf = 0.0
    for o, rep in zip(te_obs, reps):
        e = edge.decide(o.features, o.text)
        same_action += int(e["action"] == str(rep.argmax))
        same_gate += int(e["gate"] == gate(rep.stats).value)
        d_post = max(d_post, float(np.abs(e["posterior"] - rep.posterior).max()))
        d_epi = max(d_epi, abs(e["stats"]["epistemic"] - rep.stats.epistemic))
        d_conf = max(d_conf, abs(e["stats"]["confidence"] - rep.stats.confidence))
    n = len(te_obs)
    eq = {"n": n, "action_agreement": same_action / n, "gate_agreement": same_gate / n,
          "max_abs_posterior_diff": d_post, "max_abs_epistemic_diff": d_epi,
          "max_abs_confidence_diff": d_conf}
    print(f"  equivalence on {n:,} held-out observations: actions "
          f"{same_action}/{n}, gate {same_gate}/{n}, max |dP| = {d_post:.2e}")

    # ---- 2 + 3. memory and latency in fresh processes --------------------
    with tempfile.TemporaryDirectory() as td:
        obs_file = Path(td) / "obs.json"
        obs_file.write_text(json.dumps([{"f": dict(o.features), "t": o.text}
                                        for o in te_obs[:600]]))
        full_file = Path(td) / "agent.joblib"
        joblib.dump({"fz": fz, "brain": brain}, full_file, compress=3)

        full = _run(_FULL_PROBE, str(full_file), str(obs_file), str(ROOT))
        rt_free = _run(_RUNTIME_PROBE, str(RUNTIME), str(EXPORT), "0", str(obs_file))
        rt_512 = _run(_RUNTIME_PROBE, str(RUNTIME), str(EXPORT), "512", str(obs_file))
        rt_256 = _run(_RUNTIME_PROBE, str(RUNTIME), str(EXPORT), "256", str(obs_file))
        full_512 = _run(_FULL_PROBE.replace(
            "import sys, time, json, statistics, resource, warnings",
            "import sys, time, json, statistics, resource, warnings\n"
            "resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))"),
            str(full_file), str(obs_file), str(ROOT))

    print(f"\n  {'stack':<38s}{'median':>9s}{'p99':>9s}{'peak RSS':>11s}")
    print("  " + "-" * 67)
    for label, r in [("scikit-learn agent (no cap)", full),
                     ("scikit-learn agent, 512 MB cap", full_512),
                     ("NumPy runtime (no cap)", rt_free),
                     ("NumPy runtime, 512 MB cap", rt_512),
                     ("NumPy runtime, 256 MB cap", rt_256)]:
        if "error" in r:
            print(f"  {label:<38s}  FAILED: {r['error'].strip().splitlines()[-1][:60]}")
        else:
            print(f"  {label:<38s}{r['median_ms']:>8.3f}m{r['p99_ms']:>8.3f}m"
                  f"{r['peak_rss_mb']:>9.1f} MB")

    boards = {}
    for key in ["jetson_nano", "rpi4b_4gb", "rpi3bp", "rpi_zero2w"]:
        d = PROFILES[key]
        boards[key] = {
            "device": d.name, "ram_mb": d.ram_mb,
            "runtime_est_median_ms": rt_free.get("median_ms", float("nan")) / d.scalar_score,
            "runtime_est_p99_ms": rt_free.get("p99_ms", float("nan")) / d.scalar_score,
            "full_est_median_ms": full.get("median_ms", float("nan")) / d.scalar_score,
            "runtime_fits": rt_free.get("peak_rss_mb", 1e9) < d.ram_mb,
            "full_fits": full.get("peak_rss_mb", 1e9) < d.ram_mb,
        }
    print("\n  estimated on-device median latency (host / relative core score)")
    for b in boards.values():
        print(f"    {b['device']:<26s} runtime {b['runtime_est_median_ms']:6.2f} ms   "
              f"scikit-learn {b['full_est_median_ms']:6.2f} ms")

    # ---- the closed-loop agent of Experiments 3 and 7, in the runtime -----
    brain7, fz7, _ = train_conclave()
    exp7_file = Path(tempfile.gettempdir()) / "edge_agent_exp7.npz"
    info7 = export_agent(str(exp7_file), fz7, brain7, gate)
    edge7 = EdgeAgent(str(exp7_file))
    reps7 = brain7.report(fz7.transform(te_obs))
    agree7 = sum(int(edge7.decide(o.features, o.text)["action"] == str(r.argmax))
                 for o, r in zip(te_obs, reps7))
    with tempfile.TemporaryDirectory() as td:
        obs_file = Path(td) / "obs.json"
        obs_file.write_text(json.dumps([{"f": dict(o.features), "t": o.text}
                                        for o in te_obs[:600]]))
        rt7 = _run(_RUNTIME_PROBE, str(RUNTIME), str(exp7_file), "0", str(obs_file))
    # closed loop with the runtime itself, on the episode seeds of Exp. 3/7
    from conclave.core.types import Action, ActionKind
    from conclave.envs.disaster import DisasterConfig, FloodResponseEnv
    from experiments.exp3_sequential import HORIZON, N_EVAL_EPISODES
    closed = {}
    for op in (0.0, 0.5):
        rets, cas = [], []
        for e in range(N_EVAL_EPISODES):
            env = FloodResponseEnv(DisasterConfig(seed=7000 + e, horizon=HORIZON,
                                                  outage_prob=op))
            o = env.reset(); tot = 0.0
            for _ in range(HORIZON):
                a = edge7.decide(o.features, o.text)["action"]
                o, r, done, _ = env.step(Action(kind=ActionKind.TOOL, tool=a))
                tot += r
                if done:
                    break
            rets.append(tot); cas.append(env.casualties)
        closed[str(op)] = {"mean_return": float(np.mean(rets)),
                           "mean_casualties": float(np.mean(cas))}
    zero = PROFILES["rpi_zero2w"]
    exp7_agent = {"closed_loop": closed,"npz_kb": exp7_file.stat().st_size / 1e3,
                  "action_agreement": agree7 / len(te_obs), "n": len(te_obs),
                  "runtime": rt7,
                  "est_ms_pi_zero2w": rt7.get("median_ms", float("nan")) / zero.scalar_score}
    print(f"\n  closed loop with the runtime: return {closed['0.0']['mean_return']:.2f} "
          f"(0% offline), {closed['0.5']['mean_return']:.2f} (73% offline)")
    print(f"  closed-loop agent (Exp. 3/7): {exp7_agent['npz_kb']:.1f} KB, actions "
          f"{agree7}/{len(te_obs)} identical, {rt7.get('median_ms', float('nan')):.3f} ms, "
          f"{rt7.get('peak_rss_mb', float('nan')):.1f} MB")

    save("exp8_edge_runtime", {
        "exp7_agent": exp7_agent,
        "export": {"path": str(EXPORT.relative_to(ROOT)), "npz_kb": npz_kb,
                   "columns_kept": info["columns_kept"],
                   "columns_total": info["columns_total"]},
        "equivalence": eq,
        "full_stack": full, "full_stack_512mb_cap": full_512,
        "runtime": rt_free, "runtime_512mb_cap": rt_512, "runtime_256mb_cap": rt_256,
        "boards": boards,
        "note": "Peak RSS is VmHWM (peak resident set of the process's own address "
                "space) of a fresh process that loads the agent and makes 600 decisions; "
                "ru_maxrss is not used because Linux carries it over from the parent "
                "across fork/exec. Caps are RLIMIT_AS limits set before NumPy is "
                "imported. Board latencies are estimates, as in Experiment 4.",
    })


if __name__ == "__main__":
    main()
