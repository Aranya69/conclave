#!/usr/bin/env python3
"""Interactive playground for poking at CONCLAVE by hand.

Run it with no arguments for a menu, or name a section directly:

    python demo.py            # menu
    python demo.py episode    # watch an agent run, decision by decision
    python demo.py route      # type a request, see which tool it picks
    python demo.py uncertainty# degrade the sensors, watch confidence collapse
    python demo.py speed      # how fast is a decision on *your* machine

Nothing here touches the network, and only `route` needs the datasets.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

# A demo should print its own output, not scikit-learn's convergence chatter.
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conclave import Agent, EnsembleBrain, EpisodicMemory, UncertaintyGate
from conclave.brain.featurize import (BudgetedTextFeaturizer, FusedFeaturizer,
                                   TabularFeaturizer)
from conclave.brain.uncertainty import decompose
from conclave.core.types import Action, ActionKind, Observation
from conclave.envs.disaster import (ACTIONS, DisasterConfig, FloodResponseEnv,
                                 expert_policy)

BAR = "=" * 72


def hr(title: str = "") -> None:
    print("\n" + BAR)
    if title:
        print(f"  {title}")
        print(BAR)


def train(n_episodes: int = 120, seed: int = 7, quiet: bool = False):
    """Behaviour-clone the scripted expert. ~5 s on a laptop."""
    if not quiet:
        print(f"  training on {n_episodes} expert episodes ...", end="", flush=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    obs, lab = [], []
    for e in range(n_episodes):
        env = FloodResponseEnv(DisasterConfig(seed=seed * 1000 + e, horizon=40))
        o = env.reset()
        for _ in range(40):
            a = expert_policy(o, rng)
            obs.append(o)
            lab.append(a)
            o, _, done, _ = env.step(Action(kind=ActionKind.TOOL, tool=a))
            if done:
                break
    fz = FusedFeaturizer(
        text=BudgetedTextFeaturizer(n_word_features=2 ** 13, n_char_features=2 ** 13),
        tabular=TabularFeaturizer(),
    ).fit(obs)
    brain = EnsembleBrain().fit(fz.transform(obs), np.array(lab))
    if not quiet:
        print(f" done in {time.perf_counter() - t0:.1f}s "
              f"({len(obs)} decisions, members: {', '.join(brain.member_names_)})")
    return brain, fz


# --------------------------------------------------------------------------
def demo_episode() -> None:
    """Watch the agent think, one tick at a time."""
    hr("EPISODE — the agent decides, and shows its working")
    brain, fz = train()
    gate = UncertaintyGate(epistemic_max=0.35, aleatoric_max=0.80,
                           confidence_min=0.25, margin_min=0.02,
                           allow_gather=False)
    agent = Agent(brain=brain, featurizer=fz, gate=gate,
                  memory=EpisodicMemory(capacity=500))

    env = FloodResponseEnv(DisasterConfig(seed=2024, horizon=40, outage_prob=0.25))
    obs = env.reset()
    total = 0.0
    print(f"\n  {'t':>3s} {'level':>6s} {'rain':>6s} {'risk':>5s} {'net':>4s} "
          f"| {'action':<16s} {'conf':>5s} {'disagr':>6s} {'gate':>9s} {'ms':>5s}")
    print("  " + "-" * 88)
    for t in range(40):
        d = agent.decide(obs)
        risk = obs.meta.get("true_risk", 0.0)
        net = "up" if obs.meta.get("online", True) else "DOWN"
        print(f"  {t:>3d} {obs.features['river_level']:>6.2f} "
              f"{obs.features['rainfall']:>6.2f} {risk:>5.2f} {net:>4s} "
              f"| {str(d.action.tool or d.action.kind.value):<16s} "
              f"{d.confidence:>5.2f} {d.disagreement:>6.2f} {d.gate:>9s} "
              f"{d.latency_ms:>5.1f}")
        obs, r, done, _ = env.step(d.action)
        total += r
        if done:
            break

    print(f"\n  episode return {total:.1f}   {env.summary()}")
    print("\n  Note the 'net' column: connectivity drops out, and nothing changes.")
    print("  A cloud-backed agent would have produced no action on those ticks.")


# --------------------------------------------------------------------------
def demo_route() -> None:
    """Type a request; watch the router pick a tool with no LLM involved."""
    hr("TOOL ROUTING — type a request, see what it picks")
    from conclave.brain.router import ToolRouter

    data_dir = Path(__file__).resolve().parent / "data" / "bfcl"
    if not data_dir.exists():
        print("  data/bfcl not found. Run:  python scripts/fetch_data.py")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiments.exp1_tool_routing import load_pairs

    print("  loading BFCL and fitting the router ...", end="", flush=True)
    t0 = time.perf_counter()
    data = [d for d in load_pairs("BFCL_v3_live_multiple") if len(d["funcs"]) > 1]
    router = ToolRouter(abstain=False).fit(
        [d["q"] for d in data], [d["funcs"] for d in data],
        [d["target"] for d in data])
    print(f" done in {time.perf_counter() - t0:.1f}s ({len(data)} requests)")

    # a small, readable tool set for the interactive part
    tools = [
        {"name": "get_weather", "description": "Current weather for a city",
         "parameters": {"properties": {"city": {"type": "string",
                                                "description": "City name"}},
                        "required": ["city"]}},
        {"name": "send_sms_alert", "description": "Broadcast an emergency SMS to a zone",
         "parameters": {"properties": {"zone": {"type": "string", "description": "Zone id"},
                                       "message": {"type": "string", "description": "Alert text"}},
                        "required": ["zone", "message"]}},
        {"name": "river_gauge_reading", "description": "Read current river water level from a gauge station",
         "parameters": {"properties": {"station": {"type": "string", "description": "Gauge station id"}},
                        "required": ["station"]}},
        {"name": "dispatch_rescue_boat", "description": "Send a rescue boat to stranded people at a location",
         "parameters": {"properties": {"location": {"type": "string", "description": "Where to send it"},
                                       "capacity": {"type": "integer", "description": "People it can carry"}},
                        "required": ["location"]}},
        {"name": "convert_currency", "description": "Convert an amount between two currencies",
         "parameters": {"properties": {"amount": {"type": "number", "description": "Amount"},
                                       "frm": {"type": "string", "description": "From currency"},
                                       "to": {"type": "string", "description": "To currency"}},
                        "required": ["amount", "frm", "to"]}},
    ]
    print("\n  Registered tools (none of these were in the training set):")
    for t in tools:
        print(f"    - {t['name']:<22s} {t['description']}")

    examples = [
        "how high is the water at station 7 right now",
        "there are people stuck on a roof in sector 4, send help",
        "warn everyone in zone B that the embankment is about to fail",
        "what is 200 dollars in taka",
    ]
    print("\n  Try these, or type your own ('q' to quit):")
    for e in examples:
        print(f"    · {e}")
    print()

    while True:
        try:
            q = input("  request> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"q", "quit", "exit"}:
            break
        t0 = time.perf_counter()
        res = router.route(q, tools)
        ms = (time.perf_counter() - t0) * 1000
        ranked = sorted(res.scores.items(), key=lambda kv: -kv[1])
        top, second = ranked[0][1], ranked[1][1]
        margin = top - second
        # A flat score distribution means "none of these really fit" -- which is
        # what the abstention head is for.  Show that judgement rather than
        # pretending the argmax is always a real answer.
        verdict = (f"-> {res.tool}" if margin >= 0.15
                   else f"-> {res.tool}  (WEAK — would abstain: margin {margin:.2f})")
        print(f"    {verdict}   ({ms:.2f} ms, disagreement {res.stats.epistemic:.2f})")
        for name, score in ranked:
            bar = "#" * int(score * 34)
            print(f"       {name:<22s} {score:5.3f} {bar}")
        if margin < 0.15:
            print("       ^ scores are nearly flat: the router has no real winner.")
            print("         In an agent you set abstain=True and it declines instead")
            print("         of guessing. Lexical features cannot capture every domain")
            print("         (currency names, here) — that is a real documented limit.")
        print()


# --------------------------------------------------------------------------
def demo_uncertainty() -> None:
    """Break the sensors and watch the ensemble notice."""
    hr("UNCERTAINTY — degrade the input, watch the committee disagree")
    brain, fz = train()

    env = FloodResponseEnv(DisasterConfig(seed=99, horizon=40))
    obs = env.reset()
    for _ in range(12):                     # get somewhere interesting
        obs, _, _, _ = env.step(Action(kind=ActionKind.TOOL, tool="monitor"))

    rng = np.random.default_rng(0)
    print(f"\n  {'sensors dropped':>16s} | {'action':<16s} {'conf':>6s} "
          f"{'aleatoric':>10s} {'epistemic':>10s}")
    print("  " + "-" * 68)
    for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        feats = {k: (0.0 if rng.random() < frac else v)
                 for k, v in obs.features.items()}
        degraded = Observation(features=feats, text=obs.text, meta=obs.meta)
        rep = brain.report(fz.transform([degraded]))[0]
        s = rep.stats
        print(f"  {frac:>15.0%} | {str(rep.argmax):<16s} {s.confidence:>6.2f} "
              f"{s.aleatoric:>10.2f} {s.epistemic:>10.2f}")

    print("\n  Confidence falls as the input degrades — that is the signal the")
    print("  gate uses to abstain or escalate instead of guessing.")
    print("\n  Now the same thing with per-member votes, at 60% dropout:")
    feats = {k: (0.0 if rng.random() < 0.6 else v) for k, v in obs.features.items()}
    rep = brain.report(fz.transform([Observation(features=feats, text=obs.text)]))[0]
    for name, votes in rep.member_mapping(brain.member_names_).items():
        top = sorted(votes.items(), key=lambda kv: -kv[1])[:3]
        print(f"    {name:<8s} " + "  ".join(f"{k}={v:.2f}" for k, v in top))


# --------------------------------------------------------------------------
def demo_speed() -> None:
    """How fast is one decision on this machine?"""
    hr("SPEED — single-decision latency on your hardware")
    import platform

    from conclave.edge.profile import (MEMORY_LIMIT_SUPPORTED, PROFILES,
                                    cpu_calibration_score, host_description,
                                    profile_call)

    host = host_description()
    print(f"  {host['processor']} | {host['logical_cores']} cores | "
          f"{host.get('total_ram_gb', '?')} GB | {platform.system()}")
    if not MEMORY_LIMIT_SUPPORTED:
        print("  (this platform cannot enforce a RAM ceiling; thread caps still apply)")

    brain, fz = train(quiet=False)
    env = FloodResponseEnv(DisasterConfig(seed=5))
    obs = env.reset()

    def one():
        brain.report(fz.transform([obs]))

    m = profile_call(one, label="decision", n=200, warmup=25)
    print(f"\n  mean {m.mean_ms:.2f} ms | p95 {m.p95_ms:.2f} ms | "
          f"p99 {m.p99_ms:.2f} ms | max {m.max_ms:.2f} ms")
    print(f"  agent footprint: {brain.footprint_bytes()['_total'] / 1e3:.0f} KB")

    score = cpu_calibration_score(0.4)
    print(f"\n  your CPU score: {score:.0f}")
    print(f"  {'board':<24s}{'est. mean':>11s}{'est. p99':>11s}")
    for key in ["rpi4b_4gb", "rpi3bp", "rpi_zero2w"]:
        e = m.estimate_on(PROFILES[key])
        print(f"  {PROFILES[key].name:<24s}{e['est_mean_ms']:>10.1f}m"
              f"{e['est_p99_ms']:>10.1f}m")
    print("\n  (board figures are scaled estimates, not measurements — see the paper)")


# --------------------------------------------------------------------------
DEMOS = {
    "episode": ("Watch an agent run an episode, decision by decision", demo_episode),
    "route": ("Type a request, see which tool it picks (needs data/)", demo_route),
    "uncertainty": ("Degrade the sensors, watch confidence collapse", demo_uncertainty),
    "speed": ("Measure single-decision latency on your machine", demo_speed),
}


def main() -> None:
    if len(sys.argv) > 1:
        key = sys.argv[1].lower()
        if key in DEMOS:
            DEMOS[key][1]()
            return
        print(f"unknown demo {key!r}. options: {', '.join(DEMOS)}")
        sys.exit(1)

    hr("CONCLAVE — interactive demos")
    keys = list(DEMOS)
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {k:<12s} {DEMOS[k][0]}")
    print("  q. quit")
    while True:
        try:
            c = input("\n  choose> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if c in {"q", "quit", "exit"}:
            return
        if c.isdigit() and 1 <= int(c) <= len(keys):
            DEMOS[keys[int(c) - 1]][1]()
            return
        if c in DEMOS:
            DEMOS[c][1]()
            return
        print("  ?")


if __name__ == "__main__":
    main()
