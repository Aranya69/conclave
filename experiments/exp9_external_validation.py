"""Experiment 9 - external validation on observed rainfall (NASA POWER).

Question
--------
Every flood result so far comes from episodes of our own simulator.  Does a
policy learned there still work when the weather is real?  Here the agents
of Experiment 7 -- trained only on synthetic episodes, not retrained, not
re-tuned -- are run on episodes driven by observed daily rainfall at six
flood-prone districts of Bangladesh (NASA POWER, MERRA-2, 1991-2024).

Design
------
* Data       NASA POWER daily corrected precipitation (PRECTOTCORR, mm/day)
             at Sylhet, Sunamganj, Moulvibazar, Netrokona (north-east flash-
             flood basin), Feni (south-east) and Kurigram (Brahmaputra).
             File: data/external/nasa_power_bd_daily_rain_1991_2024.csv
* Split      Calibration 1991-2014 (fits the two conversions below; no agent
             sees it).  Test 2015-2024, April-October, nine 40-day windows
             per station and year -> 6 x 10 x 9 = 540 real episodes.
* Forcing    quantile mapping (rain -> sensor scale) and an antecedent
             precipitation index (rain -> river stage); see
             conclave/envs/real_forcing.py.  Main setting kappa = 0.85, flood
             stage at the 95th percentile of calibration API; a 3 x 3 grid of
             (kappa, quantile) is the sensitivity analysis.
* Agents     the nine offline ML agents of Experiment 7 with identical
             training, plus two references: fixed gauge thresholds (no
             learning) and the expert controller that reads the true risk
             (an upper bound no deployable agent has).
* Conditions network up (0 %) and down (~73 % of steps offline); field
             reports on, and off (gauges only -- the reports of the task are
             written from the true risk, so this condition removes any help
             they could give).

Run (each part resumes from a cache, so it can be split into short jobs):
    python experiments/exp9_external_validation.py train
    python experiments/exp9_external_validation.py main
    python experiments/exp9_external_validation.py accuracy
    python experiments/exp9_external_validation.py sens
    python experiments/exp9_external_validation.py cases
    python experiments/exp9_external_validation.py summary
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conclave.brain.uncertainty import UncertaintyGate, decompose_batch  # noqa: E402
from conclave.core.types import Action, ActionKind, Observation        # noqa: E402
from conclave.brain.ensemble import EnsembleBrain                      # noqa: E402
from conclave.brain.featurize import (BudgetedTextFeaturizer,          # noqa: E402
                                      FusedFeaturizer, TabularFeaturizer)
from conclave.envs.disaster import (DisasterConfig, FloodResponseEnv,   # noqa: E402
                                    expert_policy)
from conclave.envs.real_forcing import (FLOOD_LEVEL, ForcingModel,      # noqa: E402
                                        ReplayFloodEnv, flood_onset)
from experiments.common import DATA, RESULTS, banner, save              # noqa: E402
from experiments.exp3_sequential import (HORIZON, collect, static_rules,  # noqa: E402
                                         train_conclave)
from experiments.exp7_ml_agents import flood_agents, paired_ci          # noqa: E402

warnings.filterwarnings("ignore")

CSV = DATA / "external" / "nasa_power_bd_daily_rain_1991_2024.csv"
STATIONS = {                                   # name: (lat, lon), as requested
    "sylhet": (24.90, 91.87), "sunamganj": (25.07, 91.40),
    "moulvibazar": (24.48, 91.78), "netrokona": (24.88, 90.73),
    "feni": (23.01, 91.40), "kurigram": (25.81, 89.64),
}
CALIB_YEARS = (1991, 2014)
TEST_YEARS = (2015, 2024)
SEASON = (4, 10)                               # April-October
WINDOW_STARTS = [(4, 1), (4, 21), (5, 11), (5, 31), (6, 20),
                 (7, 10), (7, 30), (8, 19), (9, 8)]
LOOKAHEAD = 10
MAIN = (0.85, 0.95)
GRID = [(k, q) for k in (0.80, 0.85, 0.90) for q in (0.90, 0.95, 0.98)]
OUTAGES = [0.0, 0.5]
SEED0 = 90000
AGENT_ORDER = ["CONCLAVE (ours)", "Decision tree", "LightGBM", "Logistic regression",
               "MLP", "Hard-vote ensemble", "Random forest", "CBR (k-NN)",
               "Naive Bayes"]
REFERENCES = ["Gauge thresholds (rules)", "Expert with true risk (upper bound)",
              "Cloud agent (expert, needs network)"]
# report conditions: 1 = field reports present (as in training);
# 2 = gauges only, agents trained on gauges only (no report channel at all);
# 0 = reports lost at test time, agents not retrained (missing modality).
CONDITIONS = (1, 2, 0)
CASES = [("sylhet", "2022-05-01", "Sylhet, May-June 2022"),
         ("sunamganj", "2022-05-31", "Sunamganj, June 2022"),
         ("feni", "2024-07-27", "Feni, August 2024")]

CACHE = Path(tempfile.gettempdir()) / "exp9_cache"
CACHE.mkdir(exist_ok=True)
UNITS = CACHE / "units.json"


# ------------------------------------------------------------------ data
def load_rain() -> pd.DataFrame:
    d = pd.read_csv(CSV, parse_dates=["date"]).set_index("date")
    return d[list(STATIONS)]


def synthetic_rain(n_episodes: int = 160, seed0: int = 1000) -> np.ndarray:
    """True rainfall of the training episodes (the sensor scale agents know)."""
    rng = np.random.default_rng(seed0)
    vals = []
    for e in range(n_episodes):
        env = FloodResponseEnv(DisasterConfig(seed=seed0 + e, horizon=HORIZON))
        o = env.reset()
        vals.append(env.rain)
        for _ in range(HORIZON):
            o, _, done, _ = env.step(Action(kind=ActionKind.TOOL, tool=expert_policy(o, rng)))
            vals.append(env.rain)
            if done:
                break
    return np.asarray(vals)


def _season_mask(idx: pd.DatetimeIndex, years: Tuple[int, int]) -> np.ndarray:
    m = ((idx.year >= years[0]) & (idx.year <= years[1])
            & (idx.month >= SEASON[0]) & (idx.month <= SEASON[1]))
    return np.asarray(m)


def fit_forcing(rain: pd.DataFrame, kappa: float, quantile: float,
                syn: np.ndarray) -> ForcingModel:
    mask = _season_mask(rain.index, CALIB_YEARS)
    return ForcingModel.fit({s: rain[s].values for s in rain},
                            {s: mask for s in rain}, syn,
                            kappa=kappa, quantile=quantile)


def build_episodes(rain: pd.DataFrame, fm: ForcingModel,
                   years: Tuple[int, int] = TEST_YEARS) -> List[Dict[str, Any]]:
    eps = []
    for s in STATIONS:
        stage_full = fm.stage(rain[s].values)
        sensor_full = fm.sensor_rain(rain[s].values)
        for y in range(years[0], years[1] + 1):
            for m, d in WINDOW_STARTS:
                i = rain.index.get_loc(pd.Timestamp(y, m, d))
                sl = slice(i, i + HORIZON + 1 + LOOKAHEAD)
                st = stage_full[sl]
                eps.append({"station": s, "start": f"{y}-{m:02d}-{d:02d}",
                            "sensor_rain": sensor_full[sl], "stage": st,
                            "rain_mm": rain[s].values[sl],
                            "onset": flood_onset(st, HORIZON)})
    return eps


# ------------------------------------------------------------------ agents
def _mfile(name: str, gauges: bool = False) -> Path:
    return CACHE / (name.replace(" ", "_").replace("(", "").replace(")", "")
                    .replace(",", "") + ("_gauges" if gauges else "") + ".joblib")


class GaugeFeaturizer:
    """The twelve gauge and state readings only (dense, standardised).

    With no report channel the input is a small dense table, so CONCLAVE
    uses the library's tabular committee (logistic regression, random forest,
    extra trees; ``default_members(sparse_input=False)``).
    """

    def fit(self, obs: List[Observation]) -> "GaugeFeaturizer":
        self.tab_ = TabularFeaturizer().fit([o.features for o in obs])
        return self

    def transform(self, obs: List[Observation]) -> np.ndarray:
        return self.tab_.transform([o.features for o in obs])


def _blank(obs: List[Observation]) -> List[Observation]:
    return [Observation(features=o.features, text="", meta=o.meta) for o in obs]


def load_agents(train_missing: bool = False, budget_end: float = float("inf"),
                gauges: bool = False):
    """CONCLAVE and the eight other ML agents of Experiment 7, same training.

    ``gauges=True`` trains the same agents on the same episodes with the
    report channel removed, so they learn to act from the gauges alone.
    """
    if not gauges:
        brain_c, fz, gate = train_conclave()
        agents = {"CONCLAVE (ours)": brain_c}
        corpus = None
    else:
        o_, l_ = collect(160, seed0=1000)
        o_ = _blank(o_)
        fz = GaugeFeaturizer().fit(o_)
        corpus = (fz.transform(o_), l_)
        gate = train_conclave()[2]
        f = _mfile("CONCLAVE (ours)", True)
        if f.exists():
            agents = {"CONCLAVE (ours)": joblib.load(f)}
        else:
            agents = {"CONCLAVE (ours)": EnsembleBrain(weighting="uniform").fit(*corpus)}
            joblib.dump(agents["CONCLAVE (ours)"], f)
    for name in AGENT_ORDER[1:]:
        f = _mfile(name, gauges)
        if f.exists():
            agents[name] = joblib.load(f)
        elif train_missing and time.perf_counter() < budget_end:
            if corpus is None:
                o_, l_ = collect(160, seed0=1000)
                corpus = (fz.transform(o_), l_)
            t0 = time.perf_counter()
            agents[name] = flood_agents()[name].fit(*corpus)
            joblib.dump(agents[name], f)
            print(f"  fitted {name:<22s}{' (gauges)' if gauges else ''} "
                  f"{time.perf_counter() - t0:6.1f} s", flush=True)
    return agents, fz, gate


N_REAL_TRAIN = 160                    # same number of episodes as the simulator corpus


def collect_real(reports: bool, n_episodes: int = N_REAL_TRAIN, seed0: int = 50000):
    """Expert demonstrations on observed-rainfall episodes of the calibration
    years (1991-2014) -- never the test years."""
    rain = load_rain(); syn = synthetic_rain()
    eps = build_episodes(rain, fit_forcing(rain, *MAIN, syn), years=CALIB_YEARS)
    pick = np.random.default_rng(5).choice(len(eps), size=n_episodes, replace=False)
    obs, lab = [], []
    for j, k in enumerate(sorted(pick)):
        e = eps[k]
        env = ReplayFloodEnv(e["sensor_rain"], e["stage"],
                             DisasterConfig(seed=seed0 + j, horizon=HORIZON), reports=reports)
        rng = np.random.default_rng(seed0 + j)
        o = env.reset()
        for _ in range(HORIZON):
            a = expert_policy(o, rng)
            obs.append(o); lab.append(a)
            o, _, _, _ = env.step(Action(kind=ActionKind.TOOL, tool=a))
    return obs, np.asarray(lab), [eps[k]["station"] + " " + eps[k]["start"] for k in sorted(pick)]


def load_real_agents(gauges: bool, train_missing: bool = False,
                     budget_end: float = float("inf"), mix: bool = False):
    """The same nine agents, trained on past observed rainfall (1991-2014).

    ``mix=True`` trains on the simulator corpus *plus* the past observed
    episodes (sim-to-real augmentation): the simulator covers rare actions,
    the observed record covers realistic river behaviour.
    """
    o_, l_, _ = collect_real(reports=not gauges)
    if mix:
        so, sl = collect(160, seed0=1000)
        if gauges:
            so = _blank(so)
        o_, l_ = list(so) + list(o_), np.r_[sl, l_]
    if gauges:
        fz = GaugeFeaturizer().fit(o_)
    else:
        fz = FusedFeaturizer(
            text=BudgetedTextFeaturizer(n_word_features=2 ** 13, n_char_features=2 ** 13),
            tabular=TabularFeaturizer()).fit(o_)
    X = fz.transform(o_)
    agents = {}
    tag = ("_mix" if mix else "_real") + ("_gauges" if gauges else "")
    for name in AGENT_ORDER:
        f = CACHE / (_mfile(name).stem + tag + ".joblib")
        if f.exists():
            agents[name] = joblib.load(f)
        elif train_missing and time.perf_counter() < budget_end:
            t0 = time.perf_counter()
            est = (EnsembleBrain(weighting="uniform") if name == AGENT_ORDER[0]
                   else flood_agents()[name])
            agents[name] = est.fit(X, l_)
            joblib.dump(agents[name], f)
            print(f"  fitted {name:<22s} ({tag[1:]}) {time.perf_counter() - t0:6.1f} s",
                  flush=True)
    return agents, fz, None


# ------------------------------------------------------------------ rollout
def rollout(policy: str, brain, fz, episodes: List[Dict[str, Any]],
            outage: float, reports: bool, record: bool = False) -> Dict[str, Any]:
    """Run all episodes side by side (identical to running them one by one)."""
    n = len(episodes)
    envs = [ReplayFloodEnv(e["sensor_rain"], e["stage"],
                           DisasterConfig(seed=SEED0 + k, horizon=HORIZON,
                                          outage_prob=outage), reports=reports)
            for k, e in enumerate(episodes)]
    obs = [env.reset() for env in envs]
    rngs = [np.random.default_rng(SEED0 + k) for k in range(n)]
    totals = np.zeros(n)
    first_alert = np.full(n, -1)
    trace: List[List[str]] = [[] for _ in range(n)]
    for t in range(HORIZON):
        if policy == "brain":
            acts = brain.classes_[brain.predict_proba(fz.transform(obs)).argmax(axis=1)]
        elif policy == "rules":
            acts = [static_rules(o) for o in obs]
        elif policy == "cloud":                 # expert, but only when online
            acts = [expert_policy(o, r) if o.meta.get("online", True) else "_abstain"
                    for o, r in zip(obs, rngs)]
        else:                                   # expert with the true risk
            acts = [expert_policy(o, r) for o, r in zip(obs, rngs)]
        for i, env in enumerate(envs):
            a = str(acts[i])
            if a == "issue_alert" and first_alert[i] < 0:
                first_alert[i] = t
            if record:
                trace[i].append(a)
            act = (Action(kind=ActionKind.ABSTAIN) if a == "_abstain"
                   else Action(kind=ActionKind.TOOL, tool=a))
            obs[i], r, done, _ = env.step(act)
            totals[i] += r
    onset = np.array([e["onset"] if e["onset"] is not None else -1 for e in episodes])
    out = {
        "returns": totals.tolist(),
        "casualties": [env.casualties for env in envs],
        "false_alarms": [env.false_alarms for env in envs],
        "evacuated_frac": [env.evacuated / env.cfg.population for env in envs],
        "first_alert": first_alert.tolist(),
        "onset": onset.tolist(),
        "offline_fraction": float(np.mean([env.offline_steps / HORIZON for env in envs])),
    }
    if record:
        out["actions"] = trace
    return out


def warning_stats(u: Dict[str, Any]) -> Dict[str, float]:
    """Timeliness of warnings on episodes whose flood starts inside the window."""
    on = np.asarray(u["onset"]); fa = np.asarray(u["first_alert"])
    sel = on >= 1                                # not already flooded on day 0
    hit = sel & (fa >= 0) & (fa < on)
    lead = (on - fa)[hit]
    no_flood = on < 0
    return {
        "n_flood_episodes": int(sel.sum()),
        "n_no_flood_episodes": int(no_flood.sum()),
        "warned_in_time_pct": float(100 * hit.sum() / max(sel.sum(), 1)),
        "median_lead_days": float(np.median(lead)) if len(lead) else float("nan"),
        "alert_without_flood_pct": float(100 * ((fa >= 0) & no_flood).sum()
                                         / max(no_flood.sum(), 1)),
    }


def _load_units() -> Dict[str, Any]:
    return json.loads(UNITS.read_text()) if UNITS.exists() else {}


def _save_units(u: Dict[str, Any]) -> None:
    UNITS.write_text(json.dumps(u))


def run_units(keys: List[Tuple[Tuple[float, float], str, float, int]],
              budget_s: float) -> None:
    """Evaluate (setting, agent, outage, condition) units not yet cached."""
    units = _load_units()
    todo = [k for k in keys if _ukey(*k) not in units]
    if not todo:
        print("  all units done", flush=True)
        return
    end = time.perf_counter() + budget_s
    rain = load_rain(); syn = synthetic_rain()
    pools: Dict[bool, Any] = {}
    eps_cache: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for setting, agent, op, cond in todo:
        if time.perf_counter() > end:
            print("  budget used -- run again for the remaining units", flush=True)
            break
        if setting not in eps_cache:
            eps_cache[setting] = build_episodes(rain, fit_forcing(rain, *setting, syn))
        eps = eps_cache[setting]
        reports = cond in (1, 3, 5)
        t0 = time.perf_counter()
        if agent in REFERENCES:
            kind = {REFERENCES[0]: "rules", REFERENCES[1]: "expert",
                    REFERENCES[2]: "cloud"}[agent]
            u = rollout(kind, None, None, eps, op, reports)
        else:
            if cond not in pools:
                pools[cond] = (load_real_agents(gauges=cond == 4) if cond in (3, 4)
                               else load_real_agents(gauges=cond == 6, mix=True)
                               if cond in (5, 6) else load_agents(gauges=cond == 2))
            agents, fz, _ = pools[cond]
            if agent not in agents:
                print(f"  {agent} not trained yet -- run 'train'", flush=True)
                continue
            u = rollout("brain", agents[agent], fz, eps, op, reports)
        units[_ukey(setting, agent, op, cond)] = u
        _save_units(units)
        print(f"  {str(setting):12s} {agent:<38s} off={op:.1f} cond={cond} "
              f"ret={np.mean(u['returns']):7.2f}  {time.perf_counter() - t0:5.1f} s",
              flush=True)


def _ukey(setting, agent, op, rep) -> str:
    return f"{setting[0]}|{setting[1]}|{agent}|{op}|{int(rep)}"


ALL_POLICIES = AGENT_ORDER + REFERENCES


def main_keys():
    return [(MAIN, a, op, c) for c in CONDITIONS for op in OUTAGES
            for a in ALL_POLICIES]


def temporal_keys():
    """Agents trained on 1991-2014 observed rainfall, tested on 2015-2024."""
    return [(MAIN, a, 0.5, c) for c in (3, 4, 5, 6) for a in AGENT_ORDER]


def sens_keys():
    return [(s, a, 0.5, c) for c in (1, 2) for s in GRID if s != MAIN
            for a in ALL_POLICIES]


# ------------------------------------------------------------------ accuracy
def part_accuracy() -> Dict[str, Any]:
    """One-step decisions on states the expert visits in the real episodes."""
    banner("EXPERIMENT 9 - decision accuracy on observed-rainfall states")
    rain = load_rain(); syn = synthetic_rain()
    eps = build_episodes(rain, fit_forcing(rain, *MAIN, syn))
    agents, fz, gate = load_agents()
    out: Dict[str, Any] = {}
    for rep in (True, False):
        obs, lab = [], []
        for k, e in enumerate(eps):
            env = ReplayFloodEnv(e["sensor_rain"], e["stage"],
                                 DisasterConfig(seed=SEED0 + k, horizon=HORIZON),
                                 reports=rep)
            rng = np.random.default_rng(SEED0 + k)
            o = env.reset()
            for _ in range(HORIZON):
                a = expert_policy(o, rng)
                obs.append(o); lab.append(a)
                o, _, _, _ = env.step(Action(kind=ActionKind.TOOL, tool=a))
        lab = np.asarray(lab)
        X = fz.transform(obs)
        acc = {}
        for name in AGENT_ORDER:
            if name in agents:
                acc[name] = float(np.mean(agents[name].predict(X) == lab))
        b = agents["CONCLAVE (ours)"]
        M = b.member_proba(X)
        s = decompose_batch(M)
        pred = b.classes_[b.predict_proba(X).argmax(1)]
        wrong = (pred != lab).astype(int)
        from sklearn.metrics import roc_auc_score
        epi = np.asarray(s["epistemic"])
        act = epi <= gate.epistemic_max
        unc = {
            "mean_epistemic": float(epi.mean()),
            "auroc_epistemic_flags_errors": float(roc_auc_score(wrong, epi))
            if 0 < wrong.sum() < len(wrong) else None,
            "gate_act_pct": float(100 * act.mean()),
            "accuracy_when_gate_acts": float(1 - wrong[act].mean()) if act.any() else None,
            "accuracy_when_gate_escalates": float(1 - wrong[~act].mean()) if (~act).any() else None,
        }
        out["reports" if rep else "gauges_only"] = {
            "n_decisions": int(len(lab)), "accuracy": acc, "conclave_uncertainty": unc,
            "label_mix": {k: int(v) for k, v in zip(*np.unique(lab, return_counts=True))}}
        print(f"  reports={rep}: n={len(lab):,}  " + "  ".join(
            f"{k.split()[0]}={v*100:.1f}" for k, v in acc.items()), flush=True)
        print(f"    CONCLAVE epistemic {unc['mean_epistemic']:.3f}  "
              f"AUROC(errors) {unc['auroc_epistemic_flags_errors']}", flush=True)
    # epistemic uncertainty on held-out synthetic decisions, for contrast
    te_obs, te_lab = collect(60, seed0=60000)
    Xs = fz.transform(te_obs)
    out["synthetic_mean_epistemic"] = float(np.mean(
        decompose_batch(agents["CONCLAVE (ours)"].member_proba(Xs))["epistemic"]))
    (CACHE / "accuracy.json").write_text(json.dumps(out))
    return out


# ------------------------------------------------------------------ cases
def part_cases() -> Dict[str, Any]:
    rain = load_rain(); syn = synthetic_rain()
    fm = fit_forcing(rain, *MAIN, syn)
    agents, fz, _ = load_agents()
    out = []
    for st, start, label in CASES:
        i = rain.index.get_loc(pd.Timestamp(start))
        sl = slice(i, i + HORIZON + 1 + LOOKAHEAD)
        e = {"station": st, "start": start, "sensor_rain": fm.sensor_rain(rain[st].values)[sl],
             "stage": fm.stage(rain[st].values)[sl], "rain_mm": rain[st].values[sl]}
        e["onset"] = flood_onset(e["stage"], HORIZON)
        row = {"station": st, "start": start, "label": label, "onset": e["onset"],
               "rain_mm": [float(x) for x in e["rain_mm"][:HORIZON + 1]],
               "stage": [float(x) for x in e["stage"][:HORIZON + 1]], "agents": {}}
        for name in ("CONCLAVE (ours)", "LightGBM", "Decision tree"):
            u = rollout("brain", agents[name], fz, [e], 0.5, True, record=True)
            row["agents"][name] = {"actions": u["actions"][0], "return": u["returns"][0],
                                   "casualties": u["casualties"][0],
                                   "first_alert": u["first_alert"][0]}
        for ref, kind in ((REFERENCES[0], "rules"), (REFERENCES[1], "expert")):
            u = rollout(kind, None, None, [e], 0.5, True, record=True)
            row["agents"][ref] = {"actions": u["actions"][0], "return": u["returns"][0],
                                  "casualties": u["casualties"][0],
                                  "first_alert": u["first_alert"][0]}
        out.append(row)
        print(f"  {label:<24s} onset day {e['onset']}  " + "  ".join(
            f"{k.split()[0]}: alert d{v['first_alert']} ret {v['return']:.1f}"
            for k, v in row["agents"].items()), flush=True)
    (CACHE / "cases.json").write_text(json.dumps(out))
    return {"cases": out}


# ------------------------------------------------------------------ summary
def summarise() -> Dict[str, Any]:
    banner("EXPERIMENT 9 - external validation on observed rainfall (summary)")
    units = _load_units()
    rain = load_rain(); syn = synthetic_rain()
    fm = fit_forcing(rain, *MAIN, syn)
    eps = build_episodes(rain, fm)
    onset = np.array([e["onset"] if e["onset"] is not None else -1 for e in eps])
    data = {
        "source": "NASA POWER daily API v2, parameter PRECTOTCORR (MERRA-2), community AG",
        "file": str(CSV.relative_to(CSV.parents[2])),
        "stations": STATIONS, "calibration_years": CALIB_YEARS, "test_years": TEST_YEARS,
        "season_months": SEASON, "window_days": HORIZON, "n_windows_per_year": len(WINDOW_STARTS),
        "n_episodes": len(eps),
        "episodes_with_flood_in_window": int((onset >= 0).sum()),
        "episodes_flooded_on_day0": int((onset == 0).sum()),
        "episodes_without_flood": int((onset < 0).sum()),
        "forcing_main": fm.as_dict(),
        "calibration_mean_rain_mm": float(np.mean(rain.values[_season_mask(rain.index, CALIB_YEARS)])),
        "test_mean_rain_mm": float(np.mean(rain.values[_season_mask(rain.index, TEST_YEARS)])),
    }
    cond_names = {1: "sim-trained, gauges + reports",
                  2: "sim-trained, gauges only",
                  0: "sim-trained with reports, reports lost at test",
                  3: "trained on observed 1991-2014, gauges + reports",
                  4: "trained on observed 1991-2014, gauges only",
                  5: "sim + observed 1991-2014, gauges + reports",
                  6: "sim + observed 1991-2014, gauges only"}
    rows = []
    for a in ALL_POLICIES:
        r = {"agent": a}
        for cond in cond_names:
            for op in OUTAGES:
                k = _ukey(MAIN, a, op, cond)
                if k not in units:
                    continue
                u = units[k]
                tag = f"c{cond}_{int(round(op * 146))}"          # offline 0 or 73 %
                rets = np.asarray(u["returns"])
                r[f"return_{tag}"] = float(rets.mean())
                r[f"sem_{tag}"] = float(rets.std() / np.sqrt(len(rets)))
                r[f"casualties_{tag}"] = float(np.mean(u["casualties"]))
                r[f"false_alarms_{tag}"] = float(np.mean(u["false_alarms"]))
                r[f"offline_{tag}"] = float(u["offline_fraction"])
                for kk, vv in warning_stats(u).items():
                    r[f"{kk}_{tag}"] = vv
        rows.append(r)
    tests = []
    for cond in cond_names:
        for op in OUTAGES:
            ka = _ukey(MAIN, AGENT_ORDER[0], op, cond)
            if ka not in units:
                continue
            for a in ALL_POLICIES[1:]:
                kb = _ukey(MAIN, a, op, cond)
                if kb in units:
                    m, lo, hi = paired_ci(units[ka]["returns"], units[kb]["returns"])
                    mc, loc, hic = paired_ci(units[ka]["casualties"], units[kb]["casualties"])
                    tests.append({"vs": a, "outage_prob": op, "condition": cond,
                                  "delta_mean": m, "ci_low": lo, "ci_high": hi,
                                  "significant": bool(lo > 0 or hi < 0),
                                  "delta_casualties": mc, "cas_ci_low": loc,
                                  "cas_ci_high": hic,
                                  "cas_significant": bool(loc > 0 or hic < 0)})
    sens = []
    for s_ in GRID:
        for cond in (1, 2):
            op = 0.5
            rets = {a: units.get(_ukey(s_, a, op, cond), {}).get("returns") for a in ALL_POLICIES}
            if any(rets[a] is None for a in AGENT_ORDER):
                continue
            cas = {a: float(np.mean(units[_ukey(s_, a, op, cond)]["casualties"]))
                   for a in ALL_POLICIES if rets[a] is not None}
            means = {a: float(np.mean(rets[a])) for a in ALL_POLICIES if rets[a] is not None}
            ml = {a: means[a] for a in AGENT_ORDER}
            rank = 1 + sorted(ml, key=ml.get, reverse=True).index(AGENT_ORDER[0])
            cas_rank = 1 + sorted(AGENT_ORDER, key=lambda x: cas[x]).index(AGENT_ORDER[0])
            best_other = max((a for a in AGENT_ORDER[1:]), key=ml.get)
            m, lo, hi = paired_ci(rets[AGENT_ORDER[0]], rets[best_other])
            fm_s = fit_forcing(rain, *s_, syn)
            on_s = [e["onset"] for e in build_episodes(rain, fm_s)]
            sens.append({"kappa": s_[0], "quantile": s_[1], "condition": cond,
                         "means": means, "casualties": cas,
                         "conclave_rank_among_ml": rank,
                         "conclave_casualty_rank_among_ml": cas_rank,
                         "best_other": best_other,
                         "delta_vs_best_other": m, "ci_low": lo, "ci_high": hi,
                         "flood_episode_pct": float(100 * np.mean([o is not None for o in on_s]))})
    # synthetic reference (Experiment 7) for the generalisation gap
    e7 = json.loads((RESULTS / "exp7_ml_agents.json").read_text())["flood"]
    synthetic = {r["agent"]: {"accuracy": r["accuracy"],
                              "return_offline0": r["return_offline0"],
                              "return_offline73": r["return_offline73"]}
                 for r in e7["rows"] if r["agent"] in AGENT_ORDER}
    out = {"data": data, "conditions": cond_names, "rows": rows, "paired_tests": tests,
           "sensitivity": sens, "synthetic_exp7": synthetic,
           "n_real_training_episodes": N_REAL_TRAIN}
    if (CACHE / "accuracy.json").exists():
        out["accuracy"] = json.loads((CACHE / "accuracy.json").read_text())
    if (CACHE / "cases.json").exists():
        out["cases"] = json.loads((CACHE / "cases.json").read_text())
    for cond, nm in cond_names.items():
        print(f"\n  [{cond}] {nm}")
        for r in rows:
            t = f"c{cond}_73"
            if f"return_{t}" in r:
                print(f"    {r['agent']:<38s} ret {r[f'return_{t}']:7.2f} +- {r[f'sem_{t}']:4.2f}  "
                      f"cas {r[f'casualties_{t}']:6.1f}  FA {r[f'false_alarms_{t}']:4.2f}  "
                      f"warned {r[f'warned_in_time_pct_{t}']:5.1f}%  "
                      f"lead {r[f'median_lead_days_{t}']:4.1f} d")
    for s_ in sens:
        print(f"  c{s_['condition']} kappa={s_['kappa']:.2f} q={s_['quantile']:.2f}  "
              f"rank {s_['conclave_rank_among_ml']} (cas rank {s_['conclave_casualty_rank_among_ml']})"
              f"  vs {s_['best_other']:<20s} {s_['delta_vs_best_other']:+6.2f} "
              f"[{s_['ci_low']:+.2f}, {s_['ci_high']:+.2f}]  floods {s_['flood_episode_pct']:.0f}%")
    save("exp9_external_validation", out)
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    budget = float(os.environ.get("EXP9_BUDGET_S", "inf"))
    if mode in ("train", "all"):
        end = time.perf_counter() + budget
        load_agents(train_missing=True, budget_end=end)
        load_agents(train_missing=True, budget_end=end, gauges=True)
    if mode in ("main", "all"):
        run_units(main_keys(), budget)
    if mode in ("train_real", "all"):
        end = time.perf_counter() + budget
        load_real_agents(gauges=False, train_missing=True, budget_end=end)
        load_real_agents(gauges=True, train_missing=True, budget_end=end)
        load_real_agents(gauges=False, train_missing=True, budget_end=end, mix=True)
        load_real_agents(gauges=True, train_missing=True, budget_end=end, mix=True)
    if mode in ("temporal", "all"):
        run_units(temporal_keys(), budget)
    if mode in ("accuracy", "all"):
        part_accuracy()
    if mode in ("sens", "all"):
        run_units(sens_keys(), budget)
    if mode in ("cases", "all"):
        part_cases()
    if mode in ("summary", "all"):
        summarise()


if __name__ == "__main__":
    main()
