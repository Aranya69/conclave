"""A sequential flood-response environment.

Classification benchmarks cannot show whether a framework is *agentic*.
This environment can: the agent operates a river district over a horizon,
choosing when to warn, evacuate, deploy rescue teams, and when to hold
resources back.  Decisions compound -- evacuating early costs resources
but saves lives if the flood arrives; crying wolf erodes compliance so
later warnings are ignored.

The switch that matters for this work is ``connectivity``: a schedule of
network outages.  A cloud-backed LLM agent cannot act during an outage.
CONCLAVE is unaffected, which is precisely the hypothesis under test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from conclave.core.types import Action, ActionKind, Observation

ACTIONS: List[str] = [
    "monitor",            # cheap, keeps sensing
    "issue_alert",        # warn the population
    "evacuate_zone",      # move people out (expensive, effective)
    "deploy_rescue",      # save those already stranded
    "request_supplies",   # replenish resources
    "open_sluice",        # reduce level, irreversible, damages farmland
]


@dataclass
class DisasterConfig:
    horizon: int = 40
    seed: int = 0
    population: float = 1000.0
    n_boats: int = 6
    rain_intensity: float = 1.0
    sensor_noise: float = 0.06
    outage_prob: float = 0.0        # per-step probability comms are down
    outage_len: int = 6
    false_alarm_penalty: float = 4.0
    evac_cost: float = 2.0
    sluice_farm_cost: float = 25.0


class FloodResponseEnv:
    """Discrete-time flood response with partial observability.

    Observation combines noisy sensor values with a short text field
    report, so the same agent exercises both the tabular and text paths
    of the featuriser -- which is the realistic offline situation
    (gauges plus radio traffic).
    """

    def __init__(self, config: Optional[DisasterConfig] = None) -> None:
        self.cfg = config or DisasterConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        c = self.cfg
        self.t = 0
        self.level = float(self.rng.uniform(1.0, 2.0))     # metres
        self.rain = float(self.rng.uniform(0.0, 4.0))
        self.surge_at = int(self.rng.integers(8, max(9, c.horizon - 8)))
        self.surge_mag = float(self.rng.uniform(2.5, 6.0)) * c.rain_intensity
        self.warned = False
        self.evacuated = 0.0
        self.stranded = 0.0
        self.casualties = 0.0
        self.boats = c.n_boats
        self.supplies = 5.0
        self.credibility = 1.0
        self.sluice_open = False
        self.farm_damage = 0.0
        self.false_alarms = 0
        self.offline_left = 0
        self.offline_steps = 0
        self._done = False
        return self._observe()

    # ------------------------------------------------------------------
    @property
    def online(self) -> bool:
        return self.offline_left <= 0

    def _flood_risk(self) -> float:
        """Ground-truth probability the district floods within 3 steps."""
        dt = self.surge_at - self.t
        prox = math.exp(-max(dt, 0) / 4.0) if dt >= 0 else 1.0
        lvl = 1 / (1 + math.exp(-(self.level - 4.0)))
        return float(np.clip(0.65 * prox + 0.35 * lvl, 0.0, 1.0))

    def _report(self) -> str:
        """Synthetic field report; vocabulary correlates with true state."""
        r = self._flood_risk()
        rng = self.rng
        if r > 0.75:
            base = rng.choice([
                "water rising fast over the embankment houses flooding now",
                "urgent people trapped on rooftops need boats immediately",
                "bridge submerged road impassable families stranded",
            ])
        elif r > 0.45:
            base = rng.choice([
                "river level climbing steadily above normal for this month",
                "heavy rain continuing upstream villages watching closely",
                "embankment seepage reported near the school",
            ])
        elif r > 0.2:
            base = rng.choice([
                "light rain river slightly up no damage reported",
                "conditions stable market open roads clear",
            ])
        else:
            base = rng.choice([
                "clear weather river low normal activity",
                "no incidents to report all quiet",
            ])
        if self.stranded > 0:
            base += f" about {int(self.stranded)} people awaiting rescue"
        return str(base)

    def _observe(self) -> Observation:
        n = self.cfg.sensor_noise
        j = lambda v: float(v * (1 + self.rng.normal(0, n)))
        feats = {
            "river_level": j(self.level),
            "rainfall": j(self.rain),
            "level_rate": j(self.level - getattr(self, "_prev_level", self.level)),
            "hours_elapsed": float(self.t),
            "boats_available": float(self.boats),
            "supplies": float(self.supplies),
            "evacuated_frac": self.evacuated / self.cfg.population,
            "stranded": float(self.stranded),
            "credibility": float(self.credibility),
            "warned": float(self.warned),
            "sluice_open": float(self.sluice_open),
            "comms_online": float(self.online),
        }
        self._prev_level = self.level
        return Observation(features=feats, text=self._report(),
                           meta={"t": self.t, "online": self.online,
                                 "true_risk": self._flood_risk()})

    # ------------------------------------------------------------------
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        c = self.cfg
        name = action.tool if action.kind is ActionKind.TOOL else action.kind.value
        reward = 0.0
        risk = self._flood_risk()

        # --- comms schedule -------------------------------------------------
        if self.offline_left > 0:
            self.offline_left -= 1
            self.offline_steps += 1
        elif self.rng.random() < c.outage_prob:
            self.offline_left = c.outage_len

        # --- act ------------------------------------------------------------
        if name == "issue_alert":
            if risk > 0.45:
                self.warned = True
                self.credibility = min(1.0, self.credibility + 0.05)
                reward += 3.0
            else:
                self.false_alarms += 1
                self.credibility = max(0.2, self.credibility - 0.15)
                reward -= c.false_alarm_penalty
        elif name == "evacuate_zone":
            take = min(220.0 * self.credibility, c.population - self.evacuated)
            self.evacuated += take
            reward += 0.02 * take * risk - c.evac_cost
        elif name == "deploy_rescue":
            if self.boats > 0 and self.stranded > 0:
                saved = min(self.stranded, 45.0)
                self.stranded -= saved
                self.boats -= 1
                reward += 0.12 * saved
            else:
                reward -= 1.0
        elif name == "request_supplies":
            self.supplies += 3.0
            self.boats = min(c.n_boats, self.boats + 1)
            reward -= 0.5
        elif name == "open_sluice":
            if not self.sluice_open:
                self.sluice_open = True
                self.farm_damage = c.sluice_farm_cost
                reward -= c.sluice_farm_cost * 0.2
                if risk > 0.6:
                    reward += 12.0
            else:
                reward -= 0.5
        elif name == "monitor":
            reward -= 0.05
        elif action.kind is ActionKind.ESCALATE:
            reward -= 0.4          # a human's time is not free, but it is cheap
        elif action.kind in (ActionKind.GATHER, ActionKind.ABSTAIN):
            reward -= 0.15

        # --- world dynamics --------------------------------------------------
        self.rain = float(np.clip(
            self.rain + self.rng.normal(0, 1.2)
            + (3.0 if abs(self.t - self.surge_at) < 3 else 0.0), 0, 20))
        drain = 0.45 if self.sluice_open else 0.22
        self.level = float(np.clip(
            self.level + 0.11 * self.rain - drain
            + (self.surge_mag if self.t == self.surge_at else 0.0), 0.4, 14.0))

        if self.level > 4.5:
            exposed = max(c.population - self.evacuated, 0.0)
            newly = exposed * 0.05 * min((self.level - 4.5) / 3.0, 1.0)
            self.stranded += newly
            unhelped = self.stranded * 0.03
            self.casualties += unhelped
            self.stranded = max(self.stranded - unhelped, 0.0)
            reward -= 0.35 * unhelped

        self.t += 1
        self._done = self.t >= c.horizon
        if self._done:
            reward += 0.01 * self.evacuated * 0.5
            reward -= 0.2 * self.casualties
            reward -= 0.05 * self.farm_damage

        info = {"true_risk": risk, "online": self.online,
                "casualties": self.casualties, "level": self.level,
                "stranded": self.stranded, "false_alarms": self.false_alarms}
        return self._observe(), reward, self._done, info

    # ------------------------------------------------------------------
    def succeeded(self) -> bool:
        return self.casualties < 12.0 and self.false_alarms <= 2

    def summary(self) -> Dict[str, float]:
        return {
            "casualties": round(self.casualties, 2),
            "evacuated": round(self.evacuated, 1),
            "false_alarms": float(self.false_alarms),
            "farm_damage": round(self.farm_damage, 1),
            "offline_steps": float(self.offline_steps),
            "peak_level": round(self.level, 2),
        }


def expert_policy(obs: Observation, rng: np.random.Generator) -> str:
    """A competent scripted controller used to generate training episodes.

    It sees the *true* risk (which the agent never does) and acts on a
    hand-written schedule.  The ensemble is trained by behaviour cloning
    on its decisions from noisy observations, so the agent must recover
    the policy from partial information -- the realistic setting.
    """
    r = float(obs.meta.get("true_risk", 0.0))
    f = obs.features
    if f.get("stranded", 0) > 5 and f.get("boats_available", 0) > 0:
        return "deploy_rescue"
    if r > 0.8 and f.get("river_level", 0) > 5.0 and not f.get("sluice_open"):
        return "open_sluice"
    if r > 0.55 and not f.get("warned"):
        return "issue_alert"
    if r > 0.6 and f.get("evacuated_frac", 0) < 0.7:
        return "evacuate_zone"
    if f.get("boats_available", 0) < 2 or f.get("supplies", 0) < 2:
        return "request_supplies"
    return "monitor"
