"""Replay of observed rainfall through the flood-response task.

The agents are trained only on the synthetic simulator (``disaster.py``).
For external validation they are run, unchanged, on episodes whose rain and
river level come from *observed* daily rainfall (NASA POWER, MERRA-2) at
flood-prone districts of Bangladesh.  Two conversions connect the record
to the task, and both are fitted on a calibration period that is disjoint
from the test period:

1. Rain gauge -> agent sensor scale.  Empirical quantile mapping: a day's
   rainfall is replaced by the synthetic rainfall value at the same
   quantile, so the agents see real temporal structure on the sensor
   scale they were trained on.

2. Rain -> river stage.  An antecedent precipitation index (API), the
   classical wetness index of hydrology,
       API_t = kappa * API_{t-1} + P_t ,
   is mapped linearly to stage, h_t = h0 + beta * API_t, with beta chosen
   so that the flood threshold of the task (4.5 m) is reached at the
   ``q``-quantile of API in the calibration period.  Opening the sluice
   lowers the stage by the same extra drainage as in the simulator.

Everything else -- reward, evacuation, rescue, casualties, credibility,
outages, sensor noise -- is the simulator's own code, so a policy's return
on real forcing is directly comparable with its return on synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import math

import numpy as np

from conclave.envs.disaster import DisasterConfig, FloodResponseEnv

FLOOD_LEVEL = 4.5          # stage at which people become stranded (simulator)
SLUICE_EXTRA = 0.23        # extra drainage per step with the sluice open


def antecedent_index(rain_mm: np.ndarray, kappa: float) -> np.ndarray:
    """API_t = kappa * API_{t-1} + P_t (mm)."""
    out = np.empty(len(rain_mm), dtype=float)
    a = 0.0
    for i, p in enumerate(np.asarray(rain_mm, dtype=float)):
        a = kappa * a + p
        out[i] = a
    return out


@dataclass
class ForcingModel:
    """Calibrated conversions from observed rainfall to the task."""

    kappa: float
    quantile: float
    api_threshold: float          # API (mm) at which stage = FLOOD_LEVEL
    h0: float
    real_q: np.ndarray            # quantiles of observed rainfall (mm)
    syn_q: np.ndarray             # same quantiles of synthetic rainfall

    @property
    def beta(self) -> float:
        return (FLOOD_LEVEL - self.h0) / self.api_threshold

    @classmethod
    def fit(cls, calib_rain_mm: Dict[str, np.ndarray],
            calib_mask: Dict[str, np.ndarray], synthetic_rain: np.ndarray,
            kappa: float = 0.85, quantile: float = 0.95,
            h0: float = 1.0, n_q: int = 201) -> "ForcingModel":
        """Fit on the calibration days only.

        ``calib_rain_mm[s]`` is the full daily record of station ``s`` (API
        needs the preceding days); ``calib_mask[s]`` selects the calibration
        days within it.
        """
        api = np.concatenate([antecedent_index(calib_rain_mm[s], kappa)[calib_mask[s]]
                              for s in calib_rain_mm])
        rain = np.concatenate([np.asarray(calib_rain_mm[s])[calib_mask[s]]
                               for s in calib_rain_mm])
        qs = np.linspace(0, 1, n_q)
        return cls(kappa=kappa, quantile=quantile,
                   api_threshold=float(np.quantile(api, quantile)), h0=h0,
                   real_q=np.quantile(rain, qs),
                   syn_q=np.quantile(np.asarray(synthetic_rain, dtype=float), qs))

    def sensor_rain(self, rain_mm: np.ndarray) -> np.ndarray:
        """Quantile-map observed rainfall onto the simulator's rain scale."""
        x = np.asarray(rain_mm, dtype=float)
        # mid-rank for ties (many dry days share the value 0)
        lo = np.searchsorted(self.real_q, x, side="left")
        hi = np.searchsorted(self.real_q, x, side="right")
        u = np.clip((lo + hi) / 2.0 / (len(self.real_q) - 1), 0.0, 1.0)
        return np.interp(u, np.linspace(0, 1, len(self.syn_q)), self.syn_q)

    def stage(self, rain_mm: np.ndarray) -> np.ndarray:
        """River stage (m) from the full daily record."""
        return np.clip(self.h0 + self.beta * antecedent_index(rain_mm, self.kappa),
                       0.4, 14.0)

    def as_dict(self) -> Dict[str, float]:
        return {"kappa": self.kappa, "quantile": self.quantile,
                "api_threshold_mm": self.api_threshold, "h0": self.h0,
                "beta_m_per_mm": self.beta}


class ReplayFloodEnv(FloodResponseEnv):
    """The flood-response task driven by an observed rainfall window.

    Parameters
    ----------
    sensor_rain, stage:
        Arrays covering the episode and a short look-ahead beyond it
        (length >= horizon + 1); index 0 is the first day of the episode.
    reports:
        If False the field-report channel is empty, so the agent must act on
        the gauges alone (the synthetic reports are written from the true
        risk, which would otherwise carry information the gauges do not).
    """

    def __init__(self, sensor_rain: Sequence[float], stage: Sequence[float],
                 config: Optional[DisasterConfig] = None,
                 reports: bool = True) -> None:
        self.sensor_rain_ = np.asarray(sensor_rain, dtype=float)
        self.stage_ = np.asarray(stage, dtype=float)
        self.reports = reports
        cfg = config or DisasterConfig()
        if len(self.stage_) < cfg.horizon + 1:
            raise ValueError("forcing shorter than the horizon")
        super().__init__(cfg)

    # ------------------------------------------------------------------
    def reset(self):
        c = self.cfg
        self.t = 0
        self.level = float(self.stage_[0])
        self.rain = float(self.sensor_rain_[0])
        self.surge_at = -1                  # no synthetic surge
        self.surge_mag = 0.0
        self.sluice_drop = 0.0
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
        if hasattr(self, "_prev_level"):
            del self._prev_level
        return self._observe()

    # --- forcing ---------------------------------------------------------
    def _next_rain(self) -> float:
        return float(self.sensor_rain_[min(self.t + 1, len(self.sensor_rain_) - 1)])

    def _next_level(self, drain: float) -> float:
        if self.sluice_open:
            self.sluice_drop += SLUICE_EXTRA
        h = self.stage_[min(self.t + 1, len(self.stage_) - 1)] - self.sluice_drop
        return float(np.clip(h, 0.4, 14.0))

    # --- ground truth ----------------------------------------------------
    def steps_to_flood(self) -> Optional[int]:
        """Steps until the stage first exceeds the flood level (0 = during
        the current step), or None if it does not within the record."""
        if self.level > FLOOD_LEVEL:
            return -1
        drop = self.sluice_drop
        for k in range(self.t + 1, len(self.stage_)):
            if self.sluice_open:
                drop += SLUICE_EXTRA
            if self.stage_[k] - drop > FLOOD_LEVEL:
                return k - self.t - 1
        return None

    def _flood_risk(self) -> float:
        """Same form as the simulator: proximity of the flood plus level."""
        dt = self.steps_to_flood()
        if dt is None:
            prox = 0.0
        elif dt < 0:
            prox = 1.0
        else:
            prox = math.exp(-dt / 4.0)
        lvl = 1 / (1 + math.exp(-(self.level - 4.0)))
        return float(np.clip(0.65 * prox + 0.35 * lvl, 0.0, 1.0))

    def _report(self) -> str:
        if not self.reports:
            return ""
        return super()._report()


def flood_onset(stage: np.ndarray, horizon: int) -> Optional[int]:
    """First episode day on which the (sluice-free) stage exceeds the flood
    level, or None."""
    idx = np.nonzero(np.asarray(stage[:horizon + 1]) > FLOOD_LEVEL)[0]
    return int(idx[0]) if len(idx) else None
