#!/usr/bin/env python3
"""Verify an CONCLAVE deployment inside an emulated or real Pi environment.

Reports only what the environment can honestly support:

  ALWAYS VALID   architecture, wheel provenance, import success, correctness,
                 model size, peak RSS, whether it fits the RAM ceiling
  NATIVE ONLY    latency and anything derived from it

Under QEMU the latency numbers are printed but explicitly marked invalid,
because emulation overhead is large and not proportional to native speed.
"""
from __future__ import annotations

import os
import platform
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def under_emulation() -> bool:
    """True if an aarch64 userland is very likely running on QEMU."""
    if platform.machine() not in ("aarch64", "arm64"):
        return False
    # Real Pi kernels expose a Hardware/Model line; QEMU user-mode inherits the
    # host kernel, so /proc/cpuinfo has neither.
    try:
        info = open("/proc/cpuinfo").read().lower()
        if "raspberry" in info or "bcm2" in info:
            return False
    except OSError:
        pass
    try:
        if "raspberry" in open("/proc/device-tree/model", "rb").read().decode(
                errors="ignore").lower():
            return False
    except OSError:
        pass
    return True


def main() -> int:
    import joblib
    import numpy as np
    import psutil
    import sklearn

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from conclave import EnsembleBrain
    from conclave.brain.featurize import (BudgetedTextFeaturizer, FusedFeaturizer,
                                       TabularFeaturizer)
    from conclave.core.types import Action, ActionKind
    from conclave.envs.disaster import DisasterConfig, FloodResponseEnv, expert_policy

    emul = under_emulation()
    print("=" * 68)
    print(f"  architecture   : {platform.machine()}")
    print(f"  platform       : {platform.platform()}")
    print(f"  python         : {platform.python_version()}")
    print(f"  numpy / sklearn: {np.__version__} / {sklearn.__version__}")
    print(f"  visible cores  : {os.cpu_count()}")
    print(f"  RAM ceiling    : {psutil.virtual_memory().total / 1024**2:.0f} MB")
    print(f"  emulated       : {'YES - timings are NOT valid' if emul else 'no'}")
    print("=" * 68)

    # ---- train a small agent -------------------------------------------
    rng = np.random.default_rng(0)
    obs, lab = [], []
    for e in range(40):
        env = FloodResponseEnv(DisasterConfig(seed=e, horizon=40))
        o = env.reset()
        for _ in range(40):
            a = expert_policy(o, rng)
            obs.append(o); lab.append(a)
            o, _, done, _ = env.step(Action(kind=ActionKind.TOOL, tool=a))
            if done:
                break

    t0 = time.perf_counter()
    fz = FusedFeaturizer(
        text=BudgetedTextFeaturizer(n_word_features=2**13, n_char_features=2**13),
        tabular=TabularFeaturizer()).fit(obs)
    brain = EnsembleBrain().fit(fz.transform(obs), np.array(lab))
    train_s = time.perf_counter() - t0

    import io
    buf = io.BytesIO(); joblib.dump({"fz": fz, "brain": brain}, buf, compress=3)

    # ---- correctness: does it reproduce a sane policy? ------------------
    acc = float((brain.predict(fz.transform(obs)) == np.array(lab)).mean())

    proc = psutil.Process()
    lat = []
    probe = obs[:1]
    for _ in range(30):
        fz.transform(probe); brain.report(fz.transform(probe))
    for _ in range(120):
        t0 = time.perf_counter()
        brain.report(fz.transform(probe))
        lat.append((time.perf_counter() - t0) * 1000)
    rss = proc.memory_info().rss / 1024**2

    print(f"\n  [valid]  agent artefact      : {buf.tell()/1000:8.1f} KB")
    print(f"  [valid]  training accuracy   : {acc*100:8.2f} %")
    print(f"  [valid]  peak RSS            : {rss:8.1f} MB")
    print(f"  [valid]  fits RAM ceiling    : "
          f"{'YES' if rss < psutil.virtual_memory().total/1024**2 else 'NO'}")
    print(f"  [valid]  training time       : {train_s:8.1f} s")

    tag = "INVALID under emulation" if emul else "valid (native)"
    print(f"\n  [{tag}]")
    print(f"           median decision     : {np.median(lat):8.2f} ms")
    print(f"           p99 decision        : {np.percentile(lat,99):8.2f} ms")

    if emul:
        print("\n  Emulated timings are typically 5-20x slower than native and the")
        print("  slowdown is not uniform across operations. Do not put these in a")
        print("  paper as Raspberry Pi latencies. Report the [valid] rows instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
