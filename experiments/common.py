"""Shared plumbing for CONCLAVE experiments: paths, seeding, result IO."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def save(name: str, payload: Dict[str, Any]) -> Path:
    """Persist a result blob with full provenance."""
    import psutil
    payload = dict(payload)
    payload["_meta"] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_rev(),
        "seed": SEED,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
    }
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  -> wrote {p.relative_to(ROOT)}")
    return p


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74, flush=True)


def row(*cells, widths=(30, 12, 12, 12, 12)) -> str:
    out = []
    for c, w in zip(cells, widths):
        s = str(c)
        out.append(s.ljust(w) if len(out) == 0 else s.rjust(w))
    return "".join(out)
