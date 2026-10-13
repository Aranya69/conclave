#!/usr/bin/env python3
"""Regenerate every result, figure, and the Word manuscript.

Cross-platform replacement for run_all.sh -- works in the VS Code integrated
terminal (PowerShell) on Windows, and in any shell on macOS and Linux, without
needing bash.

    python scripts/run_all.py              # everything
    python scripts/run_all.py --quick      # skip the slow HumAID sweep
    python scripts/run_all.py exp1 exp3    # just these
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name -> (script, args, rough minutes on a 2-core machine)
STEPS = {
    "exp1": ("experiments/exp1_tool_routing.py", [], 1),
    "exp2": ("experiments/exp2_humaid.py", ["12", "14", "16"], 8),
    "exp3": ("experiments/exp3_sequential.py", [], 4),
    "exp4": ("experiments/exp4_edge.py", [], 1),
    "exp5": ("experiments/exp5_uncertainty.py", [], 5),
    "exp6": ("experiments/exp6_ablation.py", ["--fresh"], 6),
    "exp7": ("experiments/exp7_ml_agents.py", [], 25),
    "exp8": ("experiments/exp8_edge_runtime.py", [], 3),
    "figures": ("experiments/make_figures.py", [], 1),
    "docx": ("scripts/make_docx.py", [], 1),
}
QUICK_OVERRIDES = {"exp2": ["12"]}


def run(name: str, script: str, args: list[str]) -> bool:
    path = ROOT / script
    if not path.exists():
        print(f"  [{name}] SKIP - {script} not found")
        return True
    print(f"\n{'=' * 70}\n  {name}: {script} {' '.join(args)}\n{'=' * 70}",
          flush=True)
    t0 = time.time()
    # sys.executable keeps every child process inside the same virtualenv that
    # is running this script, whichever interpreter the editor selected
    r = subprocess.run([sys.executable, str(path), *args], cwd=ROOT)
    mins = (time.time() - t0) / 60
    status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    print(f"  [{name}] {status} in {mins:.1f} min", flush=True)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", default=None,
                    help=f"subset of: {', '.join(STEPS)}")
    ap.add_argument("--quick", action="store_true",
                    help="smallest feature budget only; much faster")
    args = ap.parse_args()

    chosen = args.steps or list(STEPS)
    unknown = [s for s in chosen if s not in STEPS]
    if unknown:
        print(f"unknown step(s): {unknown}\nvalid: {list(STEPS)}")
        return 2

    if not (ROOT / "data" / "humaid" / "train.parquet").exists():
        print("data/ is missing or incomplete -- run:  python scripts/fetch_data.py")
        return 1

    total = sum(STEPS[s][2] for s in chosen)
    print(f"running {len(chosen)} step(s); roughly {total} min on a slow machine")

    failed = []
    for name in chosen:
        script, argv, _ = STEPS[name]
        if args.quick and name in QUICK_OVERRIDES:
            argv = QUICK_OVERRIDES[name]
        if not run(name, script, argv):
            failed.append(name)

    print("\n" + "=" * 70)
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        return 1
    print("  all steps completed -- see results/ and paper/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
