#!/usr/bin/env python3
"""Download every dataset the CONCLAVE experiments need, into ./data.

Cross-platform replacement for fetch_data.sh -- no curl, no bash, so it runs
in the PyCharm terminal on Windows.

    python scripts/fetch_data.py            # required datasets
    python scripts/fetch_data.py --all      # plus the optional ones
    python scripts/fetch_data.py --check    # report what is present, download nothing

Datasets are used under their own licences; cite the original papers
(BFCL: Patil et al.; HumAID: Alam et al., ICWSM 2021).
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

BFCL = ("https://huggingface.co/datasets/gorilla-llm/"
        "Berkeley-Function-Calling-Leaderboard/resolve/main")
HUMAID = ("https://huggingface.co/api/datasets/QCRI/HumAID-all/"
          "parquet/default/{split}/0.parquet")

BFCL_FILES = [
    "BFCL_v3_simple", "BFCL_v3_multiple", "BFCL_v3_live_simple",
    "BFCL_v3_live_multiple", "BFCL_v3_irrelevance", "BFCL_v3_live_irrelevance",
    "BFCL_v3_parallel", "BFCL_v3_parallel_multiple",
]
BFCL_ANSWERS = ["BFCL_v3_simple", "BFCL_v3_multiple",
                "BFCL_v3_live_simple", "BFCL_v3_live_multiple"]

OPTIONAL = {
    "ai4i2020.csv":
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
    "nsl_kdd_train.txt":
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/"
        "KDDTrain%2B_20Percent.txt",
}


def targets(include_optional: bool) -> dict[Path, str]:
    jobs: dict[Path, str] = {}
    for f in BFCL_FILES:
        jobs[DATA / "bfcl" / f"{f}.json"] = f"{BFCL}/{f}.json"
    for f in BFCL_ANSWERS:
        jobs[DATA / "bfcl" / f"ans_{f}.json"] = f"{BFCL}/possible_answer/{f}.json"
    for split in ("train", "validation", "test"):
        jobs[DATA / "humaid" / f"{split}.parquet"] = HUMAID.format(split=split)
    if include_optional:
        for name, url in OPTIONAL.items():
            jobs[DATA / name] = url
    return jobs


def download(url: str, dest: Path) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "conclave-fetch"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 16):
                fh.write(chunk)
        tmp.replace(dest)
        return True, f"{dest.stat().st_size / 1e6:.1f} MB"
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="also fetch the optional AI4I and NSL-KDD sets")
    ap.add_argument("--check", action="store_true",
                    help="report what is present; download nothing")
    ap.add_argument("--force", action="store_true", help="re-download everything")
    args = ap.parse_args()

    jobs = targets(args.all)
    have = [d for d in jobs if d.exists() and d.stat().st_size > 0]

    if args.check:
        print(f"{len(have)}/{len(jobs)} files present under {DATA}")
        for d in jobs:
            mark = "ok  " if d in have else "MISS"
            size = f"{d.stat().st_size / 1e6:6.1f} MB" if d in have else "        -"
            print(f"  [{mark}] {size}  {d.relative_to(ROOT)}")
        return 0 if len(have) == len(jobs) else 1

    todo = list(jobs) if args.force else [d for d in jobs if d not in have]
    if not todo:
        print(f"all {len(jobs)} files already present under {DATA} -- nothing to do")
        return 0

    print(f"fetching {len(todo)} file(s) into {DATA}\n")
    failed = []
    for i, dest in enumerate(todo, 1):
        ok, note = download(jobs[dest], dest)
        print(f"  [{i:2d}/{len(todo)}] {'ok ' if ok else 'FAIL'}  "
              f"{dest.relative_to(ROOT)}  ({note})", flush=True)
        if not ok:
            failed.append(dest)

    if failed:
        print(f"\n{len(failed)} download(s) failed. The optional UCI/NSL-KDD sets "
              f"are not needed for any experiment in the paper.")
        required = [f for f in failed if f.name not in OPTIONAL]
        return 1 if required else 0
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
