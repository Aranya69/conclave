# Deploying CONCLAVE to an IoT device

Everything here targets one question: **will this run on my board, and will it
fit?** You do not need the hardware to answer it.

## What is in here

| File | Purpose |
|---|---|
| `Dockerfile.pi4` | A genuine `linux/arm64` image — real ARM wheels, real ARM BLAS |
| `docker-compose.pi4.yml` | Three boards (Pi 4 / Pi 3B+ / Pi Zero 2 W) with enforced RAM and CPU limits |
| `pi_check.py` | Trains an agent, exercises it, and labels each result valid or not |
| `edge_agent_flood.npz` | A trained flood agent exported for the NumPy-only runtime (113 KB) |

## Without a Raspberry Pi

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx build --platform linux/arm64 -f deploy/Dockerfile.pi4 -t conclave:pi4 --load .
docker compose -f deploy/docker-compose.pi4.yml run --rm pi4 python deploy/pi_check.py
```

### What emulation proves, and what it does not

| Measurement | Under QEMU |
|---|---|
| Wheel availability for ARM | **Valid** — and only obtainable this way |
| Correctness / accuracy | **Valid** |
| Model artefact size | **Valid** |
| Peak RSS, fits-in-RAM | **Valid** — the cgroup limit is real |
| Decision latency | **Invalid** — QEMU is 5–20× slower, non-uniformly |
| Energy per decision | **Invalid** — derived from latency |

`pi_check.py` detects emulation and marks its own timing block invalid, so you
cannot quote a QEMU number by accident.

> `memswap_limit` is set equal to `mem_limit` in the compose file. Without
> that, Docker allows swap and a "512 MB" test quietly becomes a 1 GB test.

## On a real board

Use **64-bit Raspberry Pi OS**. On 32-bit, `pip install scipy` means compiling
scipy on the Pi — hours of build time that often dies of memory exhaustion.
None of numpy, scipy, scikit-learn, lightgbm or psutil publish `armv7l` wheels
on PyPI; all publish `aarch64`.

```bash
uname -m                       # want: aarch64
sudo apt update && sudo apt install -y python3-venv python3-dev libgomp1

python3 -m venv ~/conclave-env && source ~/conclave-env/bin/activate
git clone https://github.com/Aranya69/conclave && cd conclave
pip install --only-binary=:all: -e ".[boost]"

python deploy/pi_check.py      # now every line is valid
```

Before timing anything: `sudo cpufreq-set -g performance`, and watch
`vcgencmd measure_temp` — a Pi 4 throttles under sustained load.

## Shipping a trained agent

Field devices should receive the artefact, not the training pipeline.

```python
# on your workstation
import joblib
joblib.dump({"fz": fz, "brain": brain, "gate": brain.gate},
            "agent.joblib", compress=3)          # ~200 KB
```

```python
# on the device — no pandas, no matplotlib, no training data
import joblib
from conclave import Agent
from conclave.core.types import Observation

b = joblib.load("agent.joblib")
agent = Agent(brain=b["brain"], featurizer=b["fz"],
              registry=registry, gate=b["gate"])

while True:
    d = agent.decide(Observation(features=read_sensors(), text=read_radio()))
    log(d.action, d.confidence, d.disagreement, d.gate)
```

Runtime dependencies collapse to numpy, scipy, scikit-learn and joblib —
about 60 MB of wheels plus a 200 KB agent.

> **`joblib.load` unpickles, which executes code.** Only load artefacts you
> produced yourself or received over a channel you trust. This matters the
> moment you start pushing model updates to field devices.

See `docs/CONCLAVE_Guide.pdf` §5 for the full treatment.


## The NumPy-only runtime (smallest deployment)

Training needs scikit-learn; deciding does not. `conclave/edge/runtime.py` is a
single file that imports only NumPy and the standard library, and reproduces the
scikit-learn agent's decisions exactly (Experiment 8: 2,400 / 2,400 identical
actions, peak RAM 32 MB instead of 185 MB, 0.19 ms instead of 2.7 ms per
decision on the host).

Export on the training machine:

```python
from conclave.edge.runtime import export_agent
export_agent("my_agent.npz", featurizer, brain, gate)   # linear committee only
```

Copy two files to the board -- `conclave/edge/runtime.py` and `my_agent.npz` --
and install NumPy only (`pip install numpy`). Then:

```python
from runtime import EdgeAgent
agent = EdgeAgent("my_agent.npz")
d = agent.decide({"river_level": 4.8, "rainfall": 9.1}, "water rising near the school")
print(d["action"], d["gate"], d["stats"]["epistemic"])
```

`edge_agent_flood.npz` in this folder is a ready-made example.
