# Running CONCLAVE on Windows + VS Code

Written for someone comfortable with Python and Flask. If you can set up a venv
and run pytest, there is nothing new here — the only unusual part is that some
experiments take a few minutes of CPU.

---

## 1. One-time setup

Open the `conclave` folder in VS Code (**File → Open Folder**, pick the folder that
contains `pyproject.toml` — not its parent). Then in the integrated terminal
(`` Ctrl+` ``):

```powershell
# Python 3.10-3.12. 3.13 sometimes lacks wheels for lightgbm.
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[boost,experiments,dev]"
```

If PowerShell refuses to run the activate script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

That lasts only for the current terminal, which is what you want.

Then press `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one under
`.venv`. The `.vscode/settings.json` in the repo already points at it, so this
usually happens on its own.

### What gets installed

Nothing exotic — `numpy`, `scipy`, `scikit-learn`, `joblib`, `psutil`, plus
`lightgbm` (the `boost` extra) and `pandas`/`pyarrow`/`matplotlib` (the
`experiments` extra). No GPU, no CUDA, no API keys, no network at runtime.

---

## 2. Check it works

```powershell
pytest -q
```

Expect **19 passed** in about twenty seconds. Three groups:

- `tests/test_smoke.py` — the framework itself: uncertainty maths, the planner,
  tool registry, memory, a full agent episode.
- `tests/test_paper_claims.py` — every number quoted in the paper, checked
  against `results/*.json`. If you re-run an experiment and a number moves,
  these fail on purpose.
- `tests/test_edge_runtime.py` — the NumPy-only edge runtime reproduces the
  scikit-learn agent exactly.

VS Code's Testing panel (the flask icon in the sidebar — yes, really) picks
these up automatically and lets you run them individually.

---

## 3. Poke at it by hand

This is the part worth your time. `demo.py` has four sections:

```powershell
python demo.py              # menu
python demo.py episode      # watch the agent decide, tick by tick
python demo.py route        # type a request, see which tool it picks
python demo.py uncertainty  # break the sensors, watch confidence collapse
python demo.py speed        # single-decision latency on your CPU
```

Each trains from scratch in about four seconds, so there is nothing to
pre-build.

### `episode` — the core claim, visible

Prints one row per tick: river level, true risk, whether the network is up, the
chosen action, confidence, ensemble disagreement, which gate fired, and latency.

```
    t  level   rain  risk  net | action            conf disagr      gate    ms
   12   2.76   0.69  0.46 DOWN | monitor           1.00   0.00       act   5.0
   13   2.62   3.00  0.58 DOWN | issue_alert       0.50   0.28       act   4.0
   14   3.39   6.38  0.76   up | evacuate_zone     0.98   0.01       act   6.0
   15   8.94   9.26  1.00   up | escalate          0.67   0.35  escalate   4.8
```

Watch the `net` column. It says `DOWN` for long stretches and nothing changes —
that is the whole thesis in one column. At t=15 the surge hits, the committee
splits (disagreement 0.35), and the gate escalates instead of guessing.

### `route` — tool selection with no LLM

Registers five tools that were **not** in the training data, then lets you type
requests. It shows the score for every candidate and how long it took.

Try `there are people stuck on a roof in sector 4, send help` — it picks
`dispatch_rescue_boat` in about 4 ms with a clear margin.

Then try `what is 200 dollars in taka`. It gets it **wrong**, and tells you so:
the scores come out nearly flat (0.29 / 0.23 / 0.18 / 0.18 / 0.12) and the demo
flags it as a case where the abstention head should decline instead of guessing.
Lexical features don't know that "taka" is a currency. That limitation is in the
paper; the demo shows it rather than hiding it.

### `uncertainty` — where the epistemic signal comes from

Drops an increasing fraction of the sensors and prints confidence, aleatoric,
and epistemic uncertainty. Then shows the individual member votes so you can see
*why* the ensemble is unsure — typically the linear model disagreeing with the
two others.

### `speed` — your hardware

Measures mean/p95/p99 latency for one decision, reports the agent's serialised
size, and projects onto Raspberry Pi profiles. Those projections are scaled
estimates, not measurements — the paper says so too.

---

## 4. Reproduce the paper's results

```powershell
python experiments/exp3_sequential.py    # ~3 min, no data needed
python experiments/exp5_uncertainty.py   # ~4 min, no data needed
python experiments/exp4_edge.py          # ~1 min, no data needed
python experiments/exp1_tool_routing.py  # ~2 min, needs data/bfcl
python experiments/exp2_humaid.py 12 14  # ~4 min, needs data/humaid
python experiments/exp6_ablation.py --fresh   # ~6 min, single models vs committees
python experiments/exp7_ml_agents.py     # ~25 min, CONCLAVE vs 8 other ML agents
python experiments/exp8_edge_runtime.py  # ~3 min, NumPy-only runtime vs scikit-learn
python experiments/make_figures.py       # regenerates paper/figures/
```

Or everything at once:

```powershell
python scripts/run_all.py
```

`data/` is **not** committed to the repository. The datasets belong to their
original authors and carry their own licences, so the repo ships a fetcher
instead:

```powershell
python scripts/fetch_data.py     # ~19 MB, one time
```

`exp3`, `exp4` and `exp5` need no data at all and work immediately after a
clone. Only `exp1` (BFCL) and `exp2` (HumAID) need the download.

Each experiment writes a JSON artefact to `results/` recording the platform,
seed, and git commit, so results are traceable to the run that produced them.

**One Windows difference.** `exp4_edge.py` enforces a RAM ceiling with
`RLIMIT_AS`, which is POSIX-only. On Windows it detects this, prints a notice,
and reports thread caps only — the memory limit becomes advisory. The code
handles it; just don't claim an enforced ceiling in a paper table from a Windows
run. Everything else is identical across platforms.

---

## 5. Reading the code

Start here, in this order:

| File | What it does | Lines |
|---|---|---|
| `conclave/core/types.py` | `Observation`, `Action`, `Decision`, `Episode` | 151 |
| `conclave/brain/uncertainty.py` | the aleatoric/epistemic split and the gate | ~250 |
| `conclave/brain/ensemble.py` | the committee that replaces the LLM | ~230 |
| `conclave/core/agent.py` | the sense→decide→act loop | 174 |
| `conclave/brain/router.py` | tool selection as pairwise ranking | ~200 |
| `conclave/envs/disaster.py` | the flood simulator | 271 |

The single most useful breakpoint is `Agent.decide` in `conclave/core/agent.py`.
Use the **"Debug: break inside Agent.decide"** launch config (it sets
`justMyCode: false` so you can step into scikit-learn if you want), put a
breakpoint on the `outcome = self.gate(rep.stats)` line, and inspect `rep` —
you get the per-member posteriors and the full uncertainty decomposition for
that one decision.

### The mental model, in Flask terms

The tool registry is a URL map: `ToolRegistry` holds `Tool` objects the way an
app holds view functions, and `@tool` is the decorator that registers one, with
the parameter schema inferred from the signature like a typed route converter.
The difference is what does the dispatch. Flask matches a path against the map
deterministically; CONCLAVE's `ToolRouter` *ranks* every registered tool against
the request and can decline to dispatch at all when nothing scores well enough.
The gate in `Agent.decide` is closest to a `before_request` hook that can
short-circuit — except its decision comes from how much the ensemble members
disagree, not from a rule you wrote.

---

## 6. Using it in your own code

```python
from conclave import Agent, EnsembleBrain, ToolRegistry, UncertaintyGate
from conclave.brain.featurize import (BudgetedTextFeaturizer, FusedFeaturizer,
                                   TabularFeaturizer)
from conclave.core.types import Observation
from conclave.tools.base import tool

@tool(description="Broadcast a flood warning to a zone", irreversible=True)
def issue_alert(zone: str, severity: int = 1) -> str:
    return f"alerted {zone} at level {severity}"

registry = ToolRegistry([issue_alert])

# observations: numeric sensors + optional free text
observations = [Observation(features={"level": 3.2, "rain": 8.0},
                            text="water rising near the school")]
labels = ["issue_alert"]

fz = FusedFeaturizer(
    text=BudgetedTextFeaturizer(n_word_features=2**13, n_char_features=2**13),
    tabular=TabularFeaturizer(),
).fit(observations)

brain = EnsembleBrain().fit(fz.transform(observations), labels)
agent = Agent(brain=brain, featurizer=fz, registry=registry,
              gate=UncertaintyGate())

decision = agent.decide(observations[0])
print(decision.action, decision.confidence, decision.disagreement, decision.gate)
```

Two things to know when you train on your own data:

- **`EnsembleBrain` needs at least two classes** and enough examples per class
  for its internal cross-validation. A handful of rows will raise.
- **Feature budget is the deployment knob.** `n_word_features` / `n_char_features`
  set the model size directly, because the hashing trick means there is no stored
  vocabulary. `2**12` gives a 2.6 MB artefact, `2**16` gives 36 MB. See Table III
  in the paper for what that costs in accuracy.

---

## 7. If something breaks

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: conclave` | venv not active, or you skipped `pip install -e .` |
| `lightgbm` fails to install | Python 3.13; use 3.12. The code falls back to sklearn's `HistGradientBoosting` if lightgbm is genuinely absent |
| `pytest` collects 0 tests | VS Code opened the parent folder — reopen the `conclave` folder itself |
| Experiments feel slow | Expected. They train several models per configuration; `exp3` runs 3,600 episodes |
| `test_paper_claims` fails | You re-ran an experiment and a number moved. That is the test doing its job — check the diff before updating the paper |
