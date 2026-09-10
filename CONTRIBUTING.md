# Contributing to CONCLAVE

Thanks for looking. This is research code that is meant to be reused, so
contributions that make it easier to deploy on real hardware are especially
welcome.

## Setup

```bash
git clone https://github.com/Aranya69/conclave && cd conclave
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[boost,experiments,dev]"
pytest -q          # expect 14 passed
```

Python 3.10–3.12. On 3.13 `lightgbm` wheels are sometimes missing; the code
falls back to scikit-learn's `HistGradientBoostingClassifier`.

## The one rule that surprises people

`tests/test_paper_claims.py` checks every number quoted in the paper against
`results/*.json`. **It fails on purpose when a result moves.**

If you re-run an experiment and it goes red, that is the test doing its job.
Look at the diff and decide which is wrong — the code, the paper, or your
expectation — *before* touching a threshold. Silently widening a tolerance to
get a green tick defeats the entire mechanism.

If a number legitimately changed, update the paper and the test in the same
commit, and say why in the message.

## What makes a good contribution

**Especially wanted**

- **Real hardware measurements.** Nobody has run this on an actual Raspberry Pi.
  If you have one, `python deploy/pi_check.py` produces exactly the numbers the
  paper is currently missing. Open an issue with the output and your board
  revision, OS bitness and cooling.
- **Argument extraction.** The router selects a tool but does not fill typed
  arguments from free text. This is the largest functional gap.
- **A value-of-information gate.** Instantaneous uncertainty is a poor allocator
  of oversight in sequential tasks (see the negative result in the README). A
  criterion that weights uncertainty by the consequence of error at that step is
  the most promising open direction.
- **New domains.** The framework is domain-general; the evaluation is
  disaster-heavy. Agriculture, remote clinics and industrial monitoring all fit.

**Please avoid**

- Adding a language model. There are excellent frameworks for that; this one is
  a controlled experiment in not having one.
- Committing datasets. They have their own licences — extend
  `scripts/fetch_data.py` instead.
- Committing build artefacts (`*.pdf` outputs, `__pycache__`, `.venv`).

## Style

- Follow the surrounding code. Docstrings explain *why*, not *what* — the
  existing modules are the reference.
- Keep `n_jobs=1` defaults in inference paths. That is deliberate and measured:
  thread-pool setup dominates single-decision latency by ~20×.
- New public behaviour needs a test in `tests/test_smoke.py`.
- If you change anything that feeds a paper number, re-run the experiment and
  update `results/`, the paper and the test together.

## Reporting bugs

Include your OS, Python version, whether `lightgbm` installed, and the output of:

```bash
python -c "import conclave, sklearn, numpy; print(conclave.__version__, sklearn.__version__, numpy.__version__)"
```

For anything deployment-related, `python deploy/pi_check.py` output is the most
useful thing you can attach.

## License

By contributing you agree your work is licensed under Apache-2.0.
