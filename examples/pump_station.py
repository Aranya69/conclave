"""An off-grid water-pump station agent, end to end in one file.

A solar-powered borehole pump serving a village. It has a tank level sensor, a
turbidity probe, a battery monitor and a flow meter, plus a text field where a
caretaker types what they see. There is no reliable network.

Run it:  python examples/pump_station.py

This is the smallest complete CONCLAVE agent: five tools, four sensors, a
committee, an uncertainty gate and one hard safety constraint. Read it top to
bottom -- every section is numbered.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from conclave import (Agent, EnsembleBrain, EpisodicMemory, Observation,
                   ToolRegistry, UncertaintyGate)
from conclave.brain.featurize import (BudgetedTextFeaturizer, FusedFeaturizer,
                                   TabularFeaturizer)
from conclave.planning.rules import ConstraintSet, require_confidence
from conclave.tools.base import tool

# ---------------------------------------------------------------- 1. tools
@tool(description="Start the borehole pump to refill the tank")
def start_pump(minutes: int = 30) -> str:
    return f"pump running {minutes} min"

@tool(description="Stop the pump immediately")
def stop_pump() -> str:
    return "pump stopped"

@tool(description="Open the chlorine doser to treat the tank", irreversible=True)
def dose_chlorine(ml: int = 50) -> str:
    return f"dosed {ml} ml"

@tool(description="Send an SMS to the maintenance technician")
def call_technician(reason: str = "unknown") -> str:
    return f"SMS sent: {reason}"

@tool(description="Do nothing this cycle, keep monitoring")
def hold() -> str:
    return "holding"

registry = ToolRegistry([start_pump, stop_pump, dose_chlorine,
                         call_technician, hold])

# ------------------------------------------------- 2. synthetic training set
rng = np.random.default_rng(0)
obs, labels = [], []
def make(level, turbidity, volts, flow, text, action):
    obs.append(Observation(
        features={"tank_level_pct": level, "turbidity_ntu": turbidity,
                  "battery_volts": volts, "flow_lpm": flow},
        text=text))
    labels.append(action)

for _ in range(120):
    make(rng.uniform(5, 30), rng.uniform(0, 3), rng.uniform(12.4, 13.2),
         0.0, "tank low overnight draw down", "start_pump")
    make(rng.uniform(92, 100), rng.uniform(0, 3), rng.uniform(12.4, 13.2),
         rng.uniform(20, 40), "tank full pump still running", "stop_pump")
    make(rng.uniform(40, 80), rng.uniform(6, 15), rng.uniform(12.4, 13.2),
         rng.uniform(0, 30), "water cloudy after heavy rain", "dose_chlorine")
    make(rng.uniform(30, 70), rng.uniform(0, 3), rng.uniform(10.5, 11.6),
         0.0, "pump not starting battery flat no flow", "call_technician")
    make(rng.uniform(45, 85), rng.uniform(0, 2), rng.uniform(12.6, 13.2),
         rng.uniform(10, 25), "everything normal", "hold")

# ------------------------------------------------------------ 3. featurise
fz = FusedFeaturizer(
    text=BudgetedTextFeaturizer(n_word_features=2**12, n_char_features=2**12),
    tabular=TabularFeaturizer(),
).fit(obs)
X = fz.transform(obs)

# ---------------------------------------------------------------- 4. brain
brain = EnsembleBrain().fit(X, np.array(labels))
print("members:", list(brain.members_), " weights:", brain.weights_.round(3))
print("footprint:", brain.footprint_bytes()["_total"] / 1000, "KB")

# ------------------------------------------------- 5. gate + safety + agent
gate = UncertaintyGate(epistemic_max=0.30, aleatoric_max=0.75,
                       confidence_min=0.45, margin_min=0.05,
                       allow_gather=False)
constraints = ConstraintSet([require_confidence(0.80, ["dose_chlorine"])])
agent = Agent(brain=brain, featurizer=fz, registry=registry, gate=gate,
              constraints=constraints, memory=EpisodicMemory(capacity=500))

# --------------------------------------------------------------- 6. decide
probe = [
    ("tank nearly empty this morning",
     dict(tank_level_pct=11.0, turbidity_ntu=1.2, battery_volts=12.9, flow_lpm=0.0)),
    ("very cloudy water since the storm",
     dict(tank_level_pct=60.0, turbidity_ntu=11.0, battery_volts=12.8, flow_lpm=14.0)),
    ("no flow and the battery looks flat",
     dict(tank_level_pct=52.0, turbidity_ntu=1.0, battery_volts=11.0, flow_lpm=0.0)),
    ("strange smell from the wellhead nobody can explain",
     dict(tank_level_pct=55.0, turbidity_ntu=2.0, battery_volts=12.7, flow_lpm=12.0)),
]
print(f"\n{'observation':38s} {'action':16s} {'conf':>5s} {'disag':>6s} {'gate':>9s}")
for text, feats in probe:
    d = agent.decide(Observation(features=feats, text=text))
    print(f"{text[:37]:38s} {str(d.action):16s} {d.confidence:5.2f} "
          f"{d.disagreement:6.2f} {d.gate:>9s}")
