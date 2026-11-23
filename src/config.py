"""
config.py
All simulation parameters and seeds. Input values are read from data/inputs.json,
which is written by estimate_inputs.py (run that first).
"""
import json
import os

# ---- Seeds: replication r uses seed SEED_BASE + r for ALL policies (common random numbers)
SEED_BASE = 3410

# ---- Simulation scale
N_REPLICATIONS = 50
N_PATIENTS = 200
PEAK_MINUTES = 240

# ---- Policies
POLICIES = {
    'Baseline': {'n_doctors': 2, 'n_pharmacy': 1},
    'Policy_A': {'n_doctors': 3, 'n_pharmacy': 1},
    'Policy_B': {'n_doctors': 4, 'n_pharmacy': 1},
}

# ---- Service target
TARGET_P90 = 15.0
ALT_TARGETS = [10.0, 20.0, 30.0]

# ---- Cost (assumed values; state this in the report)
DOCTOR_WAGE_PER_HOUR = 500
SHIFT_HOURS = 4.0

# ---- Inputs estimated from the dataset (synthetic): see estimate_inputs.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project folder (src/ is inside it)
_inputs_path = os.path.join(ROOT, 'data', 'inputs.json')
if not os.path.exists(_inputs_path):
    raise FileNotFoundError(f'{_inputs_path} not found. Run first:  python src/estimate_inputs.py')
with open(_inputs_path) as f:
    _inp = json.load(f)

ARRIVAL_RATE_PEAK = _inp['arrival_rate_per_min']['peak']     # patients per minute
ARRIVAL_RATE_POST = _inp['arrival_rate_per_min']['post']

def _normalize(d, cast):
    total = sum(d.values())
    return {cast(k): v / total for k, v in d.items()}

PRIORITY_MIX = _normalize(_inp['priority_mix'], int)         # sums to exactly 1
DOC_SERVICE_PMF = _normalize(_inp['doc_service_pmf'], float)
MED_SERVICE_PMF = _normalize(_inp['med_service_pmf'], float)
