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

# Secondary analysis: optional second pharmacy counter                           # <-- NEW
SECONDARY_POLICIES = {                                                         # <-- NEW
    'Baseline_2Pharm': {'n_doctors': 2, 'n_pharmacy': 2},                      # <-- NEW
    'Policy_A_2Pharm': {'n_doctors': 3, 'n_pharmacy': 2},                      # <-- NEW
    'Policy_B_2Pharm': {'n_doctors': 4, 'n_pharmacy': 2},                      # <-- NEW
}                                                                              # <-- NEW

# ---- Service target
TARGET_P90 = 15.0
ALT_TARGETS = [10.0, 20.0, 30.0]

# ---- Cost (assumed values; state this in the report)
CURRENCY = 'HKD'                                                               # <-- NEW
DOCTOR_WAGE_PER_HOUR = 500      # Assumed hourly wage per doctor (HKD/hr)
PHARMACY_WAGE_PER_HOUR = 200    # Assumed hourly wage per pharmacy counter     # <-- NEW
SHIFT_HOURS = 4.0               # Same fixed shift length for all policies (matches 240-min peak)

def compute_staffing_cost(n_doctors, n_pharmacy=1, include_pharmacy=False):    # <-- NEW
    """Return staffing cost per shift for a given policy."""                   # <-- NEW
    cost = n_doctors * DOCTOR_WAGE_PER_HOUR * SHIFT_HOURS                      # <-- NEW
    if include_pharmacy:                                                       # <-- NEW
        cost += n_pharmacy * PHARMACY_WAGE_PER_HOUR * SHIFT_HOURS              # <-- NEW
    return cost                                                                # <-- NEW

# ---- Inputs estimated from the dataset (synthetic): see estimate_inputs.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project folder (src/ is inside it)
_inputs_path = os.path.join(ROOT, 'data', 'inputs.json')
if not os.path.exists(_inputs_path):
    raise FileNotFoundError(f'{_inputs_path} not found. Run first:  python src/estimate_inputs.py')
with open(_inputs_path) as f:
    _inp = json.load(f)

ARRIVAL_RATE_PEAK = _inp['arrival_rate_per_min']['peak']     # patients per minute (for reference only)
ARRIVAL_RATE_POST = _inp['arrival_rate_per_min']['post']

def _normalize(d, cast):
    total = sum(d.values())
    return {cast(k): v / total for k, v in d.items()}

PRIORITY_MIX = _normalize(_inp['priority_mix'], int)         # sums to exactly 1
DOC_SERVICE_PMF = _normalize(_inp['doc_service_pmf'], float)
MED_SERVICE_PMF = _normalize(_inp['med_service_pmf'], float)
ARRIVAL_GAP_PMF = {period: _normalize(pmf, float) for period, pmf in _inp['arrival_gap_pmf'].items()}   # minutes between arrivals