"""
Central configuration for the SEEM3410 Outpatient Clinic DES study.
Team: Lai Sin Hang (1155213900), Lee Ka Tung (1155212062)
"""
from pathlib import Path

# Project directory layout
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

RAW_CSV_PATH = DATA_DIR / "hospital_sim_6_scenarios.csv"
CALIBRATED_JSON_PATH = DATA_DIR / "calibrated_inputs.json"

# Random number generation (Week 4: SeedSequence with independent substreams)
MASTER_SEED = 3410
MAX_STREAM_PATIENTS = 260  # Pre-allocated stream length to keep CRN prefixes aligned

# Baseline operating horizon and replication settings
N_REPLICATIONS = 50
N_PATIENTS = 200
PEAK_MINUTES = 240.0
BASELINE_SCENARIO = "Baseline_2Doc_240Peak"

# Candidate staffing policies (Proposal Table)
POLICIES = {
    "Baseline": {"label": "Baseline (2 Doctors)", "n_doctors": 2, "n_pharmacy": 1},
    "Policy_A": {"label": "Policy A (3 Doctors)", "n_doctors": 3, "n_pharmacy": 1},
    "Policy_B": {"label": "Policy B (4 Doctors)", "n_doctors": 4, "n_pharmacy": 1},
}

# Service-level target on run-level 90th-percentile doctor waiting time (minutes)
PRIMARY_TARGET_MIN = 15.0
ALT_TARGETS_MIN = (10.0, 15.0, 20.0, 30.0)

# Assumed clinic staffing cost parameters (HKD)
# A 200-patient clinic day spans ~1,030 arrival minutes (~17.2 hrs), modeled as a
# scheduled 18-hour two-shift outpatient day (2 x 9-hour shifts = 1,080 minutes).
CURRENCY = "HKD"
DOCTOR_HOURLY_WAGE = 500.0     # HKD per doctor-hour (scheduled shift)
PHARMACY_HOURLY_WAGE = 200.0   # HKD per pharmacy-counter-hour (scheduled shift)
SCHEDULED_HOURS = 18.0         # Fixed scheduled day length across all policies (1,080 min)
SCHEDULED_MINUTES = SCHEDULED_HOURS * 60.0
OVERTIME_MULTIPLIER = 1.5      # Wage multiplier for clinic hours beyond SCHEDULED_MINUTES


def compute_policy_costs(
    n_doctors: int,
    n_pharmacy: int = 1,
    makespan_minutes: float = SCHEDULED_MINUTES,
    doctor_wage: float = DOCTOR_HOURLY_WAGE,
    pharmacy_wage: float = PHARMACY_HOURLY_WAGE,
    overtime_multiplier: float = OVERTIME_MULTIPLIER,
) -> dict:
    """Compute scheduled shift costs and overtime exposure (in HKD) for one run."""
    scheduled_doc_cost = n_doctors * doctor_wage * SCHEDULED_HOURS
    scheduled_pharm_cost = n_pharmacy * pharmacy_wage * SCHEDULED_HOURS
    overtime_hours = max(0.0, makespan_minutes - SCHEDULED_MINUTES) / 60.0
    overtime_doc_cost = n_doctors * doctor_wage * overtime_multiplier * overtime_hours
    overtime_pharm_cost = n_pharmacy * pharmacy_wage * overtime_multiplier * overtime_hours
    return {
        "scheduled_doc_cost_hkd": scheduled_doc_cost,
        "scheduled_total_cost_hkd": scheduled_doc_cost + scheduled_pharm_cost,
        "overtime_hours": overtime_hours,
        "overtime_doc_cost_hkd": overtime_doc_cost,
        "effective_doc_cost_hkd": scheduled_doc_cost + overtime_doc_cost,
        "effective_total_cost_hkd": (
            scheduled_doc_cost + scheduled_pharm_cost + overtime_doc_cost + overtime_pharm_cost
        ),
    }
