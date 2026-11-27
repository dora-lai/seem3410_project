"""
Configuration constants and staffing cost model for the outpatient clinic simulation.
SEEM3410 Project - Lai Sin Hang (1155213900), Lee Ka Tung (1155212062)
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

RAW_CSV_PATH = DATA_DIR / "hospital_sim_6_scenarios.csv"
CALIBRATED_JSON_PATH = DATA_DIR / "calibrated_inputs.json"

# Fixed master seed for reproducible Common Random Numbers (CRN)
MASTER_SEED = 3410
# Pre-draw 260 uniforms per stream so +10% arrival sensitivity (220 patients) keeps CRN aligned
MAX_STREAM_PATIENTS = 260

N_REPLICATIONS = 50
N_PATIENTS = 200
PEAK_MINUTES = 240.0
BASELINE_SCENARIO = "Baseline_2Doc_240Peak"

POLICIES = {
    "Baseline": {"label": "Baseline (2 Doctors)", "n_doctors": 2, "n_pharmacy": 1},
    "Policy_A": {"label": "Policy A (3 Doctors)", "n_doctors": 3, "n_pharmacy": 1},
    "Policy_B": {"label": "Policy B (4 Doctors)", "n_doctors": 4, "n_pharmacy": 1},
}

# Primary service target: run-level 90th percentile doctor waiting time <= 15 min
PRIMARY_TARGET_MIN = 15.0
ALT_TARGETS_MIN = (10.0, 15.0, 20.0, 30.0)

# Daily staffing cost assumptions (HKD)
# 200 arrivals span roughly 17.2 hours (~1,030 min), so we model a standard
# 18-hour two-shift outpatient operating day (1,080 minutes) plus 1.5x overtime.
CURRENCY = "HKD"
DOCTOR_HOURLY_WAGE = 500.0
PHARMACY_HOURLY_WAGE = 200.0
SCHEDULED_HOURS = 18.0
SCHEDULED_MINUTES = SCHEDULED_HOURS * 60.0
OVERTIME_MULTIPLIER = 1.5


def compute_policy_costs(
    n_doctors,
    n_pharmacy=1,
    makespan_minutes=SCHEDULED_MINUTES,
    doctor_wage=DOCTOR_HOURLY_WAGE,
    pharmacy_wage=PHARMACY_HOURLY_WAGE,
    overtime_multiplier=OVERTIME_MULTIPLIER,
):
    """Calculate scheduled shift cost and overtime cost (in HKD) for one simulated day."""
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
