"""
config.py
All simulation parameters and seeds are gathered here.
"""

# ============================================================
# Random Seed Setting
# ============================================================
SEED_BASE = 3410

# ============================================================
# Simulation Scale
# ============================================================
N_REPLICATIONS = 50
N_PATIENTS = 200
PEAK_MINUTES = 240

# ============================================================
# 3 Policies
# ============================================================
POLICIES = {
    'Baseline': {'n_doctors': 2, 'n_pharmacy': 1},
    'Policy_A': {'n_doctors': 3, 'n_pharmacy': 1},
    'Policy_B': {'n_doctors': 4, 'n_pharmacy': 1},
}

# ============================================================
# Service Target
# ============================================================
TARGET_P90 = 15.0
ALT_TARGETS = [10.0, 20.0, 30.0]

# ============================================================
# Cost parameters (assumed values; to be confirmed)
# ============================================================
DOCTOR_WAGE_PER_HOUR = 500
SHIFT_HOURS = 4.0

# ============================================================
# Input distribution parameters (provisional values; to be updated once B3–B5 are completed)
# ============================================================
# Arrival rate (patients per hour)
ARRIVAL_RATE = 11.6 

# Priority mix (P1, P2, P3)
PRIORITY_MIX = {1: 0.658, 2: 0.294, 3: 0.049}

# Doctor service time (minutes) - triangular distribution
DOCTOR_SERVICE_MIN = 10.0
DOCTOR_SERVICE_MODE = 13.4
DOCTOR_SERVICE_MAX = 20.0

# Pharmacy service time (minutes) - triangular distribution
PHARMACY_SERVICE_MIN = 2.0
PHARMACY_SERVICE_MODE = 3.0
PHARMACY_SERVICE_MAX = 5.0
