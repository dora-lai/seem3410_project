"""
test_model.py (Task A6: Model Verification & Unit Tests)
"""
import numpy as np
import config
from model import Patient, generate_patients_from_config, run_policy


def test_hand_checked_example():
    """
    Hand-checked 4-patient example with 1 doctor, 1 pharmacy:
    - P0: arr=0, prio=1, doc=10, pharm=2 -> doc [0, 10], wait=0; pharm [10, 12], wait=0
    - P1: arr=2, prio=1, doc=5,  pharm=2 -> waits behind P2 & P3 -> doc [17, 22], wait=15
    - P2: arr=4, prio=3, doc=4,  pharm=2 -> non-preemptive (waits for P0), first in queue -> doc [10, 14], wait=6
    - P3: arr=5, prio=3, doc=3,  pharm=2 -> FCFS within prio 3 (after P2) -> doc [14, 17], wait=9
    """
    patients = [
        Patient(0, arrival_time=0.0, priority=1, doc_service_time=10.0, pharm_service_time=2.0),
        Patient(1, arrival_time=2.0, priority=1, doc_service_time=5.0,  pharm_service_time=2.0),
        Patient(2, arrival_time=4.0, priority=3, doc_service_time=4.0,  pharm_service_time=2.0),
        Patient(3, arrival_time=5.0, priority=3, doc_service_time=3.0,  pharm_service_time=2.0),
    ]
    df, summary = run_policy(patients, n_doctors=1, n_pharmacy=1)

    # Check exact doctor start times (verifies non-preemption, priority order, FCFS within class)
    expected_doc_start = [0.0, 17.0, 10.0, 14.0]
    expected_doc_wait = [0.0, 15.0, 6.0, 9.0]
    expected_departure = [12.0, 24.0, 16.0, 19.0]

    assert np.allclose(df["doc_start_time"], expected_doc_start), f"Got {df['doc_start_time'].tolist()}"
    assert np.allclose(df["doc_wait"], expected_doc_wait), f"Got {df['doc_wait'].tolist()}"
    assert np.allclose(df["departure_time"], expected_departure), f"Got {df['departure_time'].tolist()}"
    print("[PASS] Hand-checked example (non-preemptive priority & FCFS within class verified)")


def test_full_run_invariants():
    """Verify 200 departures, no negative waits, and CRN consistency across policies."""
    base_patients = generate_patients_from_config(seed=config.SEED_BASE)

    results = {}
    for name, params in config.POLICIES.items():
        df, summary = run_policy(base_patients, **params)

        # 1. All 200 patients depart
        assert summary["n_departed"] == config.N_PATIENTS, f"{name}: only {summary['n_departed']} departed"
        # 2. No negative waiting times
        assert (df["doc_wait"] >= -1e-9).all(), f"{name}: negative doctor wait found"
        assert (df["pharm_wait"] >= -1e-9).all(), f"{name}: negative pharmacy wait found"
        # 3. Total time >= service times
        assert (df["total_time"] >= df["doc_service_time"] + df["pharm_service_time"] - 1e-9).all()

        results[name] = df

    # 4. CRN check: identical arrivals, priorities, and service times across all policies
    ref = results["Baseline"]
    for name in ["Policy_A", "Policy_B"]:
        assert np.allclose(ref["arrival_time"], results[name]["arrival_time"])
        assert (ref["priority"] == results[name]["priority"]).all()
        assert np.allclose(ref["doc_service_time"], results[name]["doc_service_time"])
        assert np.allclose(ref["pharm_service_time"], results[name]["pharm_service_time"])

    print("[PASS] 200 departures, non-negative waits, and CRN consistency verified across all policies")


if __name__ == "__main__":
    test_hand_checked_example()
    test_full_run_invariants()
    print("\nAll A6 tests passed!")