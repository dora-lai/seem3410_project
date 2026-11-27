"""
Unit and verification tests for the outpatient clinic DES model.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibrate_inputs import calibrate_from_csv
from clinic_des import (
    CALIBRATED_JSON_PATH,
    create_crn_streams,
    generate_daily_patients,
    load_calibrated_inputs,
    simulate_clinic,
)
from config import RAW_CSV_PATH


def test_non_preemptive_priority_and_fcfs_tie_break():
    """
    Hand-checked 4-patient trace with 1 doctor and 1 pharmacy counter:
      - P1 (prio 1) arrives at t=0, consults [0, 10], pharmacy [10, 13] -> doc_wait = 0
      - P2 (prio 1) arrives at t=2, waits in queue
      - P3 (prio 3) arrives at t=4 -> higher priority than P2, does not preempt P1;
        starts consult at t=10 -> doc_wait = 10 - 4 = 6
      - P4 (prio 3) arrives at t=10 (same timestamp as P1 DOC_END) -> enters queue before
        doctor picks next patient; ties priority=3 with P3, so FCFS serves P3 at t=10
        and P4 at t=15 -> P4 doc_wait = 5, P2 starts at t=20 -> doc_wait = 18.
    """
    arr = np.array([0.0, 2.0, 4.0, 10.0])
    prio = np.array([1, 1, 3, 3])
    doc_s = np.array([10.0, 5.0, 5.0, 5.0])
    med_s = np.array([3.0, 3.0, 3.0, 3.0])

    res = simulate_clinic(arr, prio, doc_s, med_s, n_doctors=1, n_pharmacy=1, trace_limit=20)
    pats = res["patients"]
    summ = res["summary"]

    np.testing.assert_allclose(pats["doc_wait"], [0.0, 18.0, 6.0, 5.0])
    np.testing.assert_allclose(pats["med_wait"], [0.0, 0.0, 0.0, 0.0])
    assert abs(summ["area_doc_queue"] - float(np.sum(pats["doc_wait"]))) < 1e-12
    assert summ["makespan"] == 28.0


def test_sample_path_littles_law_and_crn_determinism():
    if not CALIBRATED_JSON_PATH.exists():
        calibrate_from_csv()
    cal = load_calibrated_inputs()
    s1 = create_crn_streams(3, seed=3410)
    s2 = create_crn_streams(3, seed=3410)

    for a, b in zip(s1, s2):
        np.testing.assert_allclose(a["u_arrival"], b["u_arrival"])
        day = generate_daily_patients(a, cal)
        out2 = simulate_clinic(
            day["arrival_times"], day["priorities"], day["doc_services"], day["med_services"], n_doctors=2
        )
        out3 = simulate_clinic(
            day["arrival_times"], day["priorities"], day["doc_services"], day["med_services"], n_doctors=3
        )
        out4 = simulate_clinic(
            day["arrival_times"], day["priorities"], day["doc_services"], day["med_services"], n_doctors=4
        )

        # Over a clearing day, integral of queue length equals total waiting time
        assert abs(out3["summary"]["area_doc_queue"] - float(np.sum(out3["patients"]["doc_wait"]))) < 1e-9
        assert abs(out3["summary"]["area_pharm_queue"] - float(np.sum(out3["patients"]["med_wait"]))) < 1e-9

        # Adding doctors on the exact same patient day should not increase mean consultation wait
        assert out4["summary"]["mean_doc_wait"] <= out3["summary"]["mean_doc_wait"] <= out2["summary"]["mean_doc_wait"]


def test_exact_historical_replay_zero_mismatch():
    df = pd.read_csv(RAW_CSV_PATH)
    # Check RunID=1 across all 6 scenarios (1,200 patients)
    sub_df = df[df["RunID"] == 1]
    for (_, _), grp in sub_df.groupby(["Scenario", "RunID"]):
        grp = grp.sort_values("PatientID")
        out = simulate_clinic(
            grp["ArrivalTime"].to_numpy(dtype=float),
            grp["Priority"].to_numpy(dtype=int),
            grp["DocServiceTime"].to_numpy(dtype=float),
            grp["MedServiceTime"].to_numpy(dtype=float),
            n_doctors=int(grp["NumDoctors"].iloc[0]),
            n_pharmacy=1,
            peak_minutes=float(grp["PeakLimit"].iloc[0]),
        )
        np.testing.assert_allclose(out["patients"]["doc_wait"], grp["DocWaitTime"].to_numpy(dtype=float), atol=0.0)
        np.testing.assert_allclose(out["patients"]["med_wait"], grp["MedWaitTime"].to_numpy(dtype=float), atol=0.0)
        np.testing.assert_allclose(
            out["patients"]["total_time"], grp["TotalTimeInHospital"].to_numpy(dtype=float), atol=0.0
        )
