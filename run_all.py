"""
Single-command reproducibility pipeline for the SEEM3410 Outpatient Clinic DES study.

Usage:
    python run_all.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibrate_inputs import calibrate_from_csv
from clinic_des import load_calibrated_inputs, spawn_replication_uniforms
from config import N_REPLICATIONS, RAW_CSV_PATH, RESULTS_DIR
from experiments import (
    run_factorial_2x2_experiments,
    run_main_experiments,
    run_pharmacy_counter_experiment,
    run_sensitivity_and_thresholds,
)
from plot_results import generate_all_figures
from validate_model import (
    run_analytical_mmc_comparison,
    run_exact_replay_verification,
    run_out_of_sample_validation,
)


def main() -> None:
    t0 = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Calibrating empirical input distributions from Baseline_2Doc_240Peak...")
    calibrate_from_csv()
    cal = load_calibrated_inputs()

    print("[2/5] Running 60,000-patient replay verification, out-of-sample validation, and M/M/c check...")
    raw_df = pd.read_csv(RAW_CSV_PATH)
    replay_df = run_exact_replay_verification(raw_df)
    oos_df = run_out_of_sample_validation(raw_df, cal)
    mmc_df = run_analytical_mmc_comparison(cal)

    print("[3/5] Running 50-replication CRN staffing, pharmacy, 2x2 factorial, and sensitivity experiments...")
    uniforms = spawn_replication_uniforms(N_REPLICATIONS)
    main_out = run_main_experiments(cal, uniforms)
    pharm_df = run_pharmacy_counter_experiment(cal, uniforms)
    cells_df, effects_df = run_factorial_2x2_experiments(cal, uniforms)
    sens_df, scan_df = run_sensitivity_and_thresholds(cal, uniforms)

    print("[4/5] Rendering report figures (figures/fig1..fig5)...")
    generate_all_figures()

    elapsed = time.perf_counter() - t0
    print(f"[5/5] Pipeline completed in {elapsed:.2f}s.\n")

    print("--- Verification Summary (All 6 Scenarios, 300 Runs) ---")
    print(replay_df.to_string(index=False))

    print("\n--- Main Policy Summary (R = 50 CRN Replications) ---")
    cols = [
        "policy_label",
        "p90_doc_wait_mean",
        "p90_doc_wait_ci_low",
        "p90_doc_wait_ci_high",
        "mean_doc_wait_mean",
        "p90_prio1_wait_mean",
        "mean_med_wait_mean",
        "doc_utilization_mean",
        "scheduled_total_cost_hkd_mean",
        "effective_total_cost_hkd_mean",
        "prob_meet_p90_le_15m",
    ]
    print(main_out["summary"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
