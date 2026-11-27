"""
Run input calibration, model validation, simulation experiments, and figure generation.

Usage:
    python run_all.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibrate_inputs import calibrate_from_csv
from clinic_des import create_crn_streams, load_calibrated_inputs
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Calibrating input distributions from Baseline_2Doc_240Peak...")
    calibrate_from_csv()
    cal = load_calibrated_inputs()

    print("Running trace replay verification, held-out scenario validation, and M/M/c comparison...")
    raw_df = pd.read_csv(RAW_CSV_PATH)
    replay_df = run_exact_replay_verification(raw_df)
    run_out_of_sample_validation(raw_df, cal)
    run_analytical_mmc_comparison(cal)

    print("Running 50-replication CRN staffing, pharmacy, factorial, and sensitivity experiments...")
    streams = create_crn_streams(N_REPLICATIONS)
    main_out = run_main_experiments(cal, streams)
    run_pharmacy_counter_experiment(cal, streams)
    run_factorial_2x2_experiments(cal, streams)
    run_sensitivity_and_thresholds(cal, streams)

    print("Generating figures in figures/...")
    generate_all_figures()

    print("\nVerification Summary (All 6 Scenarios, 300 Runs):")
    print(replay_df.to_string(index=False))

    print("\nMain Policy Summary (R = 50 CRN Replications):")
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
