"""
Input data checks and empirical distribution fitting from hospital_sim_6_scenarios.csv.

We fit all input distributions using only the baseline scenario (Baseline_2Doc_240Peak,
10,000 patient records across 50 runs) so that the other five scenarios can be used
as held-out validation checks.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

from config import (
    BASELINE_SCENARIO,
    CALIBRATED_JSON_PATH,
    PEAK_MINUTES,
    RAW_CSV_PATH,
    RESULTS_DIR,
)


def _pmf_summary(series, cast=int):
    counts = series.value_counts().sort_index()
    total = int(counts.sum())
    return {
        "counts": {str(cast(k)): int(v) for k, v in counts.items()},
        "probs": {str(cast(k)): float(v / total) for k, v in counts.items()},
        "n": total,
        "mean": float(series.mean()),
        "sd": float(series.std(ddof=1)),
        "cv": float(series.std(ddof=1) / series.mean()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def calibrate_from_csv(csv_path=RAW_CSV_PATH):
    df = pd.read_csv(csv_path)

    # Basic sanity checks on the raw CSV (6 scenarios x 50 runs x 200 patients)
    scenario_counts = df.groupby(["Scenario", "NumDoctors", "PeakLimit"]).size()
    assert len(df) == 60_000, f"Expected 60,000 rows, found {len(df)}"
    assert len(scenario_counts) == 6, "Expected 6 scenarios"
    assert (scenario_counts == 10_000).all(), "Each scenario must have 50 x 200 = 10,000 rows"
    assert df.isna().sum().sum() == 0, "Unexpected missing values in raw CSV"

    reconstructed_total = (
        df["DocWaitTime"] + df["DocServiceTime"] + df["MedWaitTime"] + df["MedServiceTime"]
    )
    assert (reconstructed_total == df["TotalTimeInHospital"]).all(), "Total time identity failed"

    # Check that pharmacy service intervals never overlap within a run (confirms 1 pharmacy counter)
    df_tmp = df.copy()
    df_tmp["PharmStart"] = (
        df_tmp["ArrivalTime"]
        + df_tmp["DocWaitTime"]
        + df_tmp["DocServiceTime"]
        + df_tmp["MedWaitTime"]
    )
    df_tmp["PharmEnd"] = df_tmp["PharmStart"] + df_tmp["MedServiceTime"]
    df_tmp = df_tmp.sort_values(["Scenario", "RunID", "PharmStart", "PharmEnd"])
    prev_end = df_tmp.groupby(["Scenario", "RunID"])["PharmEnd"].shift(1)
    max_overlap = float((prev_end - df_tmp["PharmStart"]).dropna().max())
    assert max_overlap <= 0.0, f"Detected overlapping pharmacy service: {max_overlap}"

    # Fit input distributions on Baseline_2Doc_240Peak only
    base = (
        df[df["Scenario"] == BASELINE_SCENARIO]
        .sort_values(["RunID", "PatientID"])
        .reset_index(drop=True)
    )
    # Keep Patient 1's gap from t = 0 via fill_value=0 instead of .diff(), since the first
    # patient's arrival time is already an interarrival draw from clinic opening.
    base["PrevArrivalTime"] = base.groupby("RunID")["ArrivalTime"].shift(1, fill_value=0)
    base["InterarrivalGap"] = base["ArrivalTime"] - base["PrevArrivalTime"]

    # Peak vs off-peak is determined by whether the clock is < 240 min before drawing the next gap
    peak_gaps = base.loc[base["PrevArrivalTime"] < PEAK_MINUTES, "InterarrivalGap"]
    offpeak_gaps = base.loc[base["PrevArrivalTime"] >= PEAK_MINUTES, "InterarrivalGap"]

    peak_info = _pmf_summary(peak_gaps, int)
    offpeak_info = _pmf_summary(offpeak_gaps, int)
    prio_info = _pmf_summary(base["Priority"], int)
    doc_info = _pmf_summary(base["DocServiceTime"], int)
    med_info = _pmf_summary(base["MedServiceTime"], int)

    # Check independence across priority, consultation time, pharmacy time, and consecutive gaps
    ct_doc_prio = pd.crosstab(base["Priority"], base["DocServiceTime"])
    chi2_dp, p_dp, dof_dp, _ = stats.chi2_contingency(ct_doc_prio)

    ct_med_prio = pd.crosstab(base["Priority"], base["MedServiceTime"])
    chi2_mp, p_mp, dof_mp, _ = stats.chi2_contingency(ct_med_prio)

    r_dm, p_dm = stats.pearsonr(base["DocServiceTime"], base["MedServiceTime"])
    r_gap_lag1, p_gap_lag1 = stats.pearsonr(
        peak_gaps.iloc[:-1].to_numpy(), peak_gaps.iloc[1:].to_numpy()
    )

    lam_peak_min = 1.0 / peak_info["mean"]
    lam_off_min = 1.0 / offpeak_info["mean"]
    mu_doc_min = 1.0 / doc_info["mean"]
    mu_med_min = 1.0 / med_info["mean"]

    calibrated = {
        "source_dataset": "hospital_sim_6_scenarios.csv",
        "calibration_scenario": BASELINE_SCENARIO,
        "n_runs": 50,
        "n_patients_per_run": 200,
        "peak_cutoff_minutes": PEAK_MINUTES,
        "interarrival_peak": peak_info,
        "interarrival_offpeak": offpeak_info,
        "priority_class": prio_info,
        "doc_service_time": doc_info,
        "med_service_time": med_info,
        "rates_and_loads": {
            "lambda_peak_per_min": lam_peak_min,
            "lambda_peak_per_hr": lam_peak_min * 60.0,
            "lambda_offpeak_per_min": lam_off_min,
            "lambda_offpeak_per_hr": lam_off_min * 60.0,
            "mu_doc_per_min": mu_doc_min,
            "mu_doc_per_hr": mu_doc_min * 60.0,
            "mu_pharm_per_min": mu_med_min,
            "mu_pharm_per_hr": mu_med_min * 60.0,
            "offered_load_peak_doctors": lam_peak_min / mu_doc_min,
            "offered_load_offpeak_doctors": lam_off_min / mu_doc_min,
            "rho_peak_by_c": {str(c): (lam_peak_min / (c * mu_doc_min)) for c in (2, 3, 4)},
            "rho_offpeak_by_c": {str(c): (lam_off_min / (c * mu_doc_min)) for c in (2, 3, 4)},
        },
        "independence_checks": {
            "doc_service_vs_priority_chi2": {
                "chi2": float(chi2_dp),
                "dof": int(dof_dp),
                "p_value": float(p_dp),
            },
            "med_service_vs_priority_chi2": {
                "chi2": float(chi2_mp),
                "dof": int(dof_mp),
                "p_value": float(p_mp),
            },
            "doc_vs_med_service_pearson": {"r": float(r_dm), "p_value": float(p_dm)},
            "peak_gap_lag1_autocorr": {"r": float(r_gap_lag1), "p_value": float(p_gap_lag1)},
            "max_pharmacy_overlap_minutes": max_overlap,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(calibrated, f, indent=2)

    summary_rows = [
        {
            "Input": "Interarrival gap (Peak, t < 240 min)",
            "Source": f"Empirical ({BASELINE_SCENARIO}, n={peak_info['n']})",
            "Distribution": "Discrete empirical PMF on {3, 4, 5, 6, 7}",
            "Parameters": ", ".join(f"{k}:{v:.3f}" for k, v in peak_info["probs"].items()),
            "Mean": round(peak_info["mean"], 3),
            "SD": round(peak_info["sd"], 3),
            "CV": round(peak_info["cv"], 3),
            "Units": "min",
        },
        {
            "Input": "Interarrival gap (Off-peak, t >= 240 min)",
            "Source": f"Empirical ({BASELINE_SCENARIO}, n={offpeak_info['n']})",
            "Distribution": "Discrete empirical PMF on {3, 4, 5, 6, 7}",
            "Parameters": ", ".join(f"{k}:{v:.3f}" for k, v in offpeak_info["probs"].items()),
            "Mean": round(offpeak_info["mean"], 3),
            "SD": round(offpeak_info["sd"], 3),
            "CV": round(offpeak_info["cv"], 3),
            "Units": "min",
        },
        {
            "Input": "Patient priority class (3 = highest)",
            "Source": f"Empirical ({BASELINE_SCENARIO}, n={prio_info['n']})",
            "Distribution": "Categorical on {1, 2, 3}",
            "Parameters": ", ".join(f"P{k}:{v:.4f}" for k, v in prio_info["probs"].items()),
            "Mean": round(prio_info["mean"], 3),
            "SD": round(prio_info["sd"], 3),
            "CV": round(prio_info["cv"], 3),
            "Units": "class",
        },
        {
            "Input": "Doctor consultation time",
            "Source": f"Empirical ({BASELINE_SCENARIO}, n={doc_info['n']})",
            "Distribution": "Discrete empirical PMF on {10, 13, 15, 18, 20}",
            "Parameters": ", ".join(f"{k}:{v:.4f}" for k, v in doc_info["probs"].items()),
            "Mean": round(doc_info["mean"], 3),
            "SD": round(doc_info["sd"], 3),
            "CV": round(doc_info["cv"], 3),
            "Units": "min",
        },
        {
            "Input": "Pharmacy dispensing time",
            "Source": f"Empirical ({BASELINE_SCENARIO}, n={med_info['n']})",
            "Distribution": "Discrete empirical PMF on {2, 3, 4, 5}",
            "Parameters": ", ".join(f"{k}:{v:.4f}" for k, v in med_info["probs"].items()),
            "Mean": round(med_info["mean"], 3),
            "SD": round(med_info["sd"], 3),
            "CV": round(med_info["cv"], 3),
            "Units": "min",
        },
    ]
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "input_calibration_summary.csv", index=False)
    return calibrated


if __name__ == "__main__":
    cal = calibrate_from_csv()
    r = cal["rates_and_loads"]
    ic = cal["independence_checks"]
    print(f"Saved calibrated parameters to {CALIBRATED_JSON_PATH}")
    print(
        f"Peak gap mean = {cal['interarrival_peak']['mean']:.3f} min "
        f"(rate = {r['lambda_peak_per_hr']:.2f}/hr, CV = {cal['interarrival_peak']['cv']:.3f})"
    )
    print(
        f"Off-peak gap mean = {cal['interarrival_offpeak']['mean']:.3f} min "
        f"(rate = {r['lambda_offpeak_per_hr']:.2f}/hr, CV = {cal['interarrival_offpeak']['cv']:.3f})"
    )
    print(
        f"Doctor service mean = {cal['doc_service_time']['mean']:.3f} min "
        f"(mu = {r['mu_doc_per_hr']:.2f}/hr); Pharmacy mean = {cal['med_service_time']['mean']:.3f} min"
    )
    print(
        "Peak rho by doctors:",
        {k: round(v, 3) for k, v in r["rho_peak_by_c"].items()},
        "| Off-peak rho:",
        {k: round(v, 3) for k, v in r["rho_offpeak_by_c"].items()},
    )
    print(
        f"Independence p-values: Doc~Priority p={ic['doc_service_vs_priority_chi2']['p_value']:.3f}, "
        f"Med~Priority p={ic['med_service_vs_priority_chi2']['p_value']:.3f}, "
        f"Doc~Med p={ic['doc_vs_med_service_pearson']['p_value']:.3f}"
    )
