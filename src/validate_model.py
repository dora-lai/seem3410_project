"""
Model verification and validation checks:
  1. Replay all 300 runs (60,000 patients) in hospital_sim_6_scenarios.csv through
     simulate_clinic to check for exact agreement on waiting times, total hospital
     time, and sample-path Little's Law.
  2. Validate our baseline-calibrated simulation against all 6 scenarios in the dataset
     (5 of which were held out during input fitting).
  3. Compare stationary M/M/c (Erlang C) formulas against the simulation to show why
     steady-state queueing formulas do not capture peak-period buildup or priorities.
"""
import math
import numpy as np
import pandas as pd
from scipy import stats

from calibrate_inputs import calibrate_from_csv
from clinic_des import (
    CALIBRATED_JSON_PATH,
    create_crn_streams,
    generate_daily_patients,
    load_calibrated_inputs,
    simulate_clinic,
)
from config import N_PATIENTS, N_REPLICATIONS, RAW_CSV_PATH, RESULTS_DIR


def erlang_c_wq(lam_per_min, mu_per_min, c):
    """
    Steady-state M/M/c mean queue wait W_q (minutes) and utilization rho = lam / (c * mu).
    Returns (rho, inf) when rho >= 1.0 since the steady-state queue does not exist.
    """
    a = lam_per_min / mu_per_min
    rho = a / c
    if rho >= 1.0:
        return rho, math.inf
    sum_terms = sum((a ** k) / math.factorial(k) for k in range(c))
    last_term = ((a ** c) / math.factorial(c)) * (1.0 / (1.0 - rho))
    p0 = 1.0 / (sum_terms + last_term)
    pw = last_term * p0
    wq = pw / (c * mu_per_min - lam_per_min)
    return rho, wq


def run_exact_replay_verification(df):
    """Feed historical arrival and service times into simulate_clinic and verify zero error."""
    records = []
    for (scenario, run_id), grp in df.groupby(["Scenario", "RunID"], sort=True):
        sub = grp.sort_values("PatientID").reset_index(drop=True)
        n_doc = int(sub["NumDoctors"].iloc[0])
        peak_lim = float(sub["PeakLimit"].iloc[0])

        out = simulate_clinic(
            arrival_times=sub["ArrivalTime"].to_numpy(dtype=float),
            priorities=sub["Priority"].to_numpy(dtype=int),
            doc_services=sub["DocServiceTime"].to_numpy(dtype=float),
            med_services=sub["MedServiceTime"].to_numpy(dtype=float),
            n_doctors=n_doc,
            n_pharmacy=1,
            peak_minutes=peak_lim,
        )
        pats = out["patients"]
        summ = out["summary"]

        dw_err = float(np.max(np.abs(pats["doc_wait"] - sub["DocWaitTime"].to_numpy(dtype=float))))
        mw_err = float(np.max(np.abs(pats["med_wait"] - sub["MedWaitTime"].to_numpy(dtype=float))))
        tt_err = float(
            np.max(np.abs(pats["total_time"] - sub["TotalTimeInHospital"].to_numpy(dtype=float)))
        )

        # Sample-path Little's Law over a clearing day: integral of Q(t) dt == sum of waits
        littles_doc_gap = abs(summ["area_doc_queue"] - float(np.sum(pats["doc_wait"])))
        littles_pharm_gap = abs(summ["area_pharm_queue"] - float(np.sum(pats["med_wait"])))

        records.append(
            {
                "Scenario": scenario,
                "RunID": int(run_id),
                "NumDoctors": n_doc,
                "PeakLimit": peak_lim,
                "max_abs_doc_wait_err": dw_err,
                "max_abs_med_wait_err": mw_err,
                "max_abs_total_time_err": tt_err,
                "littles_doc_abs_err": littles_doc_gap,
                "littles_pharm_abs_err": littles_pharm_gap,
            }
        )

    rep_df = pd.DataFrame(records)
    summary = (
        rep_df.groupby(["Scenario", "NumDoctors", "PeakLimit"], as_index=False)
        .agg(
            runs_checked=("RunID", "count"),
            max_doc_wait_err=("max_abs_doc_wait_err", "max"),
            max_med_wait_err=("max_abs_med_wait_err", "max"),
            max_total_time_err=("max_abs_total_time_err", "max"),
            max_littles_doc_err=("littles_doc_abs_err", "max"),
            max_littles_pharm_err=("littles_pharm_abs_err", "max"),
        )
    )
    summary.to_csv(RESULTS_DIR / "verification_exact_replay.csv", index=False)
    return summary


def run_out_of_sample_validation(df, cal):
    """
    Simulate 50 runs for each of the 6 scenarios using only the baseline-fitted inputs
    and compare run-level waiting times against the historical CSV.
    """
    streams = create_crn_streams(N_REPLICATIONS)
    t_crit = float(stats.t.ppf(0.975, N_REPLICATIONS - 1))

    rows = []
    scenario_specs = (
        df[["Scenario", "NumDoctors", "PeakLimit"]]
        .drop_duplicates()
        .sort_values(["PeakLimit", "NumDoctors"])
    )

    for _, spec in scenario_specs.iterrows():
        scen = str(spec["Scenario"])
        n_doc = int(spec["NumDoctors"])
        peak_lim = float(spec["PeakLimit"])

        emp_grp = df[df["Scenario"] == scen]
        emp_run_p90 = emp_grp.groupby("RunID")["DocWaitTime"].quantile(0.90)
        emp_run_mean_dw = emp_grp.groupby("RunID")["DocWaitTime"].mean()
        emp_run_mean_tot = emp_grp.groupby("RunID")["TotalTimeInHospital"].mean()

        sim_p90, sim_mean_dw, sim_mean_tot = [], [], []
        for u in streams:
            day = generate_daily_patients(
                u, cal, n_patients=N_PATIENTS, peak_minutes=peak_lim, arrival_mode="empirical"
            )
            res = simulate_clinic(
                day["arrival_times"],
                day["priorities"],
                day["doc_services"],
                day["med_services"],
                n_doctors=n_doc,
                n_pharmacy=1,
                peak_minutes=peak_lim,
            )["summary"]
            sim_p90.append(res["p90_doc_wait"])
            sim_mean_dw.append(res["mean_doc_wait"])
            sim_mean_tot.append(res["mean_total_time"])

        emp_m = float(emp_run_p90.mean())
        emp_hw = t_crit * float(emp_run_p90.std(ddof=1) / np.sqrt(50))
        sim_s = pd.Series(sim_p90)
        sim_m = float(sim_s.mean())
        sim_hw = t_crit * float(sim_s.std(ddof=1) / np.sqrt(50))
        _, p_val = stats.ttest_ind(emp_run_p90.to_numpy(), sim_s.to_numpy(), equal_var=False)

        rows.append(
            {
                "Scenario": scen,
                "Split": "Calibration" if scen == "Baseline_2Doc_240Peak" else "Held-out validation",
                "Doctors": n_doc,
                "Peak_Cutoff_Min": int(peak_lim),
                "Empirical_P90_Mean": round(emp_m, 2),
                "Empirical_P90_CI95": f"[{max(0.0, emp_m - emp_hw):.2f}, {emp_m + emp_hw:.2f}]",
                "DES_P90_Mean": round(sim_m, 2),
                "DES_P90_CI95": f"[{max(0.0, sim_m - sim_hw):.2f}, {sim_m + sim_hw:.2f}]",
                "Empirical_Mean_DocWait": round(float(emp_run_mean_dw.mean()), 2),
                "DES_Mean_DocWait": round(float(np.mean(sim_mean_dw)), 2),
                "Empirical_Mean_TotalTime": round(float(emp_run_mean_tot.mean()), 2),
                "DES_Mean_TotalTime": round(float(np.mean(sim_mean_tot)), 2),
                "Welch_p_value_P90": round(float(p_val), 3) if not np.isnan(p_val) else 1.0,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS_DIR / "validation_out_of_sample.csv", index=False)
    return out_df


def run_analytical_mmc_comparison(cal):
    """Compare M/M/c (Erlang C) steady-state formulas against the transient DES model."""
    raw = cal["raw"]
    lam_pk = raw["rates_and_loads"]["lambda_peak_per_min"]
    lam_of = raw["rates_and_loads"]["lambda_offpeak_per_min"]
    mean_arrival_span = 200.0 / ((240.0 * lam_pk + 792.0 * lam_of) / 1032.0)
    lam_day = 200.0 / mean_arrival_span
    mu_doc = raw["rates_and_loads"]["mu_doc_per_min"]

    streams = create_crn_streams(N_REPLICATIONS)
    rows = []
    for c in (2, 3, 4):
        rho_pk, wq_pk = erlang_c_wq(lam_pk, mu_doc, c)
        rho_of, wq_of = erlang_c_wq(lam_of, mu_doc, c)
        rho_day, wq_day = erlang_c_wq(lam_day, mu_doc, c)

        des_wq_emp, des_p90_emp, des_wq_pois, des_p90_pois = [], [], [], []
        for u in streams:
            d_emp = generate_daily_patients(u, cal, arrival_mode="empirical")
            s_emp = simulate_clinic(
                d_emp["arrival_times"],
                d_emp["priorities"],
                d_emp["doc_services"],
                d_emp["med_services"],
                n_doctors=c,
            )["summary"]
            des_wq_emp.append(s_emp["mean_doc_wait"])
            des_p90_emp.append(s_emp["p90_doc_wait"])

            d_pois = generate_daily_patients(u, cal, arrival_mode="poisson")
            s_pois = simulate_clinic(
                d_pois["arrival_times"],
                d_pois["priorities"],
                d_pois["doc_services"],
                d_pois["med_services"],
                n_doctors=c,
            )["summary"]
            des_wq_pois.append(s_pois["mean_doc_wait"])
            des_p90_pois.append(s_pois["p90_doc_wait"])

        rows.append(
            {
                "Doctors_c": c,
                "rho_peak": round(rho_pk, 3),
                "rho_offpeak": round(rho_of, 3),
                "rho_day_avg": round(rho_day, 3),
                "MMc_Peak_Wq_min": "Unstable (inf)" if math.isinf(wq_pk) else f"{wq_pk:.2f}",
                "MMc_OffPeak_Wq_min": "Unstable (inf)" if math.isinf(wq_of) else f"{wq_of:.2f}",
                "MMc_DayAvg_Wq_min": "Unstable (inf)" if math.isinf(wq_day) else f"{wq_day:.2f}",
                "DES_Empirical_Mean_Wq_min": round(float(np.mean(des_wq_emp)), 2),
                "DES_Empirical_P90_Wq_min": round(float(np.mean(des_p90_emp)), 2),
                "DES_Poisson_Mean_Wq_min": round(float(np.mean(des_wq_pois)), 2),
                "DES_Poisson_P90_Wq_min": round(float(np.mean(des_p90_pois)), 2),
            }
        )

    mmc_df = pd.DataFrame(rows)
    mmc_df.to_csv(RESULTS_DIR / "analytical_mmc_vs_des.csv", index=False)
    return mmc_df


if __name__ == "__main__":
    if not CALIBRATED_JSON_PATH.exists():
        calibrate_from_csv()
    cal = load_calibrated_inputs()
    raw_df = pd.read_csv(RAW_CSV_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    replay_summary = run_exact_replay_verification(raw_df)
    print("Exact historical trace replay (60,000 patients across 300 runs):")
    print(replay_summary.to_string(index=False))

    oos_summary = run_out_of_sample_validation(raw_df, cal)
    print("\nOut-of-sample validation across all 6 scenarios:")
    cols = ["Scenario", "Split", "Empirical_P90_Mean", "DES_P90_Mean", "Welch_p_value_P90"]
    print(oos_summary[cols].to_string(index=False))

    mmc_summary = run_analytical_mmc_comparison(cal)
    print("\nStationary M/M/c vs transient priority DES:")
    print(mmc_summary.to_string(index=False))
