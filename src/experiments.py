"""
Simulation experimental design and policy analysis (SEEM3410 Week 6):
  1. Main 3-policy CRN comparison (2, 3, 4 doctors; 50 paired replications)
  2. Paired t-confidence intervals and CRN variance-reduction quantification
  3. Downstream 1- vs 2-counter pharmacy analysis
  4. Week 6 2^2 full factorial experiments (main effects and interaction effects)
  5. One-way sensitivity suite and Policy A break-even threshold scan
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from calibrate_inputs import calibrate_from_csv
from clinic_des import (
    CALIBRATED_JSON_PATH,
    load_calibrated_inputs,
    realize_patient_cohort,
    simulate_clinic,
    spawn_replication_uniforms,
)
from config import (
    ALT_TARGETS_MIN,
    N_PATIENTS,
    N_REPLICATIONS,
    PEAK_MINUTES,
    POLICIES,
    PRIMARY_TARGET_MIN,
    RESULTS_DIR,
    compute_policy_costs,
)


def mean_ci95(values: np.ndarray | pd.Series, nonneg: bool = False) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    se = sd / np.sqrt(n)
    hw = float(stats.t.ppf(0.975, n - 1) * se)
    lo = max(0.0, mean - hw) if nonneg else (mean - hw)
    return mean, sd, lo, mean + hw


def paired_diff_stats(y_new: np.ndarray, y_ref: np.ndarray) -> dict:
    diff = np.asarray(y_new, dtype=float) - np.asarray(y_ref, dtype=float)
    n = len(diff)
    mean_d, sd_d, lo_d, hi_d = mean_ci95(diff)
    se_crn = sd_d / np.sqrt(n)
    se_indep = float(np.sqrt(np.var(y_new, ddof=1) / n + np.var(y_ref, ddof=1) / n))
    vr_ratio = (se_indep ** 2) / (se_crn ** 2) if se_crn > 0 else np.nan
    t_stat, p_val = stats.ttest_1samp(diff, 0.0)
    return {
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "ci95_low": lo_d,
        "ci95_high": hi_d,
        "se_crn": se_crn,
        "se_independent": se_indep,
        "variance_reduction_factor": float(vr_ratio),
        "p_value": float(p_val),
    }


def run_main_experiments(cal: dict, uniforms: list[dict]) -> dict[str, pd.DataFrame]:
    traj_grid = np.arange(0.0, 1405.0, 5.0)
    rep_rows = []
    traj_store: dict[str, list[np.ndarray]] = {k: [] for k in POLICIES}

    for u in uniforms:
        cohort = realize_patient_cohort(u, cal, n_patients=N_PATIENTS, arrival_mode="empirical")
        for pol_key, pol_cfg in POLICIES.items():
            out = simulate_clinic(
                cohort["arrival_times"],
                cohort["priorities"],
                cohort["doc_services"],
                cohort["med_services"],
                n_doctors=pol_cfg["n_doctors"],
                n_pharmacy=pol_cfg["n_pharmacy"],
                trajectory_grid=traj_grid,
            )
            row = {"policy_key": pol_key, "policy_label": pol_cfg["label"], "rep_id": u["rep_id"]}
            row.update(out["summary"])
            rep_rows.append(row)
            traj_store[pol_key].append(out["trajectory_q_doc"])

    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(RESULTS_DIR / "main_replications.csv", index=False)

    # Aggregate policy summary table with 95% t-CIs
    metrics = [
        "p90_doc_wait",
        "mean_doc_wait",
        "median_doc_wait",
        "p95_doc_wait",
        "p90_prio1_wait",
        "p90_prio2_wait",
        "p90_prio3_wait",
        "mean_prio1_wait",
        "mean_prio2_wait",
        "mean_prio3_wait",
        "p90_peak_wait",
        "p90_offpeak_wait",
        "mean_med_wait",
        "p90_med_wait",
        "mean_total_time",
        "doc_utilization",
        "pharm_utilization",
        "time_avg_doc_queue",
        "makespan",
        "overtime_hours",
        "scheduled_doc_cost_hkd",
        "scheduled_total_cost_hkd",
        "effective_doc_cost_hkd",
        "effective_total_cost_hkd",
    ]

    summary_rows = []
    for pol_key, pol_cfg in POLICIES.items():
        sub = rep_df[rep_df["policy_key"] == pol_key].sort_values("rep_id")
        s_row: dict[str, object] = {
            "policy_key": pol_key,
            "policy_label": pol_cfg["label"],
            "n_doctors": pol_cfg["n_doctors"],
            "n_pharmacy": pol_cfg["n_pharmacy"],
        }
        for m in metrics:
            mean_v, sd_v, lo_v, hi_v = mean_ci95(sub[m], nonneg=True)
            s_row[f"{m}_mean"] = round(mean_v, 4)
            s_row[f"{m}_sd"] = round(sd_v, 4)
            s_row[f"{m}_ci_low"] = round(lo_v, 4)
            s_row[f"{m}_ci_high"] = round(hi_v, 4)
        for tgt in ALT_TARGETS_MIN:
            s_row[f"prob_meet_p90_le_{int(tgt)}m"] = round(
                float((sub["p90_doc_wait"] <= tgt).mean()), 3
            )
        summary_rows.append(s_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "main_policy_summary.csv", index=False)

    # CRN paired comparisons across policies
    pairs = [
        ("Policy_A", "Baseline", "Policy A (3 Docs) vs Baseline (2 Docs)"),
        ("Policy_B", "Baseline", "Policy B (4 Docs) vs Baseline (2 Docs)"),
        ("Policy_B", "Policy_A", "Policy B (4 Docs) vs Policy A (3 Docs)"),
    ]
    paired_metrics = [
        "p90_doc_wait",
        "mean_doc_wait",
        "p90_prio1_wait",
        "mean_med_wait",
        "mean_total_time",
        "effective_total_cost_hkd",
    ]
    paired_rows = []
    for new_k, ref_k, label in pairs:
        s_new = rep_df[rep_df["policy_key"] == new_k].sort_values("rep_id")
        s_ref = rep_df[rep_df["policy_key"] == ref_k].sort_values("rep_id")
        for m in paired_metrics:
            st = paired_diff_stats(s_new[m].to_numpy(), s_ref[m].to_numpy())
            paired_rows.append({"comparison": label, "metric": m, **st})

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(RESULTS_DIR / "paired_comparisons.csv", index=False)

    # Save 5-minute queue trajectory quantiles for Figure 3
    traj_rows = []
    for idx, t_min in enumerate(traj_grid):
        t_row: dict[str, float] = {"time_min": float(t_min)}
        for pol_key in POLICIES:
            mat = np.asarray(traj_store[pol_key])[:, idx]
            t_row[f"{pol_key}_mean_q"] = float(np.mean(mat))
            t_row[f"{pol_key}_p10_q"] = float(np.quantile(mat, 0.10))
            t_row[f"{pol_key}_p90_q"] = float(np.quantile(mat, 0.90))
        traj_rows.append(t_row)
    traj_df = pd.DataFrame(traj_rows)
    traj_df.to_csv(RESULTS_DIR / "queue_trajectories.csv", index=False)

    return {
        "replications": rep_df,
        "summary": summary_df,
        "paired": paired_df,
        "trajectories": traj_df,
    }


def run_pharmacy_counter_experiment(cal: dict, uniforms: list[dict]) -> pd.DataFrame:
    """Evaluate 1 vs 2 pharmacy counters across 2, 3, and 4 doctors using CRN."""
    rows = []
    for n_doc in (2, 3, 4):
        res_m = {1: [], 2: []}
        for u in uniforms:
            cohort = realize_patient_cohort(u, cal, n_patients=N_PATIENTS, arrival_mode="empirical")
            for n_ph in (1, 2):
                res_m[n_ph].append(
                    simulate_clinic(
                        cohort["arrival_times"],
                        cohort["priorities"],
                        cohort["doc_services"],
                        cohort["med_services"],
                        n_doctors=n_doc,
                        n_pharmacy=n_ph,
                    )["summary"]
                )
        df1, df2 = pd.DataFrame(res_m[1]), pd.DataFrame(res_m[2])
        diff_mw = paired_diff_stats(df2["mean_med_wait"].to_numpy(), df1["mean_med_wait"].to_numpy())
        diff_tt = paired_diff_stats(df2["mean_total_time"].to_numpy(), df1["mean_total_time"].to_numpy())

        for n_ph, df_cur in ((1, df1), (2, df2)):
            rows.append(
                {
                    "n_doctors": n_doc,
                    "n_pharmacy": n_ph,
                    "p90_doc_wait_mean": round(float(df_cur["p90_doc_wait"].mean()), 2),
                    "mean_med_wait": round(float(df_cur["mean_med_wait"].mean()), 3),
                    "p90_med_wait": round(float(df_cur["p90_med_wait"].mean()), 3),
                    "pharm_utilization": round(float(df_cur["pharm_utilization"].mean()), 3),
                    "mean_total_time": round(float(df_cur["mean_total_time"].mean()), 2),
                    "scheduled_total_cost_hkd": round(float(df_cur["scheduled_total_cost_hkd"].mean()), 0),
                    "paired_delta_med_wait_2_minus_1": round(diff_mw["mean_diff"], 3) if n_ph == 2 else 0.0,
                    "paired_delta_total_time_2_minus_1": round(diff_tt["mean_diff"], 3) if n_ph == 2 else 0.0,
                }
            )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS_DIR / "pharmacy_counter_analysis.csv", index=False)
    return out_df


def run_factorial_2x2_experiments(cal: dict, uniforms: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Week 6 2^2 Full Factorial Designs around the critical c = 3 vs c = 4 staffing boundary:
      Design 1: Factor A = Doctors (3 vs 4) x Factor B = Arrival Variability (Empirical CV=0.26 vs Poisson CV=1.00)
      Design 2: Factor A = Doctors (3 vs 4) x Factor B = Consultation Duration (Baseline vs +10%)
    """
    cell_rows = []
    effect_rows = []

    designs = [
        {
            "design_name": "Staffing (3 vs 4 Docs) x Arrival Variability (Empirical vs Poisson)",
            "b_labels": ("Empirical (CV=0.26)", "Poisson (CV=1.00)"),
            "configs": {
                (-1, -1): {"n_doc": 3, "arrival_mode": "empirical", "doc_scale": 1.0},
                (+1, -1): {"n_doc": 4, "arrival_mode": "empirical", "doc_scale": 1.0},
                (-1, +1): {"n_doc": 3, "arrival_mode": "poisson", "doc_scale": 1.0},
                (+1, +1): {"n_doc": 4, "arrival_mode": "poisson", "doc_scale": 1.0},
            },
        },
        {
            "design_name": "Staffing (3 vs 4 Docs) x Consultation Time (Baseline vs +10%)",
            "b_labels": ("Baseline Service", "+10% Service"),
            "configs": {
                (-1, -1): {"n_doc": 3, "arrival_mode": "empirical", "doc_scale": 1.0},
                (+1, -1): {"n_doc": 4, "arrival_mode": "empirical", "doc_scale": 1.0},
                (-1, +1): {"n_doc": 3, "arrival_mode": "empirical", "doc_scale": 1.10},
                (+1, +1): {"n_doc": 4, "arrival_mode": "empirical", "doc_scale": 1.10},
            },
        },
    ]

    for d in designs:
        y_rep: dict[tuple[int, int], np.ndarray] = {}
        for key, cfg in d["configs"].items():
            vals = []
            for u in uniforms:
                cohort = realize_patient_cohort(
                    u,
                    cal,
                    n_patients=N_PATIENTS,
                    arrival_mode=cfg["arrival_mode"],
                    doc_service_scale=cfg["doc_scale"],
                )
                vals.append(
                    simulate_clinic(
                        cohort["arrival_times"],
                        cohort["priorities"],
                        cohort["doc_services"],
                        cohort["med_services"],
                        n_doctors=cfg["n_doc"],
                    )["summary"]["p90_doc_wait"]
                )
            y_rep[key] = np.asarray(vals, dtype=float)
            m, sd, lo, hi = mean_ci95(y_rep[key], nonneg=True)
            cell_rows.append(
                {
                    "design": d["design_name"],
                    "factor_A_doctors": "3 Doctors (-)" if key[0] == -1 else "4 Doctors (+)",
                    "factor_B_regime": d["b_labels"][0] if key[1] == -1 else d["b_labels"][1],
                    "p90_mean": round(m, 2),
                    "p90_ci_low": round(lo, 2),
                    "p90_ci_high": round(hi, 2),
                    "meets_15m_target": m <= PRIMARY_TARGET_MIN,
                }
            )

        # Replication-level Week 6 factorial contrasts (preserves CRN pairing!)
        cond_a_low_b = y_rep[(+1, -1)] - y_rep[(-1, -1)]
        cond_a_high_b = y_rep[(+1, +1)] - y_rep[(-1, +1)]
        main_a = 0.5 * (cond_a_low_b + cond_a_high_b)
        main_b = 0.5 * ((y_rep[(-1, +1)] - y_rep[(-1, -1)]) + (y_rep[(+1, +1)] - y_rep[(+1, -1)]))
        inter_ab = 0.5 * (cond_a_high_b - cond_a_low_b)

        for eff_name, arr in [
            ("Effect of 4th Doctor at B(-)", cond_a_low_b),
            ("Effect of 4th Doctor at B(+)", cond_a_high_b),
            ("Main Effect A (4 vs 3 Doctors)", main_a),
            ("Main Effect B (+ vs - Stress)", main_b),
            ("Interaction Effect A x B", inter_ab),
        ]:
            m, sd, lo, hi = mean_ci95(arr)
            _, p_val = stats.ttest_1samp(arr, 0.0)
            effect_rows.append(
                {
                    "design": d["design_name"],
                    "contrast": eff_name,
                    "estimate_min": round(m, 2),
                    "ci95_low": round(lo, 2),
                    "ci95_high": round(hi, 2),
                    "p_value": float(p_val),
                }
            )

    cells_df = pd.DataFrame(cell_rows)
    effects_df = pd.DataFrame(effect_rows)
    cells_df.to_csv(RESULTS_DIR / "factorial_2x2_cells.csv", index=False)
    effects_df.to_csv(RESULTS_DIR / "factorial_2x2_effects.csv", index=False)
    return cells_df, effects_df


def run_sensitivity_and_thresholds(cal: dict, uniforms: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run comprehensive one-way sensitivity suite and fine-grained Policy A break-even scan."""
    scenarios = [
        {"group": "Baseline", "scenario": "Baseline (Empirical, 240m peak)", "arr_mode": "empirical", "arr_scale": 1.0, "doc_scale": 1.0, "peak_min": 240.0, "fixed_day": True},
        {"group": "Arrival Rate", "scenario": "Arrival rate -10% (180 pts / ~17.2h day)", "arr_mode": "empirical", "arr_scale": 0.90, "doc_scale": 1.0, "peak_min": 240.0, "fixed_day": True},
        {"group": "Arrival Rate", "scenario": "Arrival rate +10% (220 pts / ~17.2h day)", "arr_mode": "empirical", "arr_scale": 1.10, "doc_scale": 1.0, "peak_min": 240.0, "fixed_day": True},
        {"group": "Arrival Rate (Fixed N=200)", "scenario": "Arrival rate +10% (Fixed N=200 cohort)", "arr_mode": "empirical", "arr_scale": 1.10, "doc_scale": 1.0, "peak_min": 240.0, "fixed_day": False},
        {"group": "Service Time", "scenario": "Consultation time -10%", "arr_mode": "empirical", "arr_scale": 1.0, "doc_scale": 0.90, "peak_min": 240.0, "fixed_day": True},
        {"group": "Service Time", "scenario": "Consultation time +10%", "arr_mode": "empirical", "arr_scale": 1.0, "doc_scale": 1.10, "peak_min": 240.0, "fixed_day": True},
        {"group": "Peak Duration", "scenario": "Shorter peak window (120 min)", "arr_mode": "empirical", "arr_scale": 1.0, "doc_scale": 1.0, "peak_min": 120.0, "fixed_day": True},
        {"group": "Peak Duration", "scenario": "Extended peak window (360 min)", "arr_mode": "empirical", "arr_scale": 1.0, "doc_scale": 1.0, "peak_min": 360.0, "fixed_day": True},
        {"group": "Arrival Variability", "scenario": "Piecewise-Poisson walk-ins (CV = 1.0)", "arr_mode": "poisson", "arr_scale": 1.0, "doc_scale": 1.0, "peak_min": 240.0, "fixed_day": True},
    ]

    sens_rows = []
    for sc in scenarios:
        n_pts = int(round(N_PATIENTS * sc["arr_scale"])) if sc["fixed_day"] else N_PATIENTS
        by_c: dict[int, list[float]] = {2: [], 3: [], 4: []}
        for u in uniforms:
            cohort = realize_patient_cohort(
                u,
                cal,
                n_patients=n_pts,
                peak_minutes=sc["peak_min"],
                arrival_mode=sc["arr_mode"],
                arrival_rate_scale=sc["arr_scale"],
                doc_service_scale=sc["doc_scale"],
            )
            for c in (2, 3, 4):
                by_c[c].append(
                    simulate_clinic(
                        cohort["arrival_times"],
                        cohort["priorities"],
                        cohort["doc_services"],
                        cohort["med_services"],
                        n_doctors=c,
                        peak_minutes=sc["peak_min"],
                    )["summary"]["p90_doc_wait"]
                )

        row: dict[str, object] = {"group": sc["group"], "scenario": sc["scenario"], "n_patients": n_pts}
        min_feasible = None
        for c in (2, 3, 4):
            m, _, lo, hi = mean_ci95(by_c[c], nonneg=True)
            row[f"c{c}_p90_mean"] = round(m, 2)
            row[f"c{c}_ci_low"] = round(lo, 2)
            row[f"c{c}_ci_high"] = round(hi, 2)
            if min_feasible is None and m <= PRIMARY_TARGET_MIN:
                min_feasible = f"{c} Doctors"
        row["recommended_for_15m_target"] = min_feasible or ">4 Doctors"
        sens_rows.append(row)

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(RESULTS_DIR / "sensitivity_summary.csv", index=False)

    # Fine break-even threshold scan for Policy A (3 doctors) from -5% to +10%
    pct_grid = [-5, -2.5, 0, 1, 2, 3, 4, 5, 7.5, 10]
    scan_rows = []
    for pct in pct_grid:
        scale = 1.0 + pct / 100.0
        n_pts_arr = int(round(N_PATIENTS * scale))
        p90_arr_3, p90_arr_4, p90_doc_3, p90_doc_4 = [], [], [], []
        for u in uniforms:
            c_arr = realize_patient_cohort(u, cal, n_patients=n_pts_arr, arrival_rate_scale=scale)
            p90_arr_3.append(
                simulate_clinic(c_arr["arrival_times"], c_arr["priorities"], c_arr["doc_services"], c_arr["med_services"], n_doctors=3)["summary"]["p90_doc_wait"]
            )
            p90_arr_4.append(
                simulate_clinic(c_arr["arrival_times"], c_arr["priorities"], c_arr["doc_services"], c_arr["med_services"], n_doctors=4)["summary"]["p90_doc_wait"]
            )

            c_doc = realize_patient_cohort(u, cal, n_patients=N_PATIENTS, doc_service_scale=scale)
            p90_doc_3.append(
                simulate_clinic(c_doc["arrival_times"], c_doc["priorities"], c_doc["doc_services"], c_doc["med_services"], n_doctors=3)["summary"]["p90_doc_wait"]
            )
            p90_doc_4.append(
                simulate_clinic(c_doc["arrival_times"], c_doc["priorities"], c_doc["doc_services"], c_doc["med_services"], n_doctors=4)["summary"]["p90_doc_wait"]
            )
        scan_rows.append(
            {
                "pct_change": pct,
                "arr_rate_c3_p90": round(float(np.mean(p90_arr_3)), 2),
                "arr_rate_c4_p90": round(float(np.mean(p90_arr_4)), 2),
                "doc_serv_c3_p90": round(float(np.mean(p90_doc_3)), 2),
                "doc_serv_c4_p90": round(float(np.mean(p90_doc_4)), 2),
            }
        )
    scan_df = pd.DataFrame(scan_rows)
    scan_df.to_csv(RESULTS_DIR / "threshold_scan.csv", index=False)
    return sens_df, scan_df


if __name__ == "__main__":
    if not CALIBRATED_JSON_PATH.exists():
        calibrate_from_csv()
    cal = load_calibrated_inputs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    uniforms = spawn_replication_uniforms(N_REPLICATIONS)

    main_out = run_main_experiments(cal, uniforms)
    print("=== Main Staffing Policies (50 CRN Replications) ===")
    cols = [
        "policy_label",
        "p90_doc_wait_mean",
        "p90_doc_wait_ci_low",
        "p90_doc_wait_ci_high",
        "mean_doc_wait_mean",
        "p90_prio1_wait_mean",
        "mean_med_wait_mean",
        "mean_total_time_mean",
        "doc_utilization_mean",
        "scheduled_doc_cost_hkd_mean",
        "effective_doc_cost_hkd_mean",
    ]
    print(main_out["summary"][cols].to_string(index=False))

    print("\n=== Paired CRN Differences (P90 Doctor Wait) ===")
    print(main_out["paired"][main_out["paired"]["metric"] == "p90_doc_wait"].to_string(index=False))

    pharm_df = run_pharmacy_counter_experiment(cal, uniforms)
    print("\n=== Pharmacy Counter Analysis (1 vs 2 Counters) ===")
    print(pharm_df.to_string(index=False))

    cells_df, effects_df = run_factorial_2x2_experiments(cal, uniforms)
    print("\n=== 2x2 Factorial Effects ===")
    print(effects_df.to_string(index=False))

    sens_df, scan_df = run_sensitivity_and_thresholds(cal, uniforms)
    print("\n=== Sensitivity Summary ===")
    print(sens_df[["scenario", "c2_p90_mean", "c3_p90_mean", "c4_p90_mean", "recommended_for_15m_target"]].to_string(index=False))
    print("\n=== Policy A Break-Even Threshold Scan ===")
    print(scan_df.to_string(index=False))
