"""
run_main.py  (Tasks A8, B9, A9)
- A8: Run 50 CRN replications for Baseline (2 doc), Policy_A (3 doc), Policy_B (4 doc);
      save run-level metrics to results/results_main.csv.
- B9: Compute all required metrics per replication: run-level P90, mean wait,
      waits by priority class (1, 2, 3), peak-arrival waits, pharmacy wait,
      total hospital time, doctor & pharmacy utilization, and staffing cost.
- A9: Compute summary statistics and paired differences vs Baseline with 95% t-CIs;
      save to results/summary_main.csv and results/paired_ci_main.csv.

Run: python src/run_main.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import config
import model


def mean_ci(series: pd.Series) -> tuple[float, float, float, float]:
    """Return (mean, half_width, ci_low, ci_high) for a 95% t confidence interval."""
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0, mean, mean
    sd = float(np.std(arr, ddof=1))
    if sd == 0.0:
        return mean, 0.0, mean, mean
    half = float(stats.t.ppf(0.975, df=n - 1) * sd / np.sqrt(n))
    return mean, half, mean - half, mean + half


def evaluate_replication(rep_idx: int, policy_name: str, params: dict, base_patients: list) -> dict:
    """Run one replication for one policy and compute all B9 metrics."""
    n_doc = params["n_doctors"]
    n_pharm = params["n_pharmacy"]

    df, summary = model.run_policy(base_patients, n_doctors=n_doc, n_pharmacy=n_pharm)

    # Priority subsets (3 = highest, 1 = lowest)
    p1 = df[df["priority"] == 1]["doc_wait"]
    p2 = df[df["priority"] == 2]["doc_wait"]
    p3 = df[df["priority"] == 3]["doc_wait"]

    # Peak arrival subset (arrived during first 240 minutes)
    peak_df = df[df["arrival_time"] <= config.PEAK_MINUTES]
    offpeak_df = df[df["arrival_time"] > config.PEAK_MINUTES]

    # Staffing cost (fixed shift length across all policies)
    cost = n_doc * config.DOCTOR_WAGE_PER_HOUR * config.SHIFT_HOURS

    p90_doc = float(np.percentile(df["doc_wait"], 90))

    return {
        "replication": rep_idx,
        "seed": config.SEED_BASE + rep_idx,
        "policy": policy_name,
        "n_doctors": n_doc,
        "n_pharmacy": n_pharm,
        # Primary measure
        "p90_doc_wait": p90_doc,
        "meets_target_15": int(p90_doc <= config.TARGET_P90),
        # Overall wait & time measures
        "mean_doc_wait": float(df["doc_wait"].mean()),
        "mean_pharm_wait": float(df["pharm_wait"].mean()),
        "p90_pharm_wait": float(np.percentile(df["pharm_wait"], 90)),
        "mean_total_time": float(df["total_time"].mean()),
        # Waits by priority class
        "mean_doc_wait_p1": float(p1.mean()) if len(p1) > 0 else np.nan,
        "mean_doc_wait_p2": float(p2.mean()) if len(p2) > 0 else np.nan,
        "mean_doc_wait_p3": float(p3.mean()) if len(p3) > 0 else np.nan,
        "p90_doc_wait_p1": float(np.percentile(p1, 90)) if len(p1) > 0 else np.nan,
        "p90_doc_wait_p2": float(np.percentile(p2, 90)) if len(p2) > 0 else np.nan,
        "p90_doc_wait_p3": float(np.percentile(p3, 90)) if len(p3) > 0 else np.nan,
        # Waits for peak vs post-peak arrivals
        "mean_doc_wait_peak": float(peak_df["doc_wait"].mean()) if len(peak_df) > 0 else np.nan,
        "p90_doc_wait_peak": float(np.percentile(peak_df["doc_wait"], 90)) if len(peak_df) > 0 else np.nan,
        "mean_doc_wait_post": float(offpeak_df["doc_wait"].mean()) if len(offpeak_df) > 0 else np.nan,
        # Utilization & cost
        "doc_utilization": float(summary["doc_utilization"]),
        "pharm_utilization": float(summary["pharm_utilization"]),
        "makespan": float(summary["makespan"]),
        "staffing_cost": float(cost),
    }


def main():
    results_dir = os.path.join(config.ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ---- A8 & B9: Run 50 CRN replications across all 3 policies ----
    all_rows = []
    for r in range(config.N_REPLICATIONS):
        seed = config.SEED_BASE + r
        # Generate patients ONCE per seed and reuse across policies (CRN)
        base_patients = model.make_patients(seed)

        for policy_name, params in config.POLICIES.items():
            row = evaluate_replication(r, policy_name, params, base_patients)
            all_rows.append(row)

    results_df = pd.DataFrame(all_rows)
    main_csv = os.path.join(results_dir, "results_main.csv")
    results_df.to_csv(main_csv, index=False)
    print(f"[A8/B9] Saved {len(results_df)} replication rows to {main_csv}")

    # ---- Policy-level Summary Table (Mean + 95% CI) ----
    metrics = [
        "p90_doc_wait",
        "mean_doc_wait",
        "mean_doc_wait_p1",
        "mean_doc_wait_p2",
        "mean_doc_wait_p3",
        "p90_doc_wait_p1",
        "p90_doc_wait_p2",
        "p90_doc_wait_p3",
        "mean_doc_wait_peak",
        "p90_doc_wait_peak",
        "mean_pharm_wait",
        "p90_pharm_wait",
        "mean_total_time",
        "doc_utilization",
        "pharm_utilization",
        "staffing_cost",
    ]

    summary_rows = []
    for policy_name in config.POLICIES.keys():
        sub = results_df[results_df["policy"] == policy_name]
        row = {
            "policy": policy_name,
            "n_doctors": int(sub["n_doctors"].iloc[0]),
            "pct_runs_meet_15min": float(sub["meets_target_15"].mean() * 100.0),
        }
        for m in metrics:
            mean, half, low, high = mean_ci(sub[m])
            row[f"{m}_mean"] = mean
            row[f"{m}_halfCI"] = half
            row[f"{m}_ci95"] = f"[{low:.2f}, {high:.2f}]"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(results_dir, "summary_main.csv")
    summary_df.to_csv(summary_csv, index=False)

    # ---- A9: Paired Differences vs Baseline (Policy - Baseline) with 95% CI ----
    base_df = results_df[results_df["policy"] == "Baseline"].sort_values("replication").reset_index(drop=True)
    paired_metrics = [
        "p90_doc_wait",
        "mean_doc_wait",
        "mean_doc_wait_p1",
        "mean_doc_wait_p2",
        "mean_doc_wait_p3",
        "mean_doc_wait_peak",
        "p90_doc_wait_peak",
        "mean_pharm_wait",
        "mean_total_time",
        "doc_utilization",
        "pharm_utilization",
        "staffing_cost",
    ]

    paired_rows = []
    for alt_policy in ["Policy_A", "Policy_B"]:
        alt_df = results_df[results_df["policy"] == alt_policy].sort_values("replication").reset_index(drop=True)
        assert (base_df["seed"] == alt_df["seed"]).all(), "CRN seed mismatch!"

        for m in paired_metrics:
            diff = alt_df[m] - base_df[m]  # Negative = reduction in wait/time
            mean_d, half_d, low_d, high_d = mean_ci(diff)
            paired_rows.append({
                "comparison": f"{alt_policy} - Baseline",
                "metric": m,
                "mean_diff": round(mean_d, 4),
                "half_CI_95": round(half_d, 4),
                "ci_low": round(low_d, 4),
                "ci_high": round(high_d, 4),
            })

    paired_df = pd.DataFrame(paired_rows)
    paired_csv = os.path.join(results_dir, "paired_ci_main.csv")
    paired_df.to_csv(paired_csv, index=False)
    print(f"[A9] Saved paired difference CI table to {paired_csv}\n")

    # Print concise terminal overview
    print("=== Summary by Policy (Mean ± 95% Half-Width) ===")
    display_cols = [
        "policy", "n_doctors", "p90_doc_wait_mean", "p90_doc_wait_halfCI",
        "pct_runs_meet_15min", "mean_doc_wait_mean", "mean_pharm_wait_mean",
        "mean_total_time_mean", "doc_utilization_mean", "pharm_utilization_mean", "staffing_cost_mean"
    ]
    print(summary_df[display_cols].round(2).to_string(index=False))

    print("\n=== Paired Differences vs Baseline (Policy - Baseline, 95% CI) ===")
    print(paired_df.to_string(index=False))


if __name__ == "__main__":
    main()