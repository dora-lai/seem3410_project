"""
run_sensitivity.py  (Tasks A10 & B11, updated after Clinic feedback J5)
- A10: Run sensitivity scenarios (arrival rate, consultation time, pharmacy service time,
       and second pharmacy counter) for all 3 doctor policies using the same 50 CRN seeds;
       save replication-level output to results/results_sens.csv.
- B11:
  * B11-1: Doctor-queue sensitivity scenarios & preferred policy across 10/15/20/30-min targets
           (excluding downstream-only pharmacy scenarios that cannot alter doctor wait).
  * B11-2: Secondary analysis of optional 2nd pharmacy counter (1x vs 2x pharmacy time, paired 95% CI).
  * B11-3: Time-aligned break-even doctor wage thresholds (both 4-hr Peak-Matched and Full-Run Matched).
  * Figures: figures/fig4_sensitivity_arr_doc.png and figures/fig5_targets_and_pharmacy.png.

Run: python src/run_sensitivity.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import config
import model

RESULTS_DIR = os.path.join(config.ROOT, "results")
FIGURES_DIR = os.path.join(config.ROOT, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def mean_ci(series) -> tuple[float, float, float, float]:
    """Return (mean, half_width, ci_low, ci_high) for a 95% t-CI."""
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    m = float(np.mean(arr))
    if n < 2:
        return m, 0.0, m, m
    sd = float(np.std(arr, ddof=1))
    if sd == 0.0:
        return m, 0.0, m, m
    half = float(stats.t.ppf(0.975, df=n - 1) * sd / np.sqrt(n))
    return m, half, m - half, m + half


def make_scaled_patients(
    seed: int,
    arr_rate_mult: float = 1.0,
    doc_time_mult: float = 1.0,
    pharm_time_mult: float = 1.0,
) -> list[model.Patient]:
    """Generate N_PATIENTS with CRN preserved across scenarios and policies."""
    rng = np.random.default_rng(seed)
    n = config.N_PATIENTS

    peak_vals = np.array(list(config.ARRIVAL_GAP_PMF["peak"].keys()))
    peak_probs = np.array(list(config.ARRIVAL_GAP_PMF["peak"].values()), dtype=float)
    post_vals = np.array(list(config.ARRIVAL_GAP_PMF["post"].keys()))
    post_probs = np.array(list(config.ARRIVAL_GAP_PMF["post"].values()), dtype=float)

    arrivals = []
    t = 0.0
    for _ in range(n):
        if t < config.PEAK_MINUTES:
            gap = rng.choice(peak_vals, p=peak_probs)
        else:
            gap = rng.choice(post_vals, p=post_probs)
        t += float(gap) / arr_rate_mult
        arrivals.append(t)

    priorities = model._sample_pmf(rng, config.PRIORITY_MIX, n)
    doc_times = model._sample_pmf(rng, config.DOC_SERVICE_PMF, n) * doc_time_mult
    pharm_times = model._sample_pmf(rng, config.MED_SERVICE_PMF, n) * pharm_time_mult

    return [
        model.Patient(
            patient_id=i,
            arrival_time=float(arrivals[i]),
            priority=int(priorities[i]),
            doc_service_time=float(doc_times[i]),
            pharm_service_time=float(pharm_times[i]),
        )
        for i in range(n)
    ]


SCENARIOS = [
    # (name, category, arr_rate_mult, doc_time_mult, pharm_time_mult, n_pharmacy)
    ("Base",                   "Base",          1.00, 1.00, 1.00, 1),
    ("Arrival_-20%",           "Arrival Rate",  0.80, 1.00, 1.00, 1),
    ("Arrival_+10%",           "Arrival Rate",  1.10, 1.00, 1.00, 1),
    ("Arrival_+20%",           "Arrival Rate",  1.20, 1.00, 1.00, 1),
    ("Arrival_+30%",           "Arrival Rate",  1.30, 1.00, 1.00, 1),
    ("DocTime_-15%",           "Doctor Time",   1.00, 0.85, 1.00, 1),
    ("DocTime_+10%",           "Doctor Time",   1.00, 1.10, 1.00, 1),
    ("DocTime_+20%",           "Doctor Time",   1.00, 1.20, 1.00, 1),
    ("PharmTime_+50%",         "Pharmacy Time", 1.00, 1.00, 1.50, 1),
    ("PharmTime_+100%",        "Pharmacy Time", 1.00, 1.00, 2.00, 1),
    ("Base_2Pharm",            "2nd Pharmacy",  1.00, 1.00, 1.00, 2),
    ("PharmTime_+100%_2Pharm", "2nd Pharmacy",  1.00, 1.00, 2.00, 2),
]

DOCTOR_LEVELS = [
    ("Baseline", 2),
    ("Policy_A", 3),
    ("Policy_B", 4),
]


def run_a10_sensitivity() -> pd.DataFrame:
    """Task A10: Run all 12 sensitivity scenarios x 3 doctor levels x 50 CRN replications."""
    pharm_wage = getattr(config, "PHARMACY_WAGE_PER_HOUR", 200)
    rows = []

    for scen_name, category, arr_m, doc_m, ph_m, n_pharm in SCENARIOS:
        for r in range(config.N_REPLICATIONS):
            seed = config.SEED_BASE + r
            patients = make_scaled_patients(
                seed, arr_rate_mult=arr_m, doc_time_mult=doc_m, pharm_time_mult=ph_m
            )
            for pol_name, n_doc in DOCTOR_LEVELS:
                df, summary = model.run_policy(patients, n_doctors=n_doc, n_pharmacy=n_pharm)
                peak_df = df[df["arrival_time"] <= config.PEAK_MINUTES]

                p90_doc = float(np.percentile(df["doc_wait"], 90))
                p90_pharm = float(np.percentile(df["pharm_wait"], 90))
                doc_cost = n_doc * config.DOCTOR_WAGE_PER_HOUR * config.SHIFT_HOURS
                total_cost = doc_cost + n_pharm * pharm_wage * config.SHIFT_HOURS

                rows.append({
                    "scenario": scen_name,
                    "category": category,
                    "arr_rate_mult": arr_m,
                    "doc_time_mult": doc_m,
                    "pharm_time_mult": ph_m,
                    "n_pharmacy": n_pharm,
                    "replication": r,
                    "seed": seed,
                    "policy": pol_name,
                    "n_doctors": n_doc,
                    "p90_doc_wait": p90_doc,
                    "mean_doc_wait": float(df["doc_wait"].mean()),
                    "n_peak_patients": int(len(peak_df)),
                    "sum_doc_wait_peak_min": float(peak_df["doc_wait"].sum()),
                    "sum_doc_wait_all_min": float(df["doc_wait"].sum()),
                    "mean_pharm_wait": float(df["pharm_wait"].mean()),
                    "p90_pharm_wait": p90_pharm,
                    "mean_total_time": float(df["total_time"].mean()),
                    "makespan_hours": float(summary["makespan"]) / 60.0,
                    "doc_utilization": float(summary["doc_utilization"]),
                    "pharm_utilization": float(summary["pharm_utilization"]),
                    "doc_cost": float(doc_cost),
                    "total_cost_with_pharm": float(total_cost),
                })

    sens_df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "results_sens.csv")
    sens_df.to_csv(out_csv, index=False)
    print(f"[A10] Saved {len(sens_df)} sensitivity rows to {out_csv}")
    return sens_df


def pick_preferred_policy(scen_sub: pd.DataFrame, target: float) -> str:
    """Return lowest-cost policy (2, 3, or 4 doctors) whose mean run-level P90 <= target."""
    for pol_name, n_doc in DOCTOR_LEVELS:
        p90_mean = scen_sub[scen_sub["policy"] == pol_name]["p90_doc_wait"].mean()
        if p90_mean <= target:
            return f"{pol_name} ({n_doc} doc)"
    return "None (>4 doc needed)"


def analyze_b11(sens_df: pd.DataFrame):
    """Task B11: Upstream doctor sensitivity, 2nd pharmacy CI, and time-aligned wage thresholds."""
    targets = [10.0, 15.0, 20.0, 30.0]

    # ---- 1. Upstream Doctor-Queue Scenarios Only (Fix Point 3: no duplicate pharmacy rows) ----
    doc_scenarios = [s for s in SCENARIOS if s[1] in ("Base", "Arrival Rate", "Doctor Time")]
    summary_rows = []

    for scen_name, category, *_ in doc_scenarios:
        sub = sens_df[sens_df["scenario"] == scen_name]
        row = {"scenario": scen_name, "category": category}
        for pol_name, n_doc in DOCTOR_LEVELS:
            psub = sub[sub["policy"] == pol_name]
            m_p90, h_p90, _, _ = mean_ci(psub["p90_doc_wait"])
            row[f"P90_{n_doc}doc"] = f"{m_p90:.2f} ± {h_p90:.2f}"
            row[f"P90_{n_doc}doc_mean"] = m_p90
            row[f"P90_{n_doc}doc_halfCI"] = h_p90

        for t in targets:
            row[f"Best_T{int(t)}"] = pick_preferred_policy(sub, t)
        summary_rows.append(row)

    sens_summary = pd.DataFrame(summary_rows)
    sens_sum_csv = os.path.join(RESULTS_DIR, "sensitivity_summary_b11.csv")
    sens_summary.to_csv(sens_sum_csv, index=False)

    print("\n=== [B11-1] Doctor-Queue Sensitivity & Preferred Policy Across Targets (10/15/20/30 min) ===")
    print("(Note: Downstream pharmacy scenarios do not affect doctor waits and are reported in B11-2.)")
    cols_show = [
        "scenario", "P90_2doc", "P90_3doc", "P90_4doc",
        "Best_T10", "Best_T15", "Best_T20", "Best_T30"
    ]
    print(sens_summary[cols_show].to_string(index=False))

    # ---- 2. Secondary Analysis: Pharmacy Speed & Optional 2nd Pharmacy Counter (Paired 95% CI) ----
    pharm_pairs = [
        ("Base (1x Pharm Time)", "Base", "Base_2Pharm"),
        ("Slow Pharmacy (2x Pharm Time)", "PharmTime_+100%", "PharmTime_+100%_2Pharm"),
    ]
    pharm_rows = []
    for label, s1, s2 in pharm_pairs:
        for pol_name, n_doc in DOCTOR_LEVELS:
            d1 = sens_df[(sens_df["scenario"] == s1) & (sens_df["policy"] == pol_name)].sort_values("replication").reset_index(drop=True)
            d2 = sens_df[(sens_df["scenario"] == s2) & (sens_df["policy"] == pol_name)].sort_values("replication").reset_index(drop=True)
            assert (d1["seed"] == d2["seed"]).all()

            m_pw, h_pw, _, _ = mean_ci(d2["mean_pharm_wait"] - d1["mean_pharm_wait"])
            m_tt, h_tt, _, _ = mean_ci(d2["mean_total_time"] - d1["mean_total_time"])

            pharm_rows.append({
                "condition": label,
                "policy": f"{pol_name} ({n_doc} doc)",
                "1Pharm_MeanWait": round(d1["mean_pharm_wait"].mean(), 2),
                "2Pharm_MeanWait": round(d2["mean_pharm_wait"].mean(), 2),
                "Diff_PharmWait (2-1)": f"{m_pw:.2f} ± {h_pw:.2f}",
                "1Pharm_TotalTime": round(d1["mean_total_time"].mean(), 2),
                "2Pharm_TotalTime": round(d2["mean_total_time"].mean(), 2),
                "Diff_TotalTime (2-1)": f"{m_tt:.2f} ± {h_tt:.2f}",
                "1Pharm_Util": f"{d1['pharm_utilization'].mean()*100:.1f}%",
                "2Pharm_Util": f"{d2['pharm_utilization'].mean()*100:.1f}%",
            })

    pharm_df = pd.DataFrame(pharm_rows)
    pharm_csv = os.path.join(RESULTS_DIR, "sensitivity_pharmacy_b11.csv")
    pharm_df.to_csv(pharm_csv, index=False)
    print("\n=== [B11-2] Secondary Analysis: Optional 2nd Pharmacy Counter (Paired 95% CI) ===")
    print(pharm_df.to_string(index=False))

    # ---- 3. Time-Aligned Break-Even Doctor Wage Thresholds (Fix Point 2 & Point 4) ----
    base_sub = sens_df[sens_df["scenario"] == "Base"]
    b2 = base_sub[base_sub["policy"] == "Baseline"]
    b3 = base_sub[base_sub["policy"] == "Policy_A"]
    b4 = base_sub[base_sub["policy"] == "Policy_B"]

    # (a) Peak-Matched (4.0-hr shift cost vs. waiting saved by peak-window arrivals t <= 240 min)
    peak_hrs_saved_2to3 = (b2["sum_doc_wait_peak_min"].mean() - b3["sum_doc_wait_peak_min"].mean()) / 60.0
    peak_hrs_saved_3to4 = (b3["sum_doc_wait_peak_min"].mean() - b4["sum_doc_wait_peak_min"].mean()) / 60.0
    avg_n_peak = b3["n_peak_patients"].mean()

    # (b) Full-Run Matched (common full clinic operating horizon L_full vs. all-200-patient waiting saved)
    # Use Policy_A's mean makespan (~17.5 hrs) as the common full-run operating window for all policies
    full_run_hours = b3["makespan_hours"].mean()
    full_hrs_saved_2to3 = (b2["sum_doc_wait_all_min"].mean() - b3["sum_doc_wait_all_min"].mean()) / 60.0
    full_hrs_saved_3to4 = (b3["sum_doc_wait_all_min"].mean() - b4["sum_doc_wait_all_min"].mean()) / 60.0

    # Break-even wage w* = v * (patient-hours saved) / (matched doctor hours paid)
    breakeven_rows = []
    for v_per_hr in [30, 60, 120, 180, 300]:  # Assumed patient waiting-time value (HKD/patient-hr)
        w_peak_2to3 = (v_per_hr * peak_hrs_saved_2to3) / config.SHIFT_HOURS
        w_peak_3to4 = (v_per_hr * peak_hrs_saved_3to4) / config.SHIFT_HOURS
        w_full_2to3 = (v_per_hr * full_hrs_saved_2to3) / full_run_hours
        w_full_3to4 = (v_per_hr * full_hrs_saved_3to4) / full_run_hours

        breakeven_rows.append({
            "Wait_Value_HKD_hr": v_per_hr,
            "PeakMatched_w*_2to3": round(w_peak_2to3, 1),
            "PeakMatched_w*_3to4": round(w_peak_3to4, 1),
            "FullRunMatched_w*_2to3": round(w_full_2to3, 1),
            "FullRunMatched_w*_3to4": round(w_full_3to4, 1),
            "Pref_PeakMatched_at_W500": (
                "Policy_B (4 doc)" if w_peak_3to4 >= config.DOCTOR_WAGE_PER_HOUR
                else ("Policy_A (3 doc)" if w_peak_2to3 >= config.DOCTOR_WAGE_PER_HOUR else "Baseline (2 doc)")
            ),
            "Pref_FullMatched_at_W500": (
                "Policy_B (4 doc)" if w_full_3to4 >= config.DOCTOR_WAGE_PER_HOUR
                else ("Policy_A (3 doc)" if w_full_2to3 >= config.DOCTOR_WAGE_PER_HOUR else "Baseline (2 doc)")
            ),
        })

    be_df = pd.DataFrame(breakeven_rows)
    be_csv = os.path.join(RESULTS_DIR, "sensitivity_wage_breakeven_b11.csv")
    be_df.to_csv(be_csv, index=False)

    print("\n=== [B11-3] Time-Aligned Break-Even Doctor Wage Thresholds (Assumed Wait Values) ===")
    print(f"Peak-Matched basis    : {config.SHIFT_HOURS:.1f}-hr shift vs. {avg_n_peak:.1f} peak arrivals "
          f"(2->3 saves {peak_hrs_saved_2to3:.1f} pat-hrs; 3->4 saves {peak_hrs_saved_3to4:.2f} pat-hrs)")
    print(f"Full-Run Matched basis: {full_run_hours:.2f}-hr clinic run vs. all 200 patients "
          f"(2->3 saves {full_hrs_saved_2to3:.1f} pat-hrs; 3->4 saves {full_hrs_saved_3to4:.2f} pat-hrs)")
    print("(Note: Under a hard 15-min P90 target with equal shift lengths, Policy A is lowest-cost for any positive doctor wage.)")
    print(be_df.to_string(index=False))

    # ---- 4. Generate Section 6 Figures ----
    make_sensitivity_figures(sens_summary, sens_df)


def make_sensitivity_figures(sens_summary: pd.DataFrame, sens_df: pd.DataFrame):
    """Generate clean figures for Report Section 6."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
    })

    # ---- Figure 4: Arrival Rate & Doctor Consultation Time Sensitivity ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    arr_scens = ["Arrival_-20%", "Base", "Arrival_+10%", "Arrival_+20%", "Arrival_+30%"]
    arr_x = [-20, 0, 10, 20, 30]
    sub_arr = sens_summary.set_index("scenario").loc[arr_scens]

    styles = [
        ("Baseline (2 Doc)", "P90_2doc", "#d95f02", "s-"),
        ("Policy A (3 Doc)", "P90_3doc", "#1b9e77", "o-"),
        ("Policy B (4 Doc)", "P90_4doc", "#7570b3", "^-"),
    ]
    for label, prefix, color, fmt in styles:
        y = sub_arr[f"{prefix}_mean"].values
        ci = sub_arr[f"{prefix}_halfCI"].values
        ax1.errorbar(arr_x, y, yerr=ci, fmt=fmt, color=color, capsize=4, linewidth=1.8, markersize=6, label=label)

    ax1.axhline(15.0, color="red", linestyle="--", linewidth=1.3, label="15-min Target")
    ax1.axhline(10.0, color="gray", linestyle=":", linewidth=1.1, label="10-min Target")
    ax1.set_xlabel("Arrival Rate Change vs. Baseline (%)")
    ax1.set_ylabel("Mean Run-Level P90 Doctor Wait (min)")
    ax1.set_title("(a) Sensitivity to Patient Arrival Rate (95% CI)")
    ax1.set_xticks(arr_x)
    ax1.set_ylim(-10, max(sub_arr["P90_3doc_mean"].max() * 1.4, 140))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8.5)

    doc_scens = ["DocTime_-15%", "Base", "DocTime_+10%", "DocTime_+20%"]
    doc_x = [-15, 0, 10, 20]
    sub_doc = sens_summary.set_index("scenario").loc[doc_scens]

    for label, prefix, color, fmt in styles:
        y = sub_doc[f"{prefix}_mean"].values
        ci = sub_doc[f"{prefix}_halfCI"].values
        ax2.errorbar(doc_x, y, yerr=ci, fmt=fmt, color=color, capsize=4, linewidth=1.8, markersize=6, label=label)

    ax2.axhline(15.0, color="red", linestyle="--", linewidth=1.3, label="15-min Target")
    ax2.axhline(10.0, color="gray", linestyle=":", linewidth=1.1, label="10-min Target")
    ax2.set_xlabel("Doctor Service Time Change vs. Baseline (%)")
    ax2.set_ylabel("Mean Run-Level P90 Doctor Wait (min)")
    ax2.set_title("(b) Sensitivity to Doctor Consultation Time (95% CI)")
    ax2.set_xticks(doc_x)
    ax2.set_ylim(-10, max(sub_doc["P90_3doc_mean"].max() * 1.4, 140))
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    out4 = os.path.join(FIGURES_DIR, "fig4_sensitivity_arr_doc.png")
    plt.savefig(out4, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out4}")

    # ---- Figure 5: Target Sensitivity & 2nd Pharmacy Counter ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    base_sub = sens_df[sens_df["scenario"] == "Base"]
    pols = ["Policy_A", "Policy_B"]
    x_pos = np.arange(len(pols))
    means = [base_sub[base_sub["policy"] == p]["p90_doc_wait"].mean() for p in pols]
    cis = [mean_ci(base_sub[base_sub["policy"] == p]["p90_doc_wait"])[1] for p in pols]

    bars = ax1.bar(x_pos, means, yerr=cis, width=0.45, capsize=5,
                   color=["#1b9e77", "#7570b3"], edgecolor="black", alpha=0.88)
    for bar, m, ci in zip(bars, means, cis):
        txt = f"{m:.2f} ± {ci:.2f}" if ci > 0 else "0.00"
        ax1.text(bar.get_x() + bar.get_width() / 2, m + ci + 0.8, txt,
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    for t, col, ls in [(10, "#e74c3c", ":"), (15, "#c0392b", "--"), (20, "#2980b9", "-."), (30, "#2c3e50", ":")]:
        ax1.axhline(t, color=col, linestyle=ls, linewidth=1.3, label=f"Target = {t} min")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(["Policy A\n(3 Doctors, HKD 6,000)", "Policy B\n(4 Doctors, HKD 8,000)"])
    ax1.set_ylabel("Mean Run-Level P90 Doctor Wait (min)")
    ax1.set_title("(a) Target Sensitivity: 10 / 15 / 20 / 30 min (Base)")
    ax1.set_ylim(0, 34)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5)

    groups = ["Base Pharmacy\n(1x Service Time)", "Slow Pharmacy\n(2x Service Time)"]
    sc_1p = ["Base", "PharmTime_+100%"]
    sc_2p = ["Base_2Pharm", "PharmTime_+100%_2Pharm"]

    m_1p, ci_1p, m_2p, ci_2p = [], [], [], []
    for s1, s2 in zip(sc_1p, sc_2p):
        m1, c1, _, _ = mean_ci(sens_df[(sens_df["scenario"] == s1) & (sens_df["policy"] == "Policy_A")]["mean_pharm_wait"])
        m2, c2, _, _ = mean_ci(sens_df[(sens_df["scenario"] == s2) & (sens_df["policy"] == "Policy_A")]["mean_pharm_wait"])
        m_1p.append(m1); ci_1p.append(c1)
        m_2p.append(m2); ci_2p.append(c2)

    xg = np.arange(len(groups))
    w = 0.34
    b1 = ax2.bar(xg - w / 2, m_1p, w, yerr=ci_1p, capsize=4, label="1 Pharmacy Counter",
                 color="#9b59b6", edgecolor="black", alpha=0.88)
    b2 = ax2.bar(xg + w / 2, m_2p, w, yerr=ci_2p, capsize=4, label="2 Pharmacy Counters",
                 color="#3498db", edgecolor="black", alpha=0.88)

    for bars_g, vals, c_vals in [(b1, m_1p, ci_1p), (b2, m_2p, ci_2p)]:
        for bar, m, c in zip(bars_g, vals, c_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, m + c + 0.4, f"{m:.2f}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xticks(xg)
    ax2.set_xticklabels(groups)
    ax2.set_ylabel("Mean Pharmacy Wait Time (min)")
    ax2.set_title("(b) 1 vs. 2 Pharmacy Counters (Policy A: 3 Doctors)")
    ax2.set_ylim(0, max(m_1p) * 1.28)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    out5 = os.path.join(FIGURES_DIR, "fig5_targets_and_pharmacy.png")
    plt.savefig(out5, bbox_inches="tight")
    plt.close()
    print(f"Saved {out5}")


if __name__ == "__main__":
    sens_df = run_a10_sensitivity()
    analyze_b11(sens_df)