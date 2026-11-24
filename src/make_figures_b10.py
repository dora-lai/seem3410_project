"""
make_figures_b10.py  (Task B10)
Generate publication-ready figures (with 95% CIs) and formatted tables for Report Section 5:
  1. figures/fig1_policy_p90_cost.png     : P90 doctor wait (95% CI) vs staffing cost trade-off
  2. figures/fig2_waits_by_class_peak.png : Mean doctor wait by priority class (P1, P2, P3) & peak arrivals
  3. figures/fig3_utilization_time.png    : Doctor vs pharmacy utilization & total hospital time breakdown
  4. results/table_section5.csv           : Clean report-ready summary table with 95% CIs

Run after run_main.py:  python src/make_figures_b10.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

RESULTS_DIR = os.path.join(config.ROOT, "results")
FIGURES_DIR = os.path.join(config.ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

summary_df = pd.read_csv(os.path.join(RESULTS_DIR, "summary_main.csv"))
paired_df = pd.read_csv(os.path.join(RESULTS_DIR, "paired_ci_main.csv"))

# Consistent ordering and labels
POLICY_ORDER = ["Baseline", "Policy_A", "Policy_B"]
LABELS = ["Baseline\n(2 Doctors)", "Policy A\n(3 Doctors)", "Policy B\n(4 Doctors)"]
summary_df = summary_df.set_index("policy").loc[POLICY_ORDER].reset_index()

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})


def make_fig1_p90_and_cost():
    """Figure 1: P90 Doctor Wait (with 95% CI & 15-min target) and Cost-Wait Trade-off."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    x = np.arange(len(POLICY_ORDER))
    p90_mean = summary_df["p90_doc_wait_mean"].values
    p90_ci = summary_df["p90_doc_wait_halfCI"].values
    costs = summary_df["staffing_cost_mean"].values
    colors = ["#d95f02", "#1b9e77", "#7570b3"]

    # Panel A: Bar chart of P90 doctor wait with 95% CI
    bars = ax1.bar(x, p90_mean, yerr=p90_ci, capsize=5, color=colors, edgecolor="black", alpha=0.88)
    ax1.axhline(config.TARGET_P90, color="red", linestyle="--", linewidth=1.4, label=f"Target ({config.TARGET_P90:.0f} min)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS)
    ax1.set_ylabel("Mean Run-Level P90 Doctor Wait (min)")
    ax1.set_title("(a) P90 Doctor Waiting Time (95% CI)")
    ax1.set_ylim(0, max(p90_mean + p90_ci) * 1.15)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    for bar, m, ci in zip(bars, p90_mean, p90_ci):
        label_txt = f"{m:.1f} ± {ci:.1f}" if ci > 0 else f"{m:.2f}"
        ax1.text(bar.get_x() + bar.get_width() / 2, m + ci + 8, label_txt,
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: Cost vs P90 trade-off scatter/line
    ax2.errorbar(costs, p90_mean, yerr=p90_ci, fmt="o-", color="#2c3e50",
                 ecolor="#2c3e50", capsize=5, linewidth=1.8, markersize=7)
    ax2.axhline(config.TARGET_P90, color="red", linestyle="--", linewidth=1.4, label=f"Target ({config.TARGET_P90:.0f} min)")

    offsets = [(0, 14), (0, 16), (-18, 16)]
    short_names = ["Baseline (2 Doc)", "Policy A (3 Doc)", "Policy B (4 Doc)"]
    for (c, m, txt, (dx, dy)) in zip(costs, p90_mean, short_names, offsets):
        ax2.annotate(f"{txt}\n({config.CURRENCY} {c:,.0f})",
                     xy=(c, m), xytext=(dx, dy), textcoords="offset points",
                     ha="center", fontsize=8.5, fontweight="bold")

    ax2.set_xlabel(f"Doctor Staffing Cost per Shift ({config.CURRENCY})")
    ax2.set_ylabel("Mean Run-Level P90 Doctor Wait (min)")
    ax2.set_title("(b) Cost vs. P90 Waiting Time Trade-Off")
    ax2.set_xlim(min(costs) - 700, max(costs) + 900)
    ax2.set_ylim(-15, max(p90_mean + p90_ci) * 1.18)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig1_policy_p90_cost.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def make_fig2_waits_by_class_and_peak():
    """Figure 2: Doctor waits by priority class (P1, P2, P3) and peak-arrival patients."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    x = np.arange(len(POLICY_ORDER))
    width = 0.25

    # Panel A: Mean wait by priority class (P1, P2, P3)
    p_cols = [
        ("Priority 1 (Lowest)", "mean_doc_wait_p1", "#e74c3c"),
        ("Priority 2 (Medium)", "mean_doc_wait_p2", "#f39c12"),
        ("Priority 3 (Highest)", "mean_doc_wait_p3", "#27ae60"),
    ]
    for i, (label, col, color) in enumerate(p_cols):
        means = summary_df[f"{col}_mean"].values
        cis = summary_df[f"{col}_halfCI"].values
        bars = ax1.bar(x + (i - 1) * width, means, width, yerr=cis, capsize=3,
                       label=label, color=color, edgecolor="black", alpha=0.88)
        for bar, m, ci in zip(bars, means, cis):
            ax1.text(bar.get_x() + bar.get_width() / 2, m + ci + 4, f"{m:.1f}",
                     ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS)
    ax1.set_ylabel("Mean Doctor Wait Time (min)")
    ax1.set_title("(a) Mean Doctor Wait by Priority Class (95% CI)")
    ax1.set_ylim(0, summary_df["mean_doc_wait_p1_mean"].max() * 1.18)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Panel B: Peak-arrival patients (Mean & P90 doctor wait)
    w2 = 0.34
    peak_mean = summary_df["mean_doc_wait_peak_mean"].values
    peak_mean_ci = summary_df["mean_doc_wait_peak_halfCI"].values
    peak_p90 = summary_df["p90_doc_wait_peak_mean"].values
    peak_p90_ci = summary_df["p90_doc_wait_peak_halfCI"].values

    b1 = ax2.bar(x - w2 / 2, peak_mean, w2, yerr=peak_mean_ci, capsize=4,
                 label="Peak Mean Wait", color="#3498db", edgecolor="black", alpha=0.88)
    b2 = ax2.bar(x + w2 / 2, peak_p90, w2, yerr=peak_p90_ci, capsize=4,
                 label="Peak P90 Wait", color="#1f4e79", edgecolor="black", alpha=0.88)

    for bars, vals, cis in [(b1, peak_mean, peak_mean_ci), (b2, peak_p90, peak_p90_ci)]:
        for bar, m, ci in zip(bars, vals, cis):
            ax2.text(bar.get_x() + bar.get_width() / 2, m + ci + 4, f"{m:.1f}",
                     ha="center", va="bottom", fontsize=8)

    ax2.axhline(config.TARGET_P90, color="red", linestyle="--", linewidth=1.2, label="15-min Target")
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS)
    ax2.set_ylabel("Doctor Wait Time for Peak Arrivals (min)")
    ax2.set_title("(b) Doctor Wait for Peak-Window Arrivals (t <= 240 min)")
    ax2.set_ylim(0, max(peak_p90 + peak_p90_ci) * 1.18)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig2_waits_by_class_peak.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def make_fig3_utilization_and_pharmacy():
    """Figure 3: Resource Utilization (Doctor & Pharmacy) and Mean Pharmacy Wait."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    x = np.arange(len(POLICY_ORDER))
    w = 0.34

    # Panel A: Doctor & Pharmacy utilization (%)
    doc_u = summary_df["doc_utilization_mean"].values * 100
    doc_u_ci = summary_df["doc_utilization_halfCI"].values * 100
    ph_u = summary_df["pharm_utilization_mean"].values * 100
    ph_u_ci = summary_df["pharm_utilization_halfCI"].values * 100

    b1 = ax1.bar(x - w / 2, doc_u, w, yerr=doc_u_ci, capsize=4,
                 label="Doctor Utilization", color="#2ecc71", edgecolor="black", alpha=0.88)
    b2 = ax1.bar(x + w / 2, ph_u, w, yerr=ph_u_ci, capsize=4,
                 label="Pharmacy Utilization", color="#9b59b6", edgecolor="black", alpha=0.88)

    for bars, vals, cis in [(b1, doc_u, doc_u_ci), (b2, ph_u, ph_u_ci)]:
        for bar, m, ci in zip(bars, vals, cis):
            ax1.text(bar.get_x() + bar.get_width() / 2, m + ci + 1.5, f"{m:.1f}%",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS)
    ax1.set_ylabel("Utilization (%)")
    ax1.set_title("(a) Doctor and Pharmacy Utilization (95% CI)")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Panel B: Mean Pharmacy Wait (shows downstream congestion shift)
    ph_w = summary_df["mean_pharm_wait_mean"].values
    ph_w_ci = summary_df["mean_pharm_wait_halfCI"].values
    bars = ax2.bar(x, ph_w, width=0.5, yerr=ph_w_ci, capsize=5,
                   color="#9b59b6", edgecolor="black", alpha=0.85)
    for bar, m, ci in zip(bars, ph_w, ph_w_ci):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + ci + 0.02, f"{m:.2f} ± {ci:.2f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS)
    ax2.set_ylabel("Mean Pharmacy Wait Time (min)")
    ax2.set_title("(b) Downstream Pharmacy Waiting Time (95% CI)")
    ax2.set_ylim(0, max(ph_w + ph_w_ci) * 1.25)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig3_utilization_pharmacy.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def save_report_table():
    """Save clean formatted table for Section 5."""
    rows = []
    for _, r in summary_df.iterrows():
        rows.append({
            "Policy": f"{r['policy']} ({int(r['n_doctors'])} doc)",
            "P90 Doc Wait (min)": f"{r['p90_doc_wait_mean']:.2f} ± {r['p90_doc_wait_halfCI']:.2f}",
            "% Runs <= 15m": f"{r['pct_runs_meet_15min']:.0f}%",
            "Mean Doc Wait (min)": f"{r['mean_doc_wait_mean']:.2f} ± {r['mean_doc_wait_halfCI']:.2f}",
            "P1 Mean Wait (min)": f"{r['mean_doc_wait_p1_mean']:.2f} ± {r['mean_doc_wait_p1_halfCI']:.2f}",
            "P2 Mean Wait (min)": f"{r['mean_doc_wait_p2_mean']:.2f} ± {r['mean_doc_wait_p2_halfCI']:.2f}",
            "P3 Mean Wait (min)": f"{r['mean_doc_wait_p3_mean']:.2f} ± {r['mean_doc_wait_p3_halfCI']:.2f}",
            "Peak P90 Wait (min)": f"{r['p90_doc_wait_peak_mean']:.2f} ± {r['p90_doc_wait_peak_halfCI']:.2f}",
            "Mean Pharm Wait (min)": f"{r['mean_pharm_wait_mean']:.2f} ± {r['mean_pharm_wait_halfCI']:.2f}",
            "Total Time (min)": f"{r['mean_total_time_mean']:.2f} ± {r['mean_total_time_halfCI']:.2f}",
            "Doc Util": f"{r['doc_utilization_mean']*100:.1f}%",
            "Pharm Util": f"{r['pharm_utilization_mean']*100:.1f}%",
            f"Cost ({config.CURRENCY})": f"{r['staffing_cost_mean']:,.0f}",
        })
    tbl = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "table_section5.csv")
    tbl.to_csv(out, index=False)
    print(f"Saved {out}\n")
    print("=== Report Section 5 Table (Mean ± 95% Half-Width) ===")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    make_fig1_p90_and_cost()
    make_fig2_waits_by_class_and_peak()
    make_fig3_utilization_and_pharmacy()
    save_report_table()