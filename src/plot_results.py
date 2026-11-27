"""
Plot figures for the outpatient clinic simulation study.

Outputs saved to figures/:
  1. fig1_clinic_flow.png  - Two-stage clinic process flow diagram
  2. fig2_validation.png   - Interarrival PMFs and 6-scenario validation
  3. fig3_main_results.png - Daily queue length trajectories and priority P90 waits
  4. fig4_sensitivity.png  - Parameter sensitivity and 2x2 arrival-variability interaction
"""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from config import CALIBRATED_JSON_PATH, FIGURES_DIR, PRIMARY_TARGET_MIN, RESULTS_DIR

COLORS = {
    "navy": "#1F3A5F",
    "slate": "#52606D",
    "teal": "#2A7B76",
    "terracotta": "#B85D43",
    "muted_blue": "#486581",
    "border": "#334E68",
    "light_fill": "#F8FAFC",
    "grid": "#D9E2EC",
}


def _apply_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.8,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlepad": 9.0,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#486581",
            "axes.linewidth": 0.85,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.75,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def make_fig1_clinic_flow():
    fig = plt.figure(figsize=(11.0, 2.65))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0, 2.65)
    ax.axis("off")

    # Subtle dashed system boundary around the two clinic stages
    boundary = FancyBboxPatch(
        (1.96, 0.26),
        7.08,
        2.12,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.0,
        linestyle="--",
        edgecolor="#829AB1",
        facecolor="#F8FAFC",
    )
    ax.add_patch(boundary)
    ax.text(
        5.50,
        2.19,
        "Two-Stage Outpatient Clinic Simulation Boundary",
        ha="center",
        va="center",
        fontsize=8.8,
        color=COLORS["slate"],
        fontweight="bold",
    )

    def add_node(x, y, w, h, title, subtitle, fill="#FFFFFF", edge=COLORS["border"]):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.05,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2.0,
            y + h * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#102A43",
        )
        ax.text(
            x + w / 2.0,
            y + h * 0.31,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.9,
            color="#334E68",
            linespacing=1.30,
        )

    y0 = 0.52
    bh = 1.24
    add_node(0.12, y0, 1.54, bh, "Patient Arrivals", "200 patients / day\nPeak (<240m) & Off-peak", fill="#F8FAFC", edge=COLORS["slate"])
    add_node(2.12, y0, 1.52, bh, "Waiting Room Q1", "Priority 3 > 2 > 1\nFCFS within class", fill="#FFFFFF", edge=COLORS["navy"])
    add_node(3.98, y0, 1.52, bh, "Stage 1: Doctors", "c in {2, 3, 4} doctors\nMean 13.4 min", fill="#F0F4F8", edge=COLORS["navy"])
    add_node(5.84, y0, 1.44, bh, "Pharm Queue Q2", "FCFS discipline\nDirect transfer", fill="#FFFFFF", edge=COLORS["muted_blue"])
    add_node(7.62, y0, 1.28, bh, "Stage 2: Pharm", "m = 1 counter\nMean 3.0 min", fill="#F0F4F8", edge=COLORS["muted_blue"])
    add_node(9.36, y0, 1.50, bh, "Clinic Exit", "Record waits &\nshift completion", fill="#F8FAFC", edge=COLORS["slate"])

    yc = y0 + bh / 2.0
    arrows = [
        (1.72, 2.06),
        (3.70, 3.92),
        (5.56, 5.78),
        (7.34, 7.56),
        (8.96, 9.30),
    ]
    for x_start, x_end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                (x_start, yc),
                (x_end, yc),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.15,
                color=COLORS["navy"],
            )
        )

    fig.savefig(FIGURES_DIR / "fig1_clinic_flow.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_fig2_validation():
    with open(CALIBRATED_JSON_PATH, "r", encoding="utf-8") as f:
        cal = json.load(f)
    oos = pd.read_csv(RESULTS_DIR / "validation_out_of_sample.csv")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 3.65),
        gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.28},
    )

    # Panel (a): Peak vs Off-peak interarrival PMF
    ax = axes[0]
    gaps = [3, 4, 5, 6, 7]
    pk_p = [cal["interarrival_peak"]["probs"][str(g)] for g in gaps]
    of_p = [cal["interarrival_offpeak"]["probs"][str(g)] for g in gaps]
    x = np.arange(len(gaps))
    w = 0.35

    ax.bar(
        x - w / 2.0,
        pk_p,
        width=w,
        color=COLORS["terracotta"],
        alpha=0.90,
        label="Peak (t < 240 min, mean 4.30 min)",
    )
    ax.bar(
        x + w / 2.0,
        of_p,
        width=w,
        color=COLORS["navy"],
        alpha=0.88,
        label="Off-peak (t >= 240 min, mean 5.49 min)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}" for g in gaps])
    ax.set_xlabel("Interarrival Gap (minutes)")
    ax.set_ylabel("Empirical Probability")
    ax.set_ylim(0, 0.55)
    ax.set_title("(a) Peak vs. Off-Peak Interarrival PMF")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

    # Panel (b): Historical vs calibrated DES across all 6 scenarios
    ax2 = axes[1]
    order = [
        "Staffing_1Doc",
        "Peak_360Min",
        "Baseline_2Doc_240Peak",
        "Peak_120Min",
        "Staffing_3Doc",
        "Staffing_4Doc",
    ]
    labels = [
        "1 Doc\n(240m)",
        "2 Docs\n(360m)",
        "2 Docs*\n(240m)",
        "2 Docs\n(120m)",
        "3 Docs\n(240m)",
        "4 Docs\n(240m)",
    ]
    oos_ord = oos.set_index("Scenario").loc[order].reset_index()
    idx = np.arange(len(oos_ord))

    def _ci_bounds(ci_str):
        lo, hi = [float(v) for v in ci_str.strip("[]").split(",")]
        return lo, hi

    emp_m = oos_ord["Empirical_P90_Mean"].to_numpy()
    des_m = oos_ord["DES_P90_Mean"].to_numpy()
    emp_bounds = np.array([_ci_bounds(s) for s in oos_ord["Empirical_P90_CI95"]])
    des_bounds = np.array([_ci_bounds(s) for s in oos_ord["DES_P90_CI95"]])

    emp_err = np.vstack([np.maximum(0.0, emp_m - emp_bounds[:, 0]), np.maximum(0.0, emp_bounds[:, 1] - emp_m)])
    des_err = np.vstack([np.maximum(0.0, des_m - des_bounds[:, 0]), np.maximum(0.0, des_bounds[:, 1] - des_m)])

    ax2.bar(
        idx - w / 2.0,
        emp_m,
        width=w,
        yerr=emp_err,
        capsize=2.8,
        error_kw={"elinewidth": 1.0, "ecolor": "#102A43"},
        color=COLORS["navy"],
        alpha=0.88,
        label="Historical CSV (50 runs)",
    )
    ax2.bar(
        idx + w / 2.0,
        des_m,
        width=w,
        yerr=des_err,
        capsize=2.8,
        error_kw={"elinewidth": 1.0, "ecolor": "#102A43"},
        color=COLORS["teal"],
        alpha=0.88,
        label="Calibrated DES (50 runs)",
    )
    ax2.set_yscale("symlog", linthresh=20.0)
    ax2.set_ylim(0, 3500)
    ax2.set_xticks(idx)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Scenario (*used for calibration)")
    ax2.set_ylabel("Run-Level P90 Doctor Wait (min, symlog)")
    ax2.set_title("(b) Historical vs. Calibrated DES Across 6 Scenarios")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

    fig.savefig(FIGURES_DIR / "fig2_validation.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_fig3_main_results():
    traj = pd.read_csv(RESULTS_DIR / "queue_trajectories.csv")
    summ = pd.read_csv(RESULTS_DIR / "main_policy_summary.csv").set_index("policy_key")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 3.65),
        gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.28},
    )

    styles = [
        ("Baseline", "2 Doctors (Baseline)", COLORS["terracotta"]),
        ("Policy_A", "3 Doctors (Policy A)", COLORS["teal"]),
        ("Policy_B", "4 Doctors (Policy B)", COLORS["navy"]),
    ]

    # Panel (a): Doctor queue length Q1(t) across the operating day
    ax = axes[0]
    ax.axvspan(0, 240, color="#E2E8F0", alpha=0.55, label="Peak window (0-240 min)")
    ax.axvline(1080, color=COLORS["slate"], linestyle="--", linewidth=1.1, label="18-hr shift end (1,080 min)")

    for key, label, col in styles:
        t = traj["time_min"]
        mean_q = traj[f"{key}_mean_q"]
        lo_q = traj[f"{key}_p10_q"]
        hi_q = traj[f"{key}_p90_q"]
        ax.plot(t, mean_q, color=col, linewidth=1.9, label=label)
        ax.fill_between(t, lo_q, hi_q, color=col, alpha=0.14)

    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 72)
    ax.set_xlabel("Time Since Clinic Opening (minutes)")
    ax.set_ylabel("Waiting Patients $Q_1(t)$")
    ax.set_title("(a) Doctor Queue Length Across the Day")
    ax.legend(loc="upper left", fontsize=8.1, frameon=True, facecolor="white", edgecolor="none")

    # Panel (b): Run-level P90 wait overall and by priority class
    ax2 = axes[1]
    groups = ["Overall", "Priority 1", "Priority 2", "Priority 3"]
    metric_keys = ["p90_doc_wait", "p90_prio1_wait", "p90_prio2_wait", "p90_prio3_wait"]
    x = np.arange(len(groups))
    w = 0.24

    for offset, (key, label, col) in zip([-w, 0.0, w], styles):
        vals = [float(summ.loc[key, f"{mk}_mean"]) for mk in metric_keys]
        ax2.bar(x + offset, vals, width=w, color=col, alpha=0.90, label=label)

    ax2.axhline(
        PRIMARY_TARGET_MIN,
        color=COLORS["slate"],
        linestyle="--",
        linewidth=1.25,
        label="15-min target",
    )
    ax2.set_yscale("symlog", linthresh=20.0)
    ax2.set_ylim(0, 800)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups)
    ax2.set_xlabel("Patient Group (Priority 3 = Urgent)")
    ax2.set_ylabel("Run-Level P90 Doctor Wait (min, symlog)")
    ax2.set_title("(b) P90 Doctor Wait by Priority Class")
    ax2.legend(loc="upper right", fontsize=8.1, frameon=True, facecolor="white", edgecolor="none")

    fig.savefig(FIGURES_DIR / "fig3_main_results.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_fig4_sensitivity():
    scan = pd.read_csv(RESULTS_DIR / "threshold_scan.csv")
    cells = pd.read_csv(RESULTS_DIR / "factorial_2x2_cells.csv")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 3.65),
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.28},
    )

    # Panel (a): -5% to +10% arrival-rate and consultation-time sensitivity
    ax = axes[0]
    ax.axhline(
        PRIMARY_TARGET_MIN,
        color=COLORS["slate"],
        linestyle="--",
        linewidth=1.2,
        label="15-min target",
    )
    ax.plot(
        scan["pct_change"],
        scan["arr_rate_c3_p90"],
        "o-",
        color=COLORS["teal"],
        linewidth=1.85,
        markersize=5.0,
        label="Policy A (3 Docs): Arrival rate",
    )
    ax.plot(
        scan["pct_change"],
        scan["doc_serv_c3_p90"],
        "s--",
        color=COLORS["terracotta"],
        linewidth=1.85,
        markersize=4.8,
        label="Policy A (3 Docs): Consult time",
    )
    ax.plot(
        scan["pct_change"],
        scan["arr_rate_c4_p90"],
        "^-",
        color=COLORS["navy"],
        linewidth=1.7,
        markersize=5.0,
        label="Policy B (4 Docs): Arrival rate",
    )
    ax.plot(
        scan["pct_change"],
        scan["doc_serv_c4_p90"],
        "d:",
        color=COLORS["muted_blue"],
        linewidth=1.7,
        markersize=4.8,
        label="Policy B (4 Docs): Consult time",
    )
    ax.set_xlim(-5.6, 10.6)
    ax.set_ylim(-1.5, 46.0)
    ax.set_xlabel("Parameter Change from Baseline (%)")
    ax.set_ylabel("Run-Level P90 Doctor Wait (minutes)")
    ax.set_title("(a) Arrival Rate & Consultation Time Sensitivity")
    ax.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="white", edgecolor="none")

    # Panel (b): Clean 2x2 interaction plot (3 vs 4 doctors x Empirical vs Poisson arrivals)
    ax2 = axes[1]
    d1 = cells[cells["design"].str.startswith("Staffing (3 vs 4 Docs) x Arrival Variability")]
    emp_3 = d1[(d1["factor_A_doctors"].str.startswith("3")) & (d1["factor_B_regime"].str.startswith("Empirical"))].iloc[0]
    emp_4 = d1[(d1["factor_A_doctors"].str.startswith("4")) & (d1["factor_B_regime"].str.startswith("Empirical"))].iloc[0]
    poi_3 = d1[(d1["factor_A_doctors"].str.startswith("3")) & (d1["factor_B_regime"].str.startswith("Poisson"))].iloc[0]
    poi_4 = d1[(d1["factor_A_doctors"].str.startswith("4")) & (d1["factor_B_regime"].str.startswith("Poisson"))].iloc[0]

    x_pos = [0, 1]
    ax2.axhline(
        PRIMARY_TARGET_MIN,
        color=COLORS["slate"],
        linestyle="--",
        linewidth=1.2,
        label="15-min target",
    )
    ax2.errorbar(
        x_pos,
        [emp_3["p90_mean"], emp_4["p90_mean"]],
        yerr=[
            [emp_3["p90_mean"] - emp_3["p90_ci_low"], emp_4["p90_mean"] - emp_4["p90_ci_low"]],
            [emp_3["p90_ci_high"] - emp_3["p90_mean"], emp_4["p90_ci_high"] - emp_4["p90_mean"]],
        ],
        fmt="o-",
        linewidth=2.0,
        markersize=6.5,
        capsize=3.5,
        color=COLORS["teal"],
        label="Empirical (CV = 0.26)",
    )
    ax2.errorbar(
        x_pos,
        [poi_3["p90_mean"], poi_4["p90_mean"]],
        yerr=[
            [poi_3["p90_mean"] - poi_3["p90_ci_low"], poi_4["p90_mean"] - poi_4["p90_ci_low"]],
            [poi_3["p90_ci_high"] - poi_3["p90_mean"], poi_4["p90_ci_high"] - poi_4["p90_mean"]],
        ],
        fmt="s--",
        linewidth=2.0,
        markersize=6.5,
        capsize=3.5,
        color=COLORS["terracotta"],
        label="Piecewise-Poisson (CV = 1.00)",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["3 Doctors (Policy A)", "4 Doctors (Policy B)"])
    ax2.set_xlim(-0.22, 1.25)
    ax2.set_ylim(-2.0, 62.0)
    ax2.set_xlabel("Doctor Staffing Level")
    ax2.set_ylabel("Run-Level P90 Doctor Wait (minutes)")
    ax2.set_title("(b) Staffing x Arrival Variability Interaction")
    ax2.legend(loc="upper right", fontsize=8.1, frameon=True, facecolor="white", edgecolor="none")

    fig.savefig(FIGURES_DIR / "fig4_sensitivity.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def generate_all_figures():
    _apply_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_fig1_clinic_flow()
    make_fig2_validation()
    make_fig3_main_results()
    make_fig4_sensitivity()
    print(f"Saved 4 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
