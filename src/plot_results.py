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
            "font.size": 10.5,
            "axes.titlesize": 11.4,
            "axes.titleweight": "bold",
            "axes.titlepad": 8.5,
            "axes.labelsize": 10.6,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#486581",
            "axes.linewidth": 0.9,
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
    fig = plt.figure(figsize=(8.8, 2.45))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 8.8)
    ax.set_ylim(0.0, 2.45)
    ax.axis("off")

    boundary = FancyBboxPatch(
        (1.50, 0.18),
        5.80,
        2.06,
        boxstyle="round,pad=0.04,rounding_size=0.07",
        linewidth=1.05,
        linestyle="--",
        edgecolor="#829AB1",
        facecolor="#F8FAFC",
    )
    ax.add_patch(boundary)
    ax.text(
        4.40,
        2.06,
        "Two-Stage Outpatient Clinic Simulation Boundary",
        ha="center",
        va="center",
        fontsize=10.0,
        color=COLORS["slate"],
        fontweight="bold",
    )

    def add_node(x, y, w, h, title, subtitle, fill="#FFFFFF", edge=COLORS["border"]):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.055",
            linewidth=1.1,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2.0,
            y + h * 0.69,
            title,
            ha="center",
            va="center",
            fontsize=9.9,
            fontweight="bold",
            color="#102A43",
        )
        ax.text(
            x + w / 2.0,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.0,
            color="#334E68",
            linespacing=1.26,
        )

    y0 = 0.40
    bh = 1.32
    add_node(0.06, y0, 1.24, bh, "Arrivals", "200 pts / day\nPeak & Off-peak", fill="#F8FAFC", edge=COLORS["slate"])
    add_node(1.62, y0, 1.26, bh, "Doc Queue Q1", "Prio 3 > 2 > 1\nFCFS in class", fill="#FFFFFF", edge=COLORS["navy"])
    add_node(3.16, y0, 1.26, bh, "Doctors", "c in {2, 3, 4}\nMean 13.4 min", fill="#F0F4F8", edge=COLORS["navy"])
    add_node(4.70, y0, 1.20, bh, "Pharm Q2", "FCFS queue\nDirect transfer", fill="#FFFFFF", edge=COLORS["muted_blue"])
    add_node(6.16, y0, 1.04, bh, "Pharmacy", "m = 1 or 2\nMean 3.0 m", fill="#F0F4F8", edge=COLORS["muted_blue"])
    add_node(7.52, y0, 1.22, bh, "Clinic Exit", "Wait KPIs &\nclosing time", fill="#F8FAFC", edge=COLORS["slate"])

    yc = y0 + bh / 2.0
    arrows = [
        (1.33, 1.58),
        (2.91, 3.12),
        (4.45, 4.66),
        (5.93, 6.12),
        (7.23, 7.48),
    ]
    for x_start, x_end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                (x_start, yc),
                (x_end, yc),
                arrowstyle="-|>",
                mutation_scale=11.5,
                linewidth=1.2,
                color=COLORS["navy"],
            )
        )

    fig.savefig(FIGURES_DIR / "fig1_clinic_flow.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def make_fig2_validation():
    with open(CALIBRATED_JSON_PATH, "r", encoding="utf-8") as f:
        cal = json.load(f)
    oos = pd.read_csv(RESULTS_DIR / "validation_out_of_sample.csv")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.9, 3.45),
        gridspec_kw={"width_ratios": [1.0, 1.22], "wspace": 0.30},
    )

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
        label="Peak (<240m, mean 4.30m)",
    )
    ax.bar(
        x + w / 2.0,
        of_p,
        width=w,
        color=COLORS["navy"],
        alpha=0.88,
        label="Off-peak (>=240m, mean 5.49m)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}" for g in gaps])
    ax.set_xlabel("Interarrival Gap (minutes)")
    ax.set_ylabel("Empirical Probability")
    ax.set_ylim(0, 0.56)
    ax.set_title("(a) Calibrated Interarrival PMFs")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

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
        label="Kaggle Dataset (50 runs)",
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
    ax2.set_ylim(0, 3800)
    ax2.set_xticks(idx)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Scenario (*used for calibration)")
    ax2.set_ylabel("P90 Doctor Wait (min, symlog)")
    ax2.set_title("(b) Dataset vs. Calibrated DES (6 Scenarios)")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

    fig.savefig(FIGURES_DIR / "fig2_validation.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def make_fig3_main_results():
    traj = pd.read_csv(RESULTS_DIR / "queue_trajectories.csv")
    summ = pd.read_csv(RESULTS_DIR / "main_policy_summary.csv").set_index("policy_key")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.9, 3.45),
        gridspec_kw={"width_ratios": [1.14, 1.0], "wspace": 0.30},
    )

    styles = [
        ("Baseline", "2 Doctors (Baseline)", COLORS["terracotta"]),
        ("Policy_A", "3 Doctors (Policy A)", COLORS["teal"]),
        ("Policy_B", "4 Doctors (Policy B)", COLORS["navy"]),
    ]

    ax = axes[0]
    ax.axvspan(0, 240, color="#E2E8F0", alpha=0.55, label="Peak (0-240 min)")
    ax.axvline(1080, color=COLORS["slate"], linestyle="--", linewidth=1.1, label="Shift end (1,080 min)")

    for key, label, col in styles:
        t = traj["time_min"]
        mean_q = traj[f"{key}_mean_q"]
        lo_q = traj[f"{key}_p10_q"]
        hi_q = traj[f"{key}_p90_q"]
        ax.plot(t, mean_q, color=col, linewidth=2.0, label=label)
        ax.fill_between(t, lo_q, hi_q, color=col, alpha=0.14)

    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 74)
    ax.set_xlabel("Time Since Opening (minutes)")
    ax.set_ylabel("Waiting Patients $Q_1(t)$")
    ax.set_title("(a) Doctor Queue Length Across the Day")
    ax.legend(loc="upper left", fontsize=8.8, frameon=True, facecolor="white", edgecolor="none")

    ax2 = axes[1]
    groups = ["Overall", "Prio 1", "Prio 2", "Prio 3"]
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
        linewidth=1.3,
        label="15-min target",
    )
    ax2.set_yscale("symlog", linthresh=20.0)
    ax2.set_ylim(0, 950)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups)
    ax2.set_xlabel("Patient Group (Prio 3 = Urgent)")
    ax2.set_ylabel("P90 Doctor Wait (min, symlog)")
    ax2.set_title("(b) P90 Doctor Wait by Priority Class")
    ax2.legend(loc="upper right", fontsize=8.8, frameon=True, facecolor="white", edgecolor="none")

    fig.savefig(FIGURES_DIR / "fig3_main_results.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def make_fig4_sensitivity():
    scan = pd.read_csv(RESULTS_DIR / "threshold_scan.csv")
    cells = pd.read_csv(RESULTS_DIR / "factorial_2x2_cells.csv")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.9, 3.45),
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.30},
    )

    ax = axes[0]
    ax.axhline(
        PRIMARY_TARGET_MIN,
        color=COLORS["slate"],
        linestyle="--",
        linewidth=1.25,
        label="15-min target",
    )
    ax.plot(
        scan["pct_change"],
        scan["arr_rate_c3_p90"],
        "o-",
        color=COLORS["teal"],
        linewidth=1.95,
        markersize=5.2,
        label="3 Docs: Arrival rate",
    )
    ax.plot(
        scan["pct_change"],
        scan["doc_serv_c3_p90"],
        "s--",
        color=COLORS["terracotta"],
        linewidth=1.95,
        markersize=5.0,
        label="3 Docs: Consult time",
    )
    ax.plot(
        scan["pct_change"],
        scan["arr_rate_c4_p90"],
        "^-",
        color=COLORS["navy"],
        linewidth=1.75,
        markersize=5.2,
        label="4 Docs: Arrival rate",
    )
    ax.plot(
        scan["pct_change"],
        scan["doc_serv_c4_p90"],
        "d:",
        color=COLORS["muted_blue"],
        linewidth=1.75,
        markersize=5.0,
        label="4 Docs: Consult time",
    )
    ax.set_xlim(-5.6, 10.6)
    ax.set_ylim(-1.5, 48.0)
    ax.set_xlabel("Change from Baseline (%)")
    ax.set_ylabel("P90 Doctor Wait (minutes)")
    ax.set_title("(a) Arrival Rate & Service Time Shifts")
    ax.legend(loc="upper left", fontsize=8.8, frameon=True, facecolor="white", edgecolor="none")

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
        linewidth=1.25,
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
        linewidth=2.1,
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
        linewidth=2.1,
        markersize=6.5,
        capsize=3.5,
        color=COLORS["terracotta"],
        label="Poisson Walk-ins (CV = 1.00)",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["3 Doctors", "4 Doctors"])
    ax2.set_xlim(-0.25, 1.25)
    ax2.set_ylim(-2.0, 64.0)
    ax2.set_xlabel("Doctor Staffing Level")
    ax2.set_ylabel("P90 Doctor Wait (minutes)")
    ax2.set_title("(b) Staffing x Arrival CV Interaction")
    ax2.legend(loc="upper right", fontsize=8.8, frameon=True, facecolor="white", edgecolor="none")

    fig.savefig(FIGURES_DIR / "fig4_sensitivity.png", bbox_inches="tight", pad_inches=0.06)
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
