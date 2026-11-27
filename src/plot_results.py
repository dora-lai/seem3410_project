"""
Generate publication-quality figures for the SEEM3410 project report.

Outputs in figures/:
  1. fig1_process_flow.png       - System boundary and two-stage outpatient DES logic
  2. fig2_input_validation.png   - Calibrated input PMFs + out-of-sample 6-scenario validation
  3. fig3_queue_dynamics.png     - Transient queue buildup/dissipation + priority-class P90 waits
  4. fig4_policy_cost_sla.png    - Cost vs P90 wait frontier + SLA attainment across targets
  5. fig5_sensitivity_2x2.png    - Break-even threshold curves + 2x2 factorial interaction plot
"""
from __future__ import annotations

import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from config import CALIBRATED_JSON_PATH, FIGURES_DIR, PRIMARY_TARGET_MIN, RESULTS_DIR

# Clean academic styling
PALETTE = {
    "navy": "#1B365D",
    "blue": "#2B6CB0",
    "teal": "#0D9488",
    "amber": "#D97706",
    "red": "#C53030",
    "slate": "#4A5568",
    "light_bg": "#F7FAFC",
    "grid": "#CBD5E0",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linestyle": "--",
            "grid.alpha": 0.55,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def make_fig1_process_flow() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.3))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.3)
    ax.axis("off")

    # System boundary dashed container
    boundary = FancyBboxPatch(
        (2.05, 0.30),
        6.55,
        2.65,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.5,
        linestyle="--",
        edgecolor=PALETTE["navy"],
        facecolor="#F8FAFC",
    )
    ax.add_patch(boundary)
    ax.text(
        5.32,
        2.77,
        "System Boundary: Unified Two-Stage Outpatient Clinic DES (FEL Tie-Break: PHARM_END -> ARRIVAL -> DOC_END)",
        ha="center",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    def draw_box(x, y, w, h, title, body, fc, ec):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.10",
            linewidth=1.4,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1A202C")
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=8.6, color="#2D3748")

    draw_box(
        0.15,
        0.92,
        1.65,
        1.42,
        "Exogenous Arrivals",
        "N = 200 patients/day\nPeak (t < 240m): 13.96/h\nOff-peak (t >= 240m): 10.94/h\nP1: 65.8%, P2: 29.4%, P3: 4.9%",
        "#EDF2F7",
        PALETTE["slate"],
    )
    draw_box(
        2.25,
        0.92,
        1.48,
        1.42,
        "Waiting Room Q1",
        "Non-preemptive\nPriority 3 > 2 > 1\n(FCFS within class)\nMetric: DocWaitTime",
        "#EBF8FF",
        PALETTE["blue"],
    )
    draw_box(
        4.02,
        0.92,
        1.52,
        1.42,
        "Stage 1: Doctors",
        "c in {2, 3, 4} servers\nDiscrete {10..20} min\nMean = 13.38 min\n(mu = 4.48/hr/doc)",
        "#E6FFFA",
        PALETTE["teal"],
    )
    draw_box(
        5.82,
        0.92,
        1.25,
        1.42,
        "Pharm Queue Q2",
        "FCFS discipline\nImmediate transfer\nMetric: MedWaitTime",
        "#FEFCBF",
        PALETTE["amber"],
    )
    draw_box(
        7.30,
        0.92,
        1.15,
        1.42,
        "Stage 2: Pharm",
        "m = 1 counter\n(m = 2 tested)\n{2,3,4,5} min\nMean = 3.00 min",
        "#FFF5F5",
        PALETTE["red"],
    )
    draw_box(
        8.82,
        0.92,
        1.52,
        1.42,
        "Clinic Exit & KPIs",
        "Primary SLA:\nP90 DocWait <= 15m\nTotalTimeInHospital\nShift + Overtime HKD",
        "#EDF2F7",
        PALETTE["slate"],
    )

    arrow_pairs = [
        ((1.80, 1.63), (2.25, 1.63)),
        ((3.73, 1.63), (4.02, 1.63)),
        ((5.54, 1.63), (5.82, 1.63)),
        ((7.07, 1.63), (7.30, 1.63)),
        ((8.45, 1.63), (8.82, 1.63)),
    ]
    for p0, p1 in arrow_pairs:
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.5,
                color=PALETTE["navy"],
            )
        )

    ax.text(
        5.32,
        0.10,
        "CRN Experimental Control: 4 independent RNG substreams per replication (arrivals, priority, consultation, pharmacy) across R = 50 days.",
        ha="center",
        va="center",
        fontsize=8.8,
        style="italic",
        color=PALETTE["slate"],
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES_DIR / "fig1_process_flow.png", bbox_inches="tight")
    plt.close(fig)


def make_fig2_input_and_validation() -> None:
    with open(CALIBRATED_JSON_PATH, "r", encoding="utf-8") as f:
        cal = json.load(f)
    oos = pd.read_csv(RESULTS_DIR / "validation_out_of_sample.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.85), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Left panel: Peak vs Off-Peak Interarrival PMF
    ax = axes[0]
    gaps = [3, 4, 5, 6, 7]
    pk_p = [cal["interarrival_peak"]["probs"][str(g)] for g in gaps]
    of_p = [cal["interarrival_offpeak"]["probs"][str(g)] for g in gaps]
    x = np.arange(len(gaps))
    w = 0.36
    ax.bar(x - w / 2, pk_p, width=w, color=PALETTE["red"], alpha=0.88, label="Peak (t < 240m, mean 4.30m)")
    ax.bar(x + w / 2, of_p, width=w, color=PALETTE["blue"], alpha=0.88, label="Off-peak (t >= 240m, mean 5.49m)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g} min" for g in gaps])
    ax.set_ylabel("Empirical Probability")
    ax.set_ylim(0, 0.47)
    ax.set_title("(a) Calibrated Interarrival Distributions")
    ax.legend(loc="upper right", frameon=True, facecolor="white")

    # Right panel: Out-of-sample validation across all 6 scenarios
    ax2 = axes[1]
    label_map = {
        "Staffing_1Doc": "1 Doc\n(240m)",
        "Peak_120Min": "2 Docs\n(120m)",
        "Baseline_2Doc_240Peak": "2 Docs*\n(240m)",
        "Peak_360Min": "2 Docs\n(360m)",
        "Staffing_3Doc": "3 Docs\n(240m)",
        "Staffing_4Doc": "4 Docs\n(240m)",
    }
    order = [
        "Staffing_1Doc",
        "Peak_360Min",
        "Baseline_2Doc_240Peak",
        "Peak_120Min",
        "Staffing_3Doc",
        "Staffing_4Doc",
    ]
    oos_ord = oos.set_index("Scenario").loc[order].reset_index()
    idx = np.arange(len(oos_ord))

    def _parse_hw(ci_str: str, mean_v: float) -> float:
        lo, hi = [float(v) for v in ci_str.strip("[]").split(",")]
        return 0.5 * (hi - lo)

    emp_m = oos_ord["Empirical_P90_Mean"].to_numpy()
    des_m = oos_ord["DES_P90_Mean"].to_numpy()
    emp_hw = np.array([_parse_hw(s, m) for s, m in zip(oos_ord["Empirical_P90_CI95"], emp_m)])
    des_hw = np.array([_parse_hw(s, m) for s, m in zip(oos_ord["DES_P90_CI95"], des_m)])

    ax2.bar(idx - w / 2, emp_m, width=w, yerr=emp_hw, capsize=3, color=PALETTE["navy"], alpha=0.88, label="Historical Data (50 runs)")
    ax2.bar(idx + w / 2, des_m, width=w, yerr=des_hw, capsize=3, color=PALETTE["teal"], alpha=0.88, label="Generative DES (Baseline-calibrated)")
    ax2.set_yscale("symlog", linthresh=20.0)
    ax2.set_xticks(idx)
    ax2.set_xticklabels([label_map[s] for s in oos_ord["Scenario"]])
    ax2.set_ylabel("Mean Run-Level P90 Wait (min, symlog)")
    ax2.set_title("(b) Out-of-Sample Validation Across All 6 Scenarios (*Calib.)")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_input_validation.png", bbox_inches="tight")
    plt.close(fig)


def make_fig3_queue_and_priority() -> None:
    traj = pd.read_csv(RESULTS_DIR / "queue_trajectories.csv")
    summ = pd.read_csv(RESULTS_DIR / "main_policy_summary.csv").set_index("policy_key")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.85), gridspec_kw={"width_ratios": [1.22, 1.0]})

    # Left panel: Transient queue trajectories over clinic operating day
    ax = axes[0]
    styles = [
        ("Baseline", "Baseline (2 Docs)", PALETTE["red"]),
        ("Policy_A", "Policy A (3 Docs)", PALETTE["teal"]),
        ("Policy_B", "Policy B (4 Docs)", PALETTE["navy"]),
    ]
    ax.axvspan(0, 240, color="#FED7D7", alpha=0.28, label="Peak arrival window (0-240m)")
    ax.axvline(1080, color=PALETTE["slate"], linestyle=":", linewidth=1.3, label="Scheduled day end (1,080m)")

    for key, label, col in styles:
        t = traj["time_min"]
        m = traj[f"{key}_mean_q"]
        lo = traj[f"{key}_p10_q"]
        hi = traj[f"{key}_p90_q"]
        ax.plot(t, m, color=col, linewidth=2.0, label=label)
        ax.fill_between(t, lo, hi, color=col, alpha=0.14)

    ax.set_xlim(0, 1400)
    ax.set_xlabel("Simulation Clock (minutes from clinic opening)")
    ax.set_ylabel("Patients Waiting for Doctor Q1(t)")
    ax.set_title("(a) Transient Queue Overload & Dissipation (Mean & P10-P90)")
    ax.legend(loc="upper left", fontsize=8.4, frameon=True, facecolor="white")

    # Right panel: Priority-class P90 waiting times under 2, 3, and 4 doctors
    ax2 = axes[1]
    classes = ["Overall", "Prio 1\n(Low, 66%)", "Prio 2\n(Med, 29%)", "Prio 3\n(Urgent, 5%)"]
    metric_keys = ["p90_doc_wait", "p90_prio1_wait", "p90_prio2_wait", "p90_prio3_wait"]
    x = np.arange(len(classes))
    w = 0.25

    for offset, (key, label, col) in zip([-w, 0.0, w], styles):
        vals = [max(float(summ.loc[key, f"{mk}_mean"]), 0.05) for mk in metric_keys]
        ax2.bar(x + offset, vals, width=w, color=col, alpha=0.90, label=label)

    ax2.axhline(PRIMARY_TARGET_MIN, color=PALETTE["amber"], linestyle="--", linewidth=1.5, label="15-min P90 SLA")
    ax2.set_yscale("symlog", linthresh=20.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.set_ylabel("90th-Percentile Doctor Wait (min, symlog)")
    ax2.set_title("(b) Priority Triage Shielding by Staffing Level")
    ax2.legend(loc="upper right", fontsize=8.3, frameon=True, facecolor="white")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_queue_dynamics.png", bbox_inches="tight")
    plt.close(fig)


def make_fig4_policy_cost_and_sla() -> None:
    summ = pd.read_csv(RESULTS_DIR / "main_policy_summary.csv")
    reps = pd.read_csv(RESULTS_DIR / "main_replications.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.85), gridspec_kw={"width_ratios": [1.12, 1.0]})

    # Left panel: Cost vs P90 waiting time trade-off (Scheduled vs Overtime-inclusive)
    ax = axes[0]
    colors = {"Baseline": PALETTE["red"], "Policy_A": PALETTE["teal"], "Policy_B": PALETTE["navy"]}
    ax.axhline(PRIMARY_TARGET_MIN, color=PALETTE["amber"], linestyle="--", linewidth=1.5, label="15-min P90 SLA target")
    ax.axhspan(0, PRIMARY_TARGET_MIN, color="#E6FFFA", alpha=0.35)

    for _, row in summ.iterrows():
        pk = row["policy_key"]
        col = colors[pk]
        sched_c = row["scheduled_total_cost_hkd_mean"] / 1000.0
        eff_c = row["effective_total_cost_hkd_mean"] / 1000.0
        y = row["p90_doc_wait_mean"]
        y_err = [[y - row["p90_doc_wait_ci_low"]], [row["p90_doc_wait_ci_high"] - y]]

        ax.errorbar(
            sched_c,
            max(y, 0.1),
            yerr=y_err,
            fmt="o",
            markersize=8.5,
            color=col,
            capsize=4,
            label=f"{row['policy_label']} (Scheduled)",
        )
        if abs(eff_c - sched_c) > 0.2:
            ax.plot(eff_c, max(y, 0.1), marker="s", markersize=8.5, color=col)
            ax.annotate(
                "",
                xy=(eff_c, max(y, 0.1)),
                xytext=(sched_c, max(y, 0.1)),
                arrowprops=dict(arrowstyle="->", linestyle=":", color=col, lw=1.5),
            )
            ax.text(
                eff_c - 1.5,
                y * 0.68,
                f"With 4.6h OT:\nHK${eff_c:.1f}k",
                fontsize=8.3,
                color=col,
                fontweight="bold",
            )

    ax.annotate("Feasible & Lowest-Cost:\nHK$30.6k, P90 = 11.4m", xy=(30.6, 11.4), xytext=(23.2, 33.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["teal"], lw=1.3),
                fontsize=8.6, fontweight="bold", color=PALETTE["teal"])

    ax.set_yscale("symlog", linthresh=20.0)
    ax.set_xlim(19.5, 42.5)
    ax.set_xlabel("Daily Clinic Staffing Cost (HK$ thousands / 18-hr day)")
    ax.set_ylabel("Mean Run-Level P90 Doctor Wait (min, symlog)")
    ax.set_title("(a) Staffing Cost vs. Service-Level Frontier")
    ax.legend(loc="upper right", fontsize=8.2, frameon=True, facecolor="white")

    # Right panel: Probability of satisfying P90 <= T across threshold T
    ax2 = axes[1]
    thresholds = np.linspace(0, 35, 141)
    for pk, label, col in [
        ("Baseline", "Baseline (2 Docs)", PALETTE["red"]),
        ("Policy_A", "Policy A (3 Docs)", PALETTE["teal"]),
        ("Policy_B", "Policy B (4 Docs)", PALETTE["navy"]),
    ]:
        vals = reps.loc[reps["policy_key"] == pk, "p90_doc_wait"].to_numpy()
        probs = [(vals <= t).mean() * 100.0 for t in thresholds]
        ax2.plot(thresholds, probs, linewidth=2.2, color=col, label=label)

    ax2.axvline(PRIMARY_TARGET_MIN, color=PALETTE["amber"], linestyle="--", linewidth=1.5, label="Primary T = 15 min")
    ax2.set_xlim(0, 35)
    ax2.set_ylim(-3, 105)
    ax2.set_xlabel("Target Threshold T for Run-Level P90 Wait (minutes)")
    ax2.set_ylabel("% of Clinic Days Meeting P90 <= T")
    ax2.set_title("(b) Daily SLA Reliability Curve (R = 50 Days)")
    ax2.legend(loc="center right", fontsize=8.5, frameon=True, facecolor="white")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_policy_cost_sla.png", bbox_inches="tight")
    plt.close(fig)


def make_fig5_sensitivity_and_factorial() -> None:
    scan = pd.read_csv(RESULTS_DIR / "threshold_scan.csv")
    cells = pd.read_csv(RESULTS_DIR / "factorial_2x2_cells.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.85))

    # Left panel: Break-even threshold scan for 3 vs 4 doctors
    ax = axes[0]
    ax.axhline(PRIMARY_TARGET_MIN, color=PALETTE["amber"], linestyle="--", linewidth=1.5, label="15-min P90 SLA")
    ax.plot(scan["pct_change"], scan["arr_rate_c3_p90"], "o-", color=PALETTE["teal"], linewidth=2.0, label="3 Docs: Arrival rate change")
    ax.plot(scan["pct_change"], scan["doc_serv_c3_p90"], "s--", color=PALETTE["red"], linewidth=2.0, label="3 Docs: Consult duration change")
    ax.plot(scan["pct_change"], scan["arr_rate_c4_p90"], "^-", color=PALETTE["navy"], linewidth=1.8, label="4 Docs: Arrival rate change")
    ax.plot(scan["pct_change"], scan["doc_serv_c4_p90"], "d:", color=PALETTE["blue"], linewidth=1.8, label="4 Docs: Consult duration change")
    ax.axvline(2.0, color=PALETTE["slate"], linestyle=":", linewidth=1.2)
    ax.text(2.3, 31.0, "Policy A crosses\n15m near +2%", fontsize=8.5, color=PALETTE["slate"], fontweight="bold")
    ax.set_xlabel("Parameter Perturbation (% from calibrated baseline)")
    ax.set_ylabel("Mean Run-Level P90 Doctor Wait (minutes)")
    ax.set_title("(a) Break-Even Sensitivity of Policy A vs. Policy B")
    ax.legend(loc="upper left", fontsize=8.1, frameon=True, facecolor="white")

    # Right panel: Week 6 2x2 factorial interaction plot (Doctors x Arrival Variability)
    ax2 = axes[1]
    d1 = cells[cells["design"].str.startswith("Staffing (3 vs 4 Docs) x Arrival Variability")]
    x_pos = [0, 1]

    emp_3 = d1[(d1["factor_A_doctors"].str.startswith("3")) & (d1["factor_B_regime"].str.startswith("Empirical"))].iloc[0]
    emp_4 = d1[(d1["factor_A_doctors"].str.startswith("4")) & (d1["factor_B_regime"].str.startswith("Empirical"))].iloc[0]
    poi_3 = d1[(d1["factor_A_doctors"].str.startswith("3")) & (d1["factor_B_regime"].str.startswith("Poisson"))].iloc[0]
    poi_4 = d1[(d1["factor_A_doctors"].str.startswith("4")) & (d1["factor_B_regime"].str.startswith("Poisson"))].iloc[0]

    ax2.axhline(PRIMARY_TARGET_MIN, color=PALETTE["amber"], linestyle="--", linewidth=1.5, label="15-min P90 SLA")
    ax2.errorbar(
        x_pos,
        [emp_3["p90_mean"], emp_4["p90_mean"]],
        yerr=[
            [emp_3["p90_mean"] - emp_3["p90_ci_low"], emp_4["p90_mean"] - emp_4["p90_ci_low"]],
            [emp_3["p90_ci_high"] - emp_3["p90_mean"], emp_4["p90_ci_high"] - emp_4["p90_mean"]],
        ],
        fmt="o-",
        linewidth=2.2,
        markersize=7.5,
        capsize=4,
        color=PALETTE["teal"],
        label="Scheduled Arrivals (Empirical CV = 0.26)",
    )
    ax2.errorbar(
        x_pos,
        [poi_3["p90_mean"], poi_4["p90_mean"]],
        yerr=[
            [poi_3["p90_mean"] - poi_3["p90_ci_low"], poi_4["p90_mean"] - poi_4["p90_ci_low"]],
            [poi_3["p90_ci_high"] - poi_3["p90_mean"], poi_4["p90_ci_high"] - poi_4["p90_mean"]],
        ],
        fmt="s--",
        linewidth=2.2,
        markersize=7.5,
        capsize=4,
        color=PALETTE["red"],
        label="Unscheduled Walk-ins (Poisson CV = 1.00)",
    )
    cond_emp = emp_4["p90_mean"] - emp_3["p90_mean"]
    cond_poi = poi_4["p90_mean"] - poi_3["p90_mean"]
    ax2.annotate(
        f"Significant A x B Interaction:\n4th doctor saves {cond_poi:.1f}m under\nPoisson vs {cond_emp:.1f}m under Empirical",
        xy=(0.48, 24.0),
        xytext=(0.18, 28.5),
        fontsize=8.3,
        color="#1A202C",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F5", edgecolor=PALETTE["red"], alpha=0.92),
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["3 Doctors (Policy A)", "4 Doctors (Policy B)"])
    ax2.set_xlim(-0.25, 1.25)
    ax2.set_ylim(-2, 46)
    ax2.set_ylabel("Mean Run-Level P90 Doctor Wait (minutes)")
    ax2.set_title("(b) 2x2 Factorial Interaction: Staffing x Arrival CV")
    ax2.legend(loc="upper right", fontsize=8.2, frameon=True, facecolor="white")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_sensitivity_2x2.png", bbox_inches="tight")
    plt.close(fig)


def generate_all_figures() -> None:
    _set_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_fig1_process_flow()
    make_fig2_input_and_validation()
    make_fig3_queue_and_priority()
    make_fig4_policy_cost_and_sla()
    make_fig5_sensitivity_and_factorial()
    print(f"Generated fig1..fig5 in {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
