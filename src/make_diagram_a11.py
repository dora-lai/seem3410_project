"""
make_diagram_a11.py  (Task A11: Section 1 System Process Diagram)
Generates a clean outpatient clinic process flow diagram.
Run: python src/make_diagram_a11.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

import config

FIGURES_DIR = os.path.join(config.ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def draw_box(ax, x, y, w, h, title, subtitle, facecolor, edgecolor="#2c3e50", lw=1.6):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=3
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1a252f", zorder=4)
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
            fontsize=8.5, color="#2c3e50", zorder=4)


def draw_arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=7,head_width=4.5",
        color="#2c3e50", linewidth=1.6, zorder=4
    )
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(12, 4.4), dpi=180)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    boundary = Rectangle(
        (2.35, 0.55), 7.35, 3.25,
        linewidth=1.5, linestyle="--", edgecolor="#5d6d7e",
        facecolor="#f8f9fa", zorder=1
    )
    ax.add_patch(boundary)
    ax.text(6.02, 3.60, "Outpatient Clinic System Boundary (Terminating Run: 200 Patients)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#34495e", zorder=2)

    draw_box(ax, 0.20, 1.50, 1.85, 1.45, "Patient Arrivals",
             "Peak (t <= 240 min)\n& Post-Peak (t > 240)\nPriority: P1, P2, P3", "#eaf2f8")
    draw_box(ax, 2.60, 1.50, 1.85, 1.45, "Doctor Queue",
             "Non-Preemptive\nPriority (P3 > P2 > P1)\nFCFS within class", "#fef5e7")
    draw_box(ax, 4.85, 1.50, 1.85, 1.45, "Doctor Consultation",
             "Policy Lever:\nn = 2, 3, or 4 Doctors\n(Service: 10-20 min)",
             "#d5f5e3", edgecolor="#1e8449", lw=2.0)
    draw_box(ax, 7.10, 1.50, 2.35, 1.45, "Pharmacy Queue & Counter",
             "FCFS Queue\n1 Counter (Main)\n2 Counters (Secondary)\n(Service: 2-5 min)", "#f4ecf7")
    draw_box(ax, 10.05, 1.50, 1.75, 1.45, "Patient Departure",
             "Record Doc Wait,\nPharm Wait &\nTotal Time", "#fdebd0")

    draw_arrow(ax, 2.07, 2.22, 2.58, 2.22)
    draw_arrow(ax, 4.47, 2.22, 4.83, 2.22)
    draw_arrow(ax, 6.72, 2.22, 7.08, 2.22)
    draw_arrow(ax, 9.47, 2.22, 10.03, 2.22)

    ax.text(6.0, 0.25,
            "Primary Metric: Mean Run-Level P90 Doctor Wait (Target <= 15 min)  |  "
            "CRN: 50 Replications (Seeds 3410-3459)  |  Cost: HKD 500/hr x 4-hr Shift",
            ha="center", va="center", fontsize=8.8, style="italic", color="#2c3e50")
    ax.text(5.775, 1.15, "Decision Lever (Main)", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#1e8449")
    ax.text(8.275, 1.15, "Secondary Analysis", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#7d3c98")

    out_path = os.path.join(FIGURES_DIR, "fig0_process_diagram.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[A11] Saved {out_path}")


if __name__ == "__main__":
    main()