"""
run_all.py
Master reproduction script for the SEEM3410 Outpatient Clinic DES project.
Run from project root:  python run_all.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

PIPELINE = [
    ("1. Data check (B1)",                  "src/data_check.py"),
    ("1b. Pharmacy 1-counter check (B2)",   "src/check_pharmacy_overlap.py"),
    ("2. Estimate inputs (B3-B5)",          "src/estimate_inputs.py"),
    ("3. Model unit tests (A6)",            "src/test_model.py"),
    ("4. Consistency check vs Kaggle (A7)", "src/a7_check.py"),
    ("5. Main experiment & CIs (A8/A9/B9)", "src/run_main.py"),
    ("6. Process diagram (A11)",            "src/make_diagram_a11.py"),
    ("7. Main figures & table (B10)",       "src/make_figures_b10.py"),
    ("8. Sensitivity analysis (A10/B11)",   "src/run_sensitivity.py"),
]


def main():
    for label, script in PIPELINE:
        script_path = os.path.join(ROOT, script)
        if not os.path.exists(script_path):
            print(f"[SKIP] {script} not found.")
            continue
        print(f"\n{'=' * 70}\n>>> Running {label}: {script}\n{'=' * 70}")
        result = subprocess.run([sys.executable, script_path], cwd=ROOT)
        if result.returncode != 0:
            print(f"\n[ERROR] Pipeline stopped because {script} exited with code {result.returncode}.")
            sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETED SUCCESSFULLY.")
    print("Tables saved in:  results/")
    print("Figures saved in: figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()