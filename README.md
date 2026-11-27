# SEEM3410 Project: Outpatient Clinic Doctor Staffing Simulation

**Course:** SEEM3410 System Modeling and Simulation (CUHK)  
**Group Members:** Lai Sin Hang (1155213900), Lee Ka Tung (1155212062)

This repository contains the Python discrete-event simulation (DES) code, input analysis, and experiment scripts for our SEEM3410 project on doctor staffing in a two-stage hospital outpatient clinic. Patients arrive under a peak/off-peak schedule, wait in a non-preemptive priority queue (Priority 3 > 2 > 1, FCFS within class) to see a doctor, and then proceed to an FCFS pharmacy counter. We evaluate 2, 3, and 4 doctors (plus 1 vs 2 pharmacy counters) against a 15-minute target on run-level 90th-percentile (`P90`) doctor waiting time.

## Project Layout

- `data/`: Public synthetic Kaggle dataset *Hospital Queue and Patient Flow Simulation* by Juneminazuki (`hospital_sim_6_scenarios.csv`, 60,000 simulated patient rows across 6 scenarios; [Kaggle link](https://www.kaggle.com/datasets/juneminazuki/hospital-queue-and-patient-flow-simulation)) and baseline-fitted input distributions (`calibrated_inputs.json`).
- `src/config.py`: Simulation constants, staffing policy definitions, random seed (`3410`), and assumed daily shift/overtime cost formulas in HKD.
- `src/calibrate_inputs.py`: Data checks, empirical PMF fitting on `Baseline_2Doc_240Peak`, and statistical independence tests.
- `src/clinic_des.py`: Two-stage next-event DES engine using a `heapq` Future Event List and Common Random Numbers (CRN) streams.
- `src/validate_model.py`: Exact 300-run dataset trace replay, held-out scenario validation, and analytical $M/M/c$ comparison.
- `src/experiments.py`: 50-replication CRN policy comparisons, pharmacy capacity check, $2 \times 2$ factorial experiments, and sensitivity scans.
- `src/plot_results.py`: Generates the four report figures in `figures/`.
- `tests/test_clinic_des.py`: `pytest` unit tests covering priority tie-breaking, sample-path Little's Law, and dataset trace replay.
- `results/` and `figures/`: CSV summary tables and 300 DPI PNG plots produced by `run_all.py`.

## Setup and Usage

Requires Python 3.10+ with `numpy`, `pandas`, `scipy`, `matplotlib`, and `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run calibration, validation, all experiments, and figure generation
python run_all.py

# Run unit tests
pytest -q
```

All simulation runs are seeded from `MASTER_SEED = 3410` via `numpy.random.SeedSequence(3410)`. Each of the 50 replications spawns four separate uniform streams (arrivals, priority class, consultation time, pharmacy time) shared across policies under Common Random Numbers, so re-running `python run_all.py` reproduces the exact same CSV tables and figures.

## Summary of Main Staffing Policies (50 Replications)

| Policy | Doctors / Pharm | P90 Doc Wait (95% CI) | Mean Doc Wait | Priority-1 P90 Wait | Mean Pharm Wait | Doc Util. | Scheduled Cost | Cost incl. Overtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 2 / 1 | 359.8 min [350.3, 369.2] | 175.6 min | 367.5 min | 0.36 min | 99.0% | HK$21,600 | HK$29,818 |
| **Policy A** | **3 / 1** | **11.4 min [8.8, 14.0]** | **3.1 min** | **14.1 min** | **0.68 min** | **85.2%** | **HK$30,600** | **HK$30,625** |
| Policy B | 4 / 1 | 0.0 min [0.0, 0.0] | 0.06 min | 0.0 min | 0.74 min | 63.9% | HK$39,600 | HK$39,630 |

Replaying all 300 benchmark runs (60,000 simulated patients) through `simulate_clinic` produces zero error on `DocWaitTime`, `MedWaitTime`, `TotalTimeInHospital`, and sample-path Little's Law. Under baseline conditions, **Policy A (3 doctors, 1 pharmacy counter)** meets the 15-minute `P90` waiting-time target (`11.4` min) while avoiding the 4.6-hour daily overtime of 2 doctors and the extra HK$9,005/day cost of a 4th doctor. Sensitivity checks show Policy A stays within the 15-minute target up to roughly `+2%` growth in arrival rate or consultation duration, while unscheduled Poisson walk-ins (`CV = 1.00`) require 4 doctors.
