"""
check_pharmacy_overlap.py  (Verification for B2: 1-pharmacy-counter assumption)
Checks all 60,000 rows in data/hospital_sim_6_scenarios.csv to confirm that no two
patients are ever served simultaneously at the pharmacy within the same run.

Run: python src/check_pharmacy_overlap.py
"""
import os
import pandas as pd
import config

csv_path = os.path.join(config.ROOT, "data", "hospital_sim_6_scenarios.csv")
df = pd.read_csv(csv_path)

# Reconstruct pharmacy service start and end times
df["pharm_start"] = df["ArrivalTime"] + df["DocWaitTime"] + df["DocServiceTime"] + df["MedWaitTime"]
df["pharm_end"] = df["pharm_start"] + df["MedServiceTime"]

# Sanity check: departure time matches ArrivalTime + TotalTimeInHospital
max_time_err = (df["pharm_end"] - (df["ArrivalTime"] + df["TotalTimeInHospital"])).abs().max()

# Count strict overlaps within each (Scenario, RunID)
overlaps = 0
for (scen, run_id), g in df.groupby(["Scenario", "RunID"]):
    g = g.sort_values(["pharm_start", "pharm_end"])
    prev_end = -1.0
    for start, end in zip(g["pharm_start"], g["pharm_end"]):
        if start < prev_end - 1e-9:
            overlaps += 1
        if end > prev_end:
            prev_end = end

print(f"Max TotalTime consistency error: {max_time_err:.6f} min")
print(f"Total simultaneous pharmacy overlaps across all 60,000 rows: {overlaps}")
assert overlaps == 0, f"Found {overlaps} overlapping pharmacy services!"
print("[PASS] Confirmed: at most 1 patient in pharmacy service at any time (1-counter assumption verified).")