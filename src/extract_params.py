import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data
df = pd.read_csv('data/hospital_sim_6_scenarios.csv')

# 只看 baseline 場景
baseline = df[df['Scenario'] == 'Baseline_2Doc_240Peak'].copy()
print(f"Baseline scenario: {len(baseline)} rows")
print(f"Runs: {baseline['RunID'].nunique()}")
print(f"Patients per run: {baseline.groupby('RunID').size().unique()}")

# ============================================================
# Task B3: Arrival Pattern
# ============================================================
print("\n" + "="*60)
print("Arrival Pattern")
print("="*60)

# Peak vs. post-peak arrivals
peak_arrivals = baseline[baseline['IsPeakArrival'] == True]
post_peak_arrivals = baseline[baseline['IsPeakArrival'] == False]
print(f"Peak arrivals: {len(peak_arrivals)} ({len(peak_arrivals)/len(baseline)*100:.1f}%)")
print(f"Post-peak arrivals: {len(post_peak_arrivals)} ({len(post_peak_arrivals)/len(baseline)*100:.1f}%)")

# Interarrival times (grouped by run)
interarrival = []
for run_id, group in baseline.groupby('RunID'):
    arrivals = group['ArrivalTime'].sort_values().values
    if len(arrivals) > 1:
        diffs = np.diff(arrivals)
        interarrival.extend(diffs)

interarrival = np.array(interarrival)
print(f"\nInterarrival time (minutes):")
print(f"  Mean: {interarrival.mean():.2f}")
print(f"  Median: {np.median(interarrival):.2f}")
print(f"  Std dev: {interarrival.std():.2f}")
print(f"  Arrival rate: {60/interarrival.mean():.1f} patients/hour")

# Plot interarrival time histogram
plt.figure(figsize=(10, 4))
plt.hist(interarrival, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Interarrival time (minutes)')
plt.ylabel('Frequency')
plt.title('Interarrival time distribution (Baseline scenario)')
plt.savefig('figures/b3_interarrival_hist.png', dpi=100, bbox_inches='tight')
plt.close()
print("  Saved plot: figures/b3_interarrival_hist.png")

# Peak arrival rate (per minute)
peak_minutes = baseline['PeakLimit'].iloc[0]
peak_rate = len(peak_arrivals) / baseline['RunID'].nunique() / peak_minutes
print(f"\nPeak arrival rate: {peak_rate:.3f} patients/minute = {peak_rate*60:.1f} patients/hour")

# ============================================================
# Task B4: Priority Mix
# ============================================================
print("\n" + "="*60)
print("B4: Priority Mix")
print("="*60)

priority_counts = baseline['Priority'].value_counts().sort_index()
priority_pct = baseline['Priority'].value_counts(normalize=True).sort_index()
print("Priority distribution:")
for p in priority_counts.index:
    print(f"  Priority {p}: {priority_counts[p]} ({priority_pct[p]*100:.1f}%)")

# Check whether service time depends on priority
print("\nDoctor service time by priority (minutes):")
for p in sorted(baseline['Priority'].unique()):
    subset = baseline[baseline['Priority'] == p]
    print(f"  Priority {p}: mean={subset['DocServiceTime'].mean():.2f}, "
          f"std={subset['DocServiceTime'].std():.2f}, n={len(subset)}")

# Check waiting time by priority
print("\nDoctor waiting time by priority (minutes):")
for p in sorted(baseline['Priority'].unique()):
    subset = baseline[baseline['Priority'] == p]
    print(f"  Priority {p}: mean={subset['DocWaitTime'].mean():.2f}, "
          f"P90={subset['DocWaitTime'].quantile(0.9):.2f}, n={len(subset)}")