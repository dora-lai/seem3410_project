import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/hospital_sim_6_scenarios.csv')
baseline = df[df['Scenario'] == 'Baseline_2Doc_240Peak']

# Waiting time by priority
print("="*60)
print("Waiting time by priority (Baseline scenario)")
print("="*60)
for p in sorted(baseline['Priority'].unique()):
    subset = baseline[baseline['Priority'] == p]
    print(f"Priority {p}: n={len(subset)}, "
          f"mean={subset['DocWaitTime'].mean():.2f}, "
          f"median={subset['DocWaitTime'].median():.2f}, "
          f"P90={subset['DocWaitTime'].quantile(0.9):.2f}")

# Check: higher priority should have lower mean wait
means = [baseline[baseline['Priority'] == p]['DocWaitTime'].mean()
         for p in sorted(baseline['Priority'].unique())]
print(f"\nMean waits (P1, P2, P3): {[f'{m:.2f}' for m in means]}")
if means[0] > means[1] > means[2]:
    print("CONSISTENT with non-preemptive priority rule: P1 waits more than P2, P2 waits more than P3")
else:
    print("INCONSISTENT: check the priority mapping")

# Plot: box plot of waiting time by priority
plt.figure(figsize=(8, 5))
data_to_plot = [baseline[baseline['Priority'] == p]['DocWaitTime'].values
                for p in sorted(baseline['Priority'].unique())]
plt.boxplot(data_to_plot, tick_labels=['Priority 1', 'Priority 2', 'Priority 3'])
plt.ylabel('Doctor waiting time (minutes)')
plt.title('Waiting time by priority class (Baseline scenario)')
plt.savefig('figures/b6_wait_by_priority.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved plot: figures/b6_wait_by_priority.png")