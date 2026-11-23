import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('data/hospital_sim_6_scenarios.csv')
baseline = df[df['Scenario'] == 'Baseline_2Doc_240Peak']

# ============================================================
# Doctor Service Time
# ============================================================
doc_service = baseline['DocServiceTime'].values
print("="*60)
print("Doctor Service Time (DocServiceTime)")
print("="*60)
print(f"Mean: {doc_service.mean():.2f}")
print(f"Std: {doc_service.std():.2f}")
print(f"Min: {doc_service.min():.2f}")
print(f"Max: {doc_service.max():.2f}")
print(f"Median: {np.median(doc_service):.2f}")

# Shapiro-Wilk normality test (inspect p-value trend on a sample for large datasets)
sample = np.random.choice(doc_service, 5000, replace=False)
stat, p = stats.shapiro(sample)
print(f"Shapiro-Wilk p-value: {p:.4f} ({'Normal' if p > 0.05 else 'Non-normal'})")

# Plot histogram + Normal fit
plt.figure(figsize=(10, 4))
plt.hist(doc_service, bins=50, density=True, edgecolor='black', alpha=0.7)
x = np.linspace(doc_service.min(), doc_service.max(), 100)
plt.plot(x, stats.norm.pdf(x, doc_service.mean(), doc_service.std()), 'r-', lw=2, label='Normal fit')
plt.xlabel('Doctor service time (minutes)')
plt.ylabel('Density')
plt.title('Doctor service time distribution')
plt.legend()
plt.savefig('figures/b5_doctor_service_hist.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved plot: figures/b5_doctor_service_hist.png")

# ============================================================
# Pharmacy Service Time
# ============================================================
med_service = baseline['MedServiceTime'].values
print("\n" + "="*60)
print("Pharmacy Service Time(MedServiceTime)")
print("="*60)
print(f"Mean: {med_service.mean():.2f}")
print(f"Std: {med_service.std():.2f}")
print(f"Min: {med_service.min():.2f}")
print(f"Max: {med_service.max():.2f}")
print(f"Median: {np.median(med_service):.2f}")

sample = np.random.choice(med_service, 5000, replace=False)
stat, p = stats.shapiro(sample)
print(f"Shapiro-Wilk p-value: {p:.4f} ({'Normal' if p > 0.05 else 'Non-normal'})")

plt.figure(figsize=(10, 4))
plt.hist(med_service, bins=50, density=True, edgecolor='black', alpha=0.7)
x = np.linspace(med_service.min(), med_service.max(), 100)
plt.plot(x, stats.norm.pdf(x, med_service.mean(), med_service.std()), 'r-', lw=2, label='Normal fit')
plt.xlabel('Pharmacy service time (minutes)')
plt.ylabel('Density')
plt.title('Pharmacy service time distribution')
plt.legend()
plt.savefig('figures/b5_pharmacy_service_hist.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved plot: figures/b5_pharmacy_service_hist.png")