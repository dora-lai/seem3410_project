"""
estimate_inputs.py  (replaces the separate B3/B4/B5 scripts for the model inputs)
Reads the baseline scenario and writes data/inputs.json, which config.py loads.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project folder (src/ is inside it)
df = pd.read_csv(os.path.join(ROOT, 'data', 'hospital_sim_6_scenarios.csv'))
b = df[df['Scenario'] == 'Baseline_2Doc_240Peak'].sort_values(['RunID', 'ArrivalTime']).copy()
b['IsPeakArrival'] = b['IsPeakArrival'].astype(int)

# ---- B3: interarrival gaps, separately for peak and post-peak arrivals ----
b['gap'] = b.groupby('RunID')['ArrivalTime'].diff()          # first patient of each run has no gap
g = b.groupby('IsPeakArrival')['gap'].agg(['mean', 'std', 'count'])
g['rate_per_hour'] = 60 / g['mean']
g['CV'] = g['std'] / g['mean']                               # CV near 1 supports an exponential model
print(g)
print('\nMost common gap values (peak):\n', b[b['IsPeakArrival'] == 1]['gap'].value_counts().head(8))

rate_peak = 1 / g.loc[1, 'mean']
rate_post = 1 / g.loc[0, 'mean'] if 0 in g.index else rate_peak   # fall back if no post-peak arrivals
if 0 not in g.index:
    print('WARNING: no post-peak arrivals found; using the peak rate for both periods')

# ---- B4: priority mix and its link to service time ----
mix = b['Priority'].value_counts(normalize=True).sort_index()
print('\nPriority mix:\n', mix)
print('\nDoctor service time by priority (rows sum to 1):\n',
      pd.crosstab(b['Priority'], b['DocServiceTime'], normalize='index').round(3))

# ---- B5: service-time distributions (empirical tables) ----
doc = b['DocServiceTime'].value_counts(normalize=True).sort_index()
med = b['MedServiceTime'].value_counts(normalize=True).sort_index()
print('\nDoctor service time pmf:\n', doc)
print('\nPharmacy service time pmf:\n', med)
if b['DocServiceTime'].nunique() > 30 or b['MedServiceTime'].nunique() > 30:
    print('NOTE: many distinct service-time values; consider fitting a continuous distribution instead')

# ---- figures for the report (bar charts, because the values are discrete) ----
os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
for name, ser, label in (('doctor', doc, 'Doctor service time (minutes)'), ('pharmacy', med, 'Pharmacy service time (minutes)')):
    plt.figure(figsize=(6, 3.5))
    plt.bar([str(int(v)) if float(v).is_integer() else str(v) for v in ser.index], ser.values, edgecolor='black')
    plt.xlabel(label); plt.ylabel('Proportion of patients')
    plt.title(f'Empirical {name} service-time distribution (baseline scenario)')
    plt.savefig(os.path.join(ROOT, 'figures', f'b5_{name}_pmf.png'), dpi=150, bbox_inches='tight'); plt.close()

# ---- arrival model: empirical interarrival-gap tables (gaps are regular, CV well below 1, so NOT exponential) ----
gap_pmf = {}
for flag, label in ((1, 'peak'), (0, 'post')):
    sg = b[b['IsPeakArrival'] == flag]['gap'].dropna()
    if len(sg):
        gap_pmf[label] = {str(k): float(v) for k, v in sg.value_counts(normalize=True).sort_index().items()}
if 'post' not in gap_pmf:
    gap_pmf['post'] = gap_pmf['peak']
if b['gap'].nunique() > 40:
    print('NOTE: many distinct gap values; consider a continuous distribution instead of a table')

out = {
    'arrival_gap_pmf': gap_pmf,
    'arrival_rate_per_min': {'peak': float(rate_peak), 'post': float(rate_post)},
    'priority_mix': {str(k): float(v) for k, v in mix.items()},
    'doc_service_pmf': {str(k): float(v) for k, v in doc.items()},
    'med_service_pmf': {str(k): float(v) for k, v in med.items()},
}
with open(os.path.join(ROOT, 'data', 'inputs.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('\nSaved', os.path.join(ROOT, 'data', 'inputs.json'))
