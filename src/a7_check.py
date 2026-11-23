"""
a7_check.py  (task A7)
Consistency check: compare the model's results with the dataset's own staffing scenarios.
Both sides use the same measure: mean over runs of the run-level P90 of doctor waiting time.
Run after estimate_inputs.py:   python src/a7_check.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import model
from config import SEED_BASE, N_REPLICATIONS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, 'data', 'hospital_sim_6_scenarios.csv'))

# Check these names against the list printed below; edit if they differ.
SCENARIOS = {2: 'Baseline_2Doc_240Peak', 3: 'Staffing_3Doc', 4: 'Staffing_4Doc'}
print('Scenarios in the file:', list(df['Scenario'].unique()))
for name in SCENARIOS.values():
    assert name in set(df['Scenario']), f'Scenario {name!r} not found; fix SCENARIOS above'


def mean_ci(x):
    """Mean and half-width of a 95% t confidence interval."""
    x = np.asarray(x, dtype=float)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), half


rows = []
for n_doc, name in SCENARIOS.items():
    d = df[df['Scenario'] == name]
    ds_p90 = d.groupby('RunID')['DocWaitTime'].apply(lambda s: np.percentile(s, 90))   # one value per run
    ds_mean = d.groupby('RunID')['DocWaitTime'].mean()

    sim_p90, sim_mean = [], []
    for r in range(N_REPLICATIONS):                                  # same seeds for every policy
        rec = model.run_one(model.make_patients(SEED_BASE + r), n_doctors=n_doc, n_pharmacy=1)
        waits = [x['doc_wait'] for x in rec]
        sim_p90.append(model.run_p90(rec))
        sim_mean.append(np.mean(waits))

    dp, dh = mean_ci(ds_p90); sp, sh = mean_ci(sim_p90)
    dm, _ = mean_ci(ds_mean); sm, _ = mean_ci(sim_mean)
    rows.append({'doctors': n_doc, 'dataset_P90': dp, 'dataset_P90_halfCI': dh, 'model_P90': sp, 'model_P90_halfCI': sh,
                 'dataset_mean_wait': dm, 'model_mean_wait': sm, 'dataset_runs': len(ds_p90), 'model_runs': len(sim_p90)})

out = pd.DataFrame(rows).round(2)
print(out.to_string(index=False))
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
out.to_csv(os.path.join(ROOT, 'results', 'a7_comparison.csv'), index=False)
print('Saved results/a7_comparison.csv')
