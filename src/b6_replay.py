"""
b6_replay.py  Replay a non-preemptive priority queue on the dataset's baseline rows and
compare with the recorded DocWaitTime.
"""
import heapq
import os
import numpy as np
import pandas as pd


def replay(run, k=2):
    """Return replayed doctor waits (aligned to run's index) for k doctors, class 3 first."""
    a = run.sort_values(['ArrivalTime', 'PatientID'])
    idx = a.index.to_numpy()
    arr = a['ArrivalTime'].to_numpy(float); pri = a['Priority'].to_numpy(); svc = a['DocServiceTime'].to_numpy(float)
    n = len(a); waits = np.zeros(n); free = [0.0] * k; heap = []; i = 0; clock = 0.0
    while i < n or heap:
        t = max(min(free), clock)            # time never goes backwards (matters for simultaneous arrivals)
        if not heap:
            t = max(t, arr[i])
        clock = t
        while i < n and arr[i] <= t:
            heapq.heappush(heap, (-pri[i], arr[i], i)); i += 1
        _, _, j = heapq.heappop(heap)
        waits[j] = t - arr[j]
        free[free.index(min(free))] = t + svc[j]
    return pd.Series(waits, index=idx)


if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(ROOT, 'data', 'hospital_sim_6_scenarios.csv'))
    base = df[df['Scenario'] == 'Baseline_2Doc_240Peak']
    diffs = []
    for rid, run in base.groupby('RunID'):
        w = replay(run, k=2)
        diffs.append((w - run.loc[w.index, 'DocWaitTime']).abs())
    d = pd.concat(diffs)
    print(f'patients: {len(d)}')
    print(f'exact matches (|diff| < 1e-6): {(d < 1e-6).mean():.1%}')
    print(f'largest difference: {d.max():.3f} minutes')
