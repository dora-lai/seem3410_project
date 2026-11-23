"""
test_model_vs_replay.py  (task A6)
Runs the simpy model and the independent heap-based replay on the SAME generated patients
and checks that the doctor waits agree. Run after estimate_inputs.py.
"""
import pandas as pd
from config import SEED_BASE
import model
from b6_replay import replay

for n_doc in (2, 3, 4):
    worst = 0.0
    for seed in range(SEED_BASE, SEED_BASE + 5):
        pts = model.make_patients(seed)
        rec = model.run_one(pts, n_doctors=n_doc, n_pharmacy=1)
        assert len(rec) == len(pts), 'not every patient departed'
        df = pd.DataFrame(pts).rename(columns={'pid': 'PatientID', 'arrival': 'ArrivalTime',
                                               'priority': 'Priority', 'doc_service': 'DocServiceTime'})
        w = replay(df, k=n_doc)
        ref = pd.Series(w.values, index=df.loc[w.index, 'PatientID'].values)
        sim = pd.DataFrame(rec).set_index('pid')['doc_wait']
        assert (sim >= -1e-9).all(), 'negative wait'
        worst = max(worst, float((sim - ref.reindex(sim.index)).abs().max()))
    print(f'{n_doc} doctors: largest difference between model and replay = {worst:.6f} minutes')
