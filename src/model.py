"""
model.py
Discrete-event simulation of the clinic: arrival -> doctor (non-preemptive priority queue)
-> pharmacy (FCFS) -> departure. Patients are generated once per seed (common random numbers)
and reused by every policy.
"""
import numpy as np
import simpy
from config import *


_GAPS = {period: (np.array(list(pmf.keys())), np.array(list(pmf.values())))
         for period, pmf in ARRIVAL_GAP_PMF.items()}


def make_patients(seed):
    """Generate the patient list for one replication. Same seed -> same list for every policy."""
    rng = np.random.default_rng(seed)
    classes = np.array(list(PRIORITY_MIX.keys()))
    class_p = np.array(list(PRIORITY_MIX.values()))
    doc_v = np.array(list(DOC_SERVICE_PMF.keys())); doc_p = np.array(list(DOC_SERVICE_PMF.values()))
    med_v = np.array(list(MED_SERVICE_PMF.keys())); med_p = np.array(list(MED_SERVICE_PMF.values()))
    t, patients = 0.0, []
    for pid in range(N_PATIENTS):
        gap_v, gap_p = _GAPS['peak' if t < PEAK_MINUTES else 'post']
        t += float(rng.choice(gap_v, p=gap_p))          # empirical interarrival gap (minutes)
        patients.append({
            'pid': pid,
            'arrival': t,
            'is_peak': t < PEAK_MINUTES,
            'priority': int(rng.choice(classes, p=class_p)),
            'doc_service': float(rng.choice(doc_v, p=doc_p)),
            'med_service': float(rng.choice(med_v, p=med_p)),
        })
    return patients


def run_one(patients, n_doctors, n_pharmacy):
    """Run one replication for one policy. Returns one record per patient."""
    env = simpy.Environment()
    doctors = simpy.PriorityResource(env, capacity=n_doctors)   # non-preemptive; lower number served first
    pharmacy = simpy.Resource(env, capacity=n_pharmacy)         # FCFS
    records = []

    def patient(p):
        yield env.timeout(p['arrival'])
        with doctors.request(priority=3 - p['priority']) as req:   # class 3 -> 0 (highest)
            yield req
            doc_start = env.now
            yield env.timeout(p['doc_service'])
        with pharmacy.request() as req:
            yield req
            med_start = env.now
            yield env.timeout(p['med_service'])
        records.append({
            'pid': p['pid'], 'priority': p['priority'], 'is_peak': p['is_peak'],
            'arrival': p['arrival'],
            'doc_wait': doc_start - p['arrival'],
            'med_wait': med_start - (doc_start + p['doc_service']),
            'total_time': env.now - p['arrival'],
            'doc_service': p['doc_service'], 'med_service': p['med_service'],
        })

    for p in patients:
        env.process(patient(p))
    env.run()                                   # ends when the last patient leaves
    return records


def run_p90(records):
    """Run-level P90 of doctor waiting time over all patients in the run."""
    return float(np.percentile([r['doc_wait'] for r in records], 90))


if __name__ == '__main__':
    pts = make_patients(SEED_BASE)
    rec = run_one(pts, n_doctors=2, n_pharmacy=1)
    print(f'{len(rec)} patients finished; run-level P90 doctor wait = {run_p90(rec):.1f} min')
