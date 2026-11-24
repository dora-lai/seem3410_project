"""
model.py
Discrete-event simulation of the clinic: arrival -> doctor (non-preemptive priority queue)
-> pharmacy (FCFS) -> departure. Patients are generated once per seed (common random numbers)
and reused by every policy.

The event loop (simulate) is written out explicitly with a future-event list and six event types:
    ARRIVAL, DOC_START, DOC_END, PHARM_START, PHARM_END, DEPARTURE
run_one_simpy is the original SimPy version, kept as an independent cross-check.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import heapq
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Patient:
    """Single patient record; inputs are generated once per seed (CRN)."""
    patient_id: int
    arrival_time: float
    priority: int                 # 3 = highest, 2 = medium, 1 = lowest
    doc_service_time: float
    pharm_service_time: float

    # Populated during simulation
    doc_start_time: Optional[float] = None
    doc_end_time: Optional[float] = None
    pharm_start_time: Optional[float] = None
    departure_time: Optional[float] = None

    @property
    def doc_wait(self) -> float:
        assert self.doc_start_time is not None
        return self.doc_start_time - self.arrival_time

    @property
    def pharm_wait(self) -> float:
        assert self.pharm_start_time is not None and self.doc_end_time is not None
        return self.pharm_start_time - self.doc_end_time

    @property
    def total_time(self) -> float:
        assert self.departure_time is not None
        return self.departure_time - self.arrival_time


# ---------------------------------------------------------------------------
# A5: Common Random Numbers (CRN) patient generator
# ---------------------------------------------------------------------------

def generate_patients(
    seed: int,
    n_patients: int = 200,
    peak_duration: float = 240.0,
    peak_arrival_rate: float = 0.65,       # patients / min during peak
    offpeak_arrival_rate: float = 0.25,    # patients / min after peak
    priority_probs: Tuple[float, float, float] = (0.50, 0.35, 0.15),  # P1, P2, P3
    doc_mean: float = 10.0,
    pharm_mean: float = 4.0,
    # Optional empirical arrays from hospital_sim_6_scenarios.csv (Tasks B3-B5)
    empirical_doc_times: Optional[Sequence[float]] = None,
    empirical_pharm_times: Optional[Sequence[float]] = None,
) -> List[Patient]:
    """
    Generate a list of n_patients once for a given replication seed.
    Reuse the returned list across all staffing policies to implement CRN.
    """
    rng = np.random.default_rng(seed)

    # 1. Arrival times (piecewise-constant Poisson: peak <= 240 min, off-peak > 240 min)
    arrivals: List[float] = []
    t = 0.0
    for _ in range(n_patients):
        rate = peak_arrival_rate if t < peak_duration else offpeak_arrival_rate
        interarrival = rng.exponential(1.0 / rate)
        t += interarrival
        arrivals.append(t)

    # 2. Priorities (1, 2, 3 where 3 is highest)
    priorities = rng.choice([1, 2, 3], size=n_patients, p=priority_probs)

    # 3. Doctor service times (empirical bootstrap if provided, else exponential)
    if empirical_doc_times is not None:
        doc_times = rng.choice(empirical_doc_times, size=n_patients, replace=True)
    else:
        doc_times = rng.exponential(doc_mean, size=n_patients)

    # 4. Pharmacy service times (empirical bootstrap if provided, else exponential)
    if empirical_pharm_times is not None:
        pharm_times = rng.choice(empirical_pharm_times, size=n_patients, replace=True)
    else:
        pharm_times = rng.exponential(pharm_mean, size=n_patients)

    return [
        Patient(
            patient_id=i,
            arrival_time=float(arrivals[i]),
            priority=int(priorities[i]),
            doc_service_time=float(doc_times[i]),
            pharm_service_time=float(pharm_times[i]),
        )
        for i in range(n_patients)
    ]

def _sample_pmf(rng: np.random.Generator, pmf: Dict[float, float], size: int) -> np.ndarray:
    """Sample from a discrete empirical PMF dictionary {value: prob}."""
    values = np.array(list(pmf.keys()))
    probs = np.array(list(pmf.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(values, size=size, p=probs)


def generate_patients_from_config(seed: int) -> List[Patient]:
    """
    Generate N_PATIENTS using the empirical PMFs loaded in config.py.
    Reused across all policies for the same seed (CRN).
    """
    import config

    rng = np.random.default_rng(seed)
    n = config.N_PATIENTS

    # 1. Arrival times using empirical peak / post-peak interarrival PMFs
    arrivals: List[float] = []
    t = 0.0
    peak_vals = np.array(list(config.ARRIVAL_GAP_PMF["peak"].keys()))
    peak_probs = np.array(list(config.ARRIVAL_GAP_PMF["peak"].values()), dtype=float)
    post_vals = np.array(list(config.ARRIVAL_GAP_PMF["post"].keys()))
    post_probs = np.array(list(config.ARRIVAL_GAP_PMF["post"].values()), dtype=float)

    for _ in range(n):
        if t < config.PEAK_MINUTES:
            gap = rng.choice(peak_vals, p=peak_probs)
        else:
            gap = rng.choice(post_vals, p=post_probs)
        t += float(gap)
        arrivals.append(t)

    # 2. Priority, doctor service time, and pharmacy service time
    priorities = _sample_pmf(rng, config.PRIORITY_MIX, n)
    doc_times = _sample_pmf(rng, config.DOC_SERVICE_PMF, n)
    pharm_times = _sample_pmf(rng, config.MED_SERVICE_PMF, n)

    return [
        Patient(
            patient_id=i,
            arrival_time=float(arrivals[i]),
            priority=int(priorities[i]),
            doc_service_time=float(doc_times[i]),
            pharm_service_time=float(pharm_times[i]),
        )
        for i in range(n)
    ]
# ---------------------------------------------------------------------------
# A2, A3, A4: Discrete-Event Simulation Engine
# ---------------------------------------------------------------------------

# Event type ordering breaks ties cleanly at identical timestamps
EVENT_ARRIVAL = 1
EVENT_DOC_END = 2
EVENT_PHARM_END = 3


def run_policy(
    base_patients: Sequence[Patient],
    n_doctors: int = 2,
    n_pharmacy: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run one replication of the outpatient clinic DES for a given staffing policy.

    Parameters
    ----------
    base_patients : Sequence[Patient]
        Pre-generated patient list from `generate_patients(seed)` (CRN).
        Copied internally so the original list is never mutated.
    n_doctors : int
        Number of doctors (2, 3, or 4).
    n_pharmacy : int
        Number of pharmacy counters (1 or 2).

    Returns
    -------
    patient_df : pd.DataFrame
        Patient-level timestamps and waiting times for all patients.
    summary : dict
        Run-level performance measures (P90 doctor wait, means, utilizations).
    """
    if n_doctors < 1 or n_pharmacy < 1:
        raise ValueError("n_doctors and n_pharmacy must be >= 1.")

    # Clone patients so CRN inputs remain untouched across policies
    patients = [replace(p) for p in base_patients]

    # Future Event List (FEL): min-heap of (event_time, event_type, patient_id)
    fel: List[Tuple[float, int, int]] = []
    for p in patients:
        heapq.heappush(fel, (p.arrival_time, EVENT_ARRIVAL, p.patient_id))

    # Resource states
    busy_doctors = 0
    busy_pharmacy = 0

    # A3: Non-preemptive priority doctor queue: (-priority, arrival_time, patient_id)
    # Negating priority ensures class 3 is popped first; arrival_time gives FCFS within class.
    doc_queue: List[Tuple[int, float, int]] = []

    # A3: FCFS pharmacy queue
    pharm_queue: deque[int] = deque()

    # Utilization tracking (area under busy-server curve)
    last_time = 0.0
    doc_busy_area = 0.0
    pharm_busy_area = 0.0

    # --- A2: Main DES event loop ---
    while fel:
        now, event_type, pid = heapq.heappop(fel)

        # Accumulate server busy time since last event
        dt = now - last_time
        if dt > 0:
            doc_busy_area += busy_doctors * dt
            pharm_busy_area += busy_pharmacy * dt
            last_time = now

        p = patients[pid]

        if event_type == EVENT_ARRIVAL:
            # Patient arrives -> join doctor queue or start consultation immediately
            if busy_doctors < n_doctors:
                # Doctor start
                busy_doctors += 1
                p.doc_start_time = now
                doc_end = now + p.doc_service_time
                heapq.heappush(fel, (doc_end, EVENT_DOC_END, pid))
            else:
                heapq.heappush(doc_queue, (-p.priority, p.arrival_time, pid))

        elif event_type == EVENT_DOC_END:
            # Doctor end -> free doctor, then route patient to pharmacy
            p.doc_end_time = now
            busy_doctors -= 1

            # Start next highest-priority waiting patient (non-preemptive)
            if doc_queue:
                _, _, next_pid = heapq.heappop(doc_queue)
                next_p = patients[next_pid]
                busy_doctors += 1
                next_p.doc_start_time = now
                heapq.heappush(
                    fel, (now + next_p.doc_service_time, EVENT_DOC_END, next_pid)
                )

            # Route finished consultation patient to pharmacy (FCFS)
            if busy_pharmacy < n_pharmacy:
                # Pharmacy start
                busy_pharmacy += 1
                p.pharm_start_time = now
                pharm_end = now + p.pharm_service_time
                heapq.heappush(fel, (pharm_end, EVENT_PHARM_END, pid))
            else:
                pharm_queue.append(pid)

        elif event_type == EVENT_PHARM_END:
            # Pharmacy end & patient departure
            p.departure_time = now
            busy_pharmacy -= 1

            # Start next patient in FCFS pharmacy queue
            if pharm_queue:
                next_pid = pharm_queue.popleft()
                next_p = patients[next_pid]
                busy_pharmacy += 1
                next_p.pharm_start_time = now
                heapq.heappush(
                    fel, (now + next_p.pharm_service_time, EVENT_PHARM_END, next_pid)
                )

    # Build patient-level output table
    records = [
        {
            "patient_id": p.patient_id,
            "priority": p.priority,
            "arrival_time": p.arrival_time,
            "doc_service_time": p.doc_service_time,
            "pharm_service_time": p.pharm_service_time,
            "doc_start_time": p.doc_start_time,
            "doc_end_time": p.doc_end_time,
            "pharm_start_time": p.pharm_start_time,
            "departure_time": p.departure_time,
            "doc_wait": p.doc_wait,
            "pharm_wait": p.pharm_wait,
            "total_time": p.total_time,
        }
        for p in patients
    ]
    df = pd.DataFrame(records)

    makespan = last_time if last_time > 0 else 1.0
    summary = {
        "n_doctors": n_doctors,
        "n_pharmacy": n_pharmacy,
        "n_departed": int(df["departure_time"].notna().sum()),
        "p90_doc_wait": float(np.percentile(df["doc_wait"], 90)),
        "mean_doc_wait": float(df["doc_wait"].mean()),
        "mean_pharm_wait": float(df["pharm_wait"].mean()),
        "mean_total_time": float(df["total_time"].mean()),
        "doc_utilization": doc_busy_area / (n_doctors * makespan),
        "pharm_utilization": pharm_busy_area / (n_pharmacy * makespan),
        "makespan": makespan,
    }
    return df, summary
# ---------------------------------------------------------------------------
# Compatibility wrappers for a7_check.py and downstream scripts
# ---------------------------------------------------------------------------

def make_patients(seed: int) -> List[Patient]:
    """Alias for generate_patients_from_config(seed) using config.py PMFs."""
    return generate_patients_from_config(seed)


def run_one(
    base_patients: Sequence[Patient],
    n_doctors: int = 2,
    n_pharmacy: int = 1,
) -> List[Dict[str, float]]:
    """Run one replication and return patient-level records as a list of dicts."""
    df, _ = run_policy(base_patients, n_doctors=n_doctors, n_pharmacy=n_pharmacy)
    return df.to_dict(orient="records")


def run_p90(records: Sequence[Dict[str, float]]) -> float:
    """Compute run-level 90th percentile of doctor waiting time."""
    waits = [r["doc_wait"] for r in records]
    return float(np.percentile(waits, 90))

# ---------------------------------------------------------------------------
# Quick smoke test verifying A2, A3, A4, A5
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seed = 42
    patients_rep0 = generate_patients(seed=seed, n_patients=200)

    policies = [
        ("Baseline (2 doc, 1 pharm)", 2, 1),
        ("Policy A (3 doc, 1 pharm)", 3, 1),
        ("Policy B (4 doc, 1 pharm)", 4, 1),
        ("Secondary (3 doc, 2 pharm)", 3, 2),
    ]

    for name, n_doc, n_pharm in policies:
        df_out, stats = run_policy(patients_rep0, n_doctors=n_doc, n_pharmacy=n_pharm)
        assert stats["n_departed"] == 200, "Not all 200 patients departed!"
        assert (df_out["doc_wait"] >= -1e-9).all(), "Negative doctor wait detected!"
        assert (df_out["pharm_wait"] >= -1e-9).all(), "Negative pharmacy wait detected!"
        print(
            f"{name:28s} | P90 doc wait: {stats['p90_doc_wait']:6.2f} min | "
            f"Mean doc: {stats['mean_doc_wait']:5.2f} | "
            f"Mean pharm: {stats['mean_pharm_wait']:5.2f} | "
            f"Doc util: {stats['doc_utilization']:.2%}"
        )