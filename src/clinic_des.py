"""
Unified two-stage Discrete-Event Simulation (DES) for an outpatient clinic
(SEEM3410 Weeks 3-4: Next-Event Time Advance, Future Event List, Continuous Time Averages).

Stage 1 (Consultation): c doctors, non-preemptive priority queue (Priority 3 > 2 > 1, FCFS within class).
Stage 2 (Pharmacy):     m counters, FCFS queue.
"""
from __future__ import annotations

import heapq
import itertools
import json
import numpy as np

from config import (
    CALIBRATED_JSON_PATH,
    MASTER_SEED,
    MAX_STREAM_PATIENTS,
    N_PATIENTS,
    N_REPLICATIONS,
    PEAK_MINUTES,
    compute_policy_costs,
)

# Simultaneous-timestamp tie-breaking order in the Future Event List (FEL):
# 0 = PHARM_END (free pharmacy counter)
# 1 = ARRIVAL   (patient enters clinic at time t before freed doctor scans waiting room at time t)
# 2 = DOC_END   (doctor finishes consultation, patient moves to pharmacy, doctor checks waiting room)
PHARM_END = 0
ARRIVAL = 1
DOC_END = 2
EVENT_NAMES = {PHARM_END: "PHARM_END", ARRIVAL: "ARRIVAL", DOC_END: "DOC_END"}


def load_calibrated_inputs(json_path=CALIBRATED_JSON_PATH) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def _to_arrays(pmf_probs: dict, dtype=float):
        items = sorted((dtype(k), float(v)) for k, v in pmf_probs.items())
        vals = np.array([k for k, _ in items], dtype=dtype)
        probs = np.array([v for _, v in items], dtype=float)
        probs /= probs.sum()
        return vals, probs

    pk_vals, pk_probs = _to_arrays(raw["interarrival_peak"]["probs"], float)
    of_vals, of_probs = _to_arrays(raw["interarrival_offpeak"]["probs"], float)
    pr_vals, pr_probs = _to_arrays(raw["priority_class"]["probs"], int)
    ds_vals, ds_probs = _to_arrays(raw["doc_service_time"]["probs"], float)
    ms_vals, ms_probs = _to_arrays(raw["med_service_time"]["probs"], float)

    return {
        "raw": raw,
        "peak_gap": (pk_vals, pk_probs, float(raw["interarrival_peak"]["mean"])),
        "offpeak_gap": (of_vals, of_probs, float(raw["interarrival_offpeak"]["mean"])),
        "priority": (pr_vals, pr_probs),
        "doc_service": (ds_vals, ds_probs),
        "med_service": (ms_vals, ms_probs),
    }


def inverse_discrete_pmf(u: np.ndarray | float, values: np.ndarray, probs: np.ndarray):
    """Inverse-transform sampler mapping U ~ Uniform(0,1) onto discrete support."""
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, u, side="right")
    return values[np.minimum(idx, len(values) - 1)]


def spawn_replication_uniforms(n_replications: int = N_REPLICATIONS, seed: int = MASTER_SEED) -> list[dict]:
    """
    Create matched Common Random Number (CRN) uniform streams for each replication.
    Each replication receives four independent RNG substreams:
      0: interarrival uniforms, 1: priority uniforms, 2: doctor service uniforms, 3: pharmacy uniforms.
    """
    root_ss = np.random.SeedSequence(seed)
    rep_seeds = root_ss.spawn(n_replications)
    streams = []
    for r, r_ss in enumerate(rep_seeds):
        child_ss = r_ss.spawn(4)
        streams.append(
            {
                "rep_id": r + 1,
                "u_arrival": np.random.default_rng(child_ss[0]).random(MAX_STREAM_PATIENTS),
                "u_priority": np.random.default_rng(child_ss[1]).random(MAX_STREAM_PATIENTS),
                "u_doc_service": np.random.default_rng(child_ss[2]).random(MAX_STREAM_PATIENTS),
                "u_med_service": np.random.default_rng(child_ss[3]).random(MAX_STREAM_PATIENTS),
            }
        )
    return streams


def realize_patient_cohort(
    uniforms: dict,
    cal: dict,
    n_patients: int = N_PATIENTS,
    peak_minutes: float = PEAK_MINUTES,
    arrival_mode: str = "empirical",
    arrival_rate_scale: float = 1.0,
    doc_service_scale: float = 1.0,
) -> dict:
    """
    Map CRN uniform substreams into concrete arrival times, priorities, and service durations.
    - arrival_mode='empirical': calibrated discrete gap PMFs (CV ~ 0.26 peak / 0.22 off-peak)
    - arrival_mode='poisson':   piecewise exponential gaps with identical mean gaps (CV = 1.0)
    """
    if n_patients > MAX_STREAM_PATIENTS:
        raise ValueError(f"n_patients={n_patients} exceeds MAX_STREAM_PATIENTS={MAX_STREAM_PATIENTS}")

    pk_vals, pk_probs, pk_mean = cal["peak_gap"]
    of_vals, of_probs, of_mean = cal["offpeak_gap"]
    pr_vals, pr_probs = cal["priority"]
    ds_vals, ds_probs = cal["doc_service"]
    ms_vals, ms_probs = cal["med_service"]

    u_arr = uniforms["u_arrival"][:n_patients]
    u_pri = uniforms["u_priority"][:n_patients]
    u_doc = uniforms["u_doc_service"][:n_patients]
    u_med = uniforms["u_med_service"][:n_patients]

    arrival_times = np.zeros(n_patients, dtype=float)
    clock = 0.0
    for i in range(n_patients):
        in_peak = clock < peak_minutes
        if arrival_mode == "empirical":
            vals, probs = (pk_vals, pk_probs) if in_peak else (of_vals, of_probs)
            base_gap = float(inverse_discrete_pmf(u_arr[i], vals, probs))
        elif arrival_mode == "poisson":
            mean_gap = pk_mean if in_peak else of_mean
            base_gap = float(-mean_gap * np.log(1.0 - u_arr[i]))
        else:
            raise ValueError(f"Unknown arrival_mode: {arrival_mode}")
        clock += base_gap / arrival_rate_scale
        arrival_times[i] = clock

    priorities = inverse_discrete_pmf(u_pri, pr_vals, pr_probs).astype(int)
    doc_services = inverse_discrete_pmf(u_doc, ds_vals, ds_probs).astype(float) * doc_service_scale
    med_services = inverse_discrete_pmf(u_med, ms_vals, ms_probs).astype(float)

    return {
        "arrival_times": arrival_times,
        "priorities": priorities,
        "doc_services": doc_services,
        "med_services": med_services,
    }


def simulate_clinic(
    arrival_times: np.ndarray,
    priorities: np.ndarray,
    doc_services: np.ndarray,
    med_services: np.ndarray,
    n_doctors: int,
    n_pharmacy: int = 1,
    peak_minutes: float = PEAK_MINUTES,
    trajectory_grid: np.ndarray | None = None,
    trace_limit: int = 0,
) -> dict:
    """
    Run one replication of the unified two-stage outpatient clinic DES from empty initial
    state until all N arriving patients complete both consultation and pharmacy dispensing.
    """
    n = len(arrival_times)
    clock = 0.0
    seq = itertools.count()

    # Future Event List: (event_time, event_type, patient_tie_key, patient_idx, seq_id)
    fel: list[tuple[float, int, int, int, int]] = []
    if n > 0:
        heapq.heappush(fel, (float(arrival_times[0]), ARRIVAL, 0, 0, next(seq)))

    idle_doctors = list(range(1, n_doctors + 1))
    heapq.heapify(idle_doctors)
    idle_pharm = n_pharmacy

    # Queues:
    # Doctor priority queue: (-priority, arrival_time, patient_idx) -> highest priority first, FCFS within class
    # Pharmacy FCFS queue:   (enter_pharm_time, patient_idx, patient_idx)
    doc_queue: list[tuple[int, float, int]] = []
    pharm_queue: list[tuple[float, int, int]] = []

    # Continuous-time state integral accumulators (Week 4)
    area_doc_queue = 0.0
    area_doc_busy = 0.0
    area_pharm_queue = 0.0
    area_pharm_busy = 0.0

    # Patient-level output arrays
    assigned_doc = np.zeros(n, dtype=int)
    doc_wait = np.zeros(n, dtype=float)
    med_wait = np.zeros(n, dtype=float)
    departure_time = np.zeros(n, dtype=float)

    # Optional trajectory recording on fixed time grid
    traj_q_doc = None
    traj_idx = 0
    if trajectory_grid is not None:
        traj_q_doc = np.zeros(len(trajectory_grid), dtype=float)

    trace_rows = []

    while fel:
        t, ev_type, _, pid, _ = heapq.heappop(fel)

        # Record queue length on any grid points in [clock, t) before state changes at time t
        if traj_q_doc is not None:
            while traj_idx < len(trajectory_grid) and trajectory_grid[traj_idx] < t:
                traj_q_doc[traj_idx] = len(doc_queue)
                traj_idx += 1

        # Advance simulation clock and accumulate continuous time integrals
        dt = t - clock
        if dt > 0.0:
            area_doc_queue += len(doc_queue) * dt
            area_doc_busy += (n_doctors - len(idle_doctors)) * dt
            area_pharm_queue += len(pharm_queue) * dt
            area_pharm_busy += (n_pharmacy - idle_pharm) * dt
            clock = t

        if ev_type == PHARM_END:
            if pharm_queue:
                enter_t, _, next_pid = heapq.heappop(pharm_queue)
                med_wait[next_pid] = clock - enter_t
                end_t = clock + float(med_services[next_pid])
                departure_time[next_pid] = end_t
                heapq.heappush(fel, (end_t, PHARM_END, next_pid, next_pid, next(seq)))
            else:
                idle_pharm += 1

        elif ev_type == ARRIVAL:
            # Dynamically schedule next patient arrival (Week 4 single-arrival chain)
            if pid + 1 < n:
                nxt = pid + 1
                heapq.heappush(fel, (float(arrival_times[nxt]), ARRIVAL, nxt, nxt, next(seq)))

            if idle_doctors:
                doc_id = heapq.heappop(idle_doctors)
                assigned_doc[pid] = doc_id
                doc_wait[pid] = 0.0
                end_t = clock + float(doc_services[pid])
                heapq.heappush(fel, (end_t, DOC_END, pid, pid, next(seq)))
            else:
                heapq.heappush(doc_queue, (-int(priorities[pid]), clock, pid))

        elif ev_type == DOC_END:
            doc_id = assigned_doc[pid]
            # Stage 1 -> Stage 2 transition: patient immediately enters pharmacy stage
            if idle_pharm > 0:
                idle_pharm -= 1
                med_wait[pid] = 0.0
                end_t = clock + float(med_services[pid])
                departure_time[pid] = end_t
                heapq.heappush(fel, (end_t, PHARM_END, pid, pid, next(seq)))
            else:
                heapq.heappush(pharm_queue, (clock, pid, pid))

            # Freed doctor selects highest-priority waiting patient (non-preemptive)
            if doc_queue:
                _, arr_t, next_pid = heapq.heappop(doc_queue)
                assigned_doc[next_pid] = doc_id
                doc_wait[next_pid] = clock - arr_t
                end_t = clock + float(doc_services[next_pid])
                heapq.heappush(fel, (end_t, DOC_END, next_pid, next_pid, next(seq)))
            else:
                heapq.heappush(idle_doctors, doc_id)

        if trace_limit and len(trace_rows) < trace_limit:
            trace_rows.append(
                {
                    "clock": clock,
                    "event": EVENT_NAMES[ev_type],
                    "patient_id": pid + 1,
                    "doc_busy": n_doctors - len(idle_doctors),
                    "doc_queue_len": len(doc_queue),
                    "pharm_busy": n_pharmacy - idle_pharm,
                    "pharm_queue_len": len(pharm_queue),
                }
            )

    # After simulation ends, system is empty (len(doc_queue) == 0 for remaining grid points)
    total_time = departure_time - arrival_times
    makespan = clock

    # Priority-class and arrival-period subsets
    peak_mask = arrival_times <= peak_minutes
    offpeak_mask = ~peak_mask

    def _safe_q90(arr: np.ndarray) -> float:
        return float(np.quantile(arr, 0.90)) if len(arr) > 0 else 0.0

    def _safe_mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    costs = compute_policy_costs(n_doctors=n_doctors, n_pharmacy=n_pharmacy, makespan_minutes=makespan)

    summary = {
        "n_patients": n,
        "n_doctors": n_doctors,
        "n_pharmacy": n_pharmacy,
        "p90_doc_wait": _safe_q90(doc_wait),
        "p95_doc_wait": float(np.quantile(doc_wait, 0.95)) if n > 0 else 0.0,
        "median_doc_wait": float(np.median(doc_wait)) if n > 0 else 0.0,
        "mean_doc_wait": _safe_mean(doc_wait),
        "p90_prio1_wait": _safe_q90(doc_wait[priorities == 1]),
        "p90_prio2_wait": _safe_q90(doc_wait[priorities == 2]),
        "p90_prio3_wait": _safe_q90(doc_wait[priorities == 3]),
        "mean_prio1_wait": _safe_mean(doc_wait[priorities == 1]),
        "mean_prio2_wait": _safe_mean(doc_wait[priorities == 2]),
        "mean_prio3_wait": _safe_mean(doc_wait[priorities == 3]),
        "p90_peak_wait": _safe_q90(doc_wait[peak_mask]),
        "p90_offpeak_wait": _safe_q90(doc_wait[offpeak_mask]),
        "mean_peak_wait": _safe_mean(doc_wait[peak_mask]),
        "mean_offpeak_wait": _safe_mean(doc_wait[offpeak_mask]),
        "mean_med_wait": _safe_mean(med_wait),
        "p90_med_wait": _safe_q90(med_wait),
        "mean_total_time": _safe_mean(total_time),
        "p90_total_time": _safe_q90(total_time),
        "last_arrival_time": float(arrival_times[-1]) if n > 0 else 0.0,
        "makespan": makespan,
        "doc_utilization": area_doc_busy / (n_doctors * makespan) if makespan > 0 else 0.0,
        "pharm_utilization": area_pharm_busy / (n_pharmacy * makespan) if makespan > 0 else 0.0,
        "time_avg_doc_queue": area_doc_queue / makespan if makespan > 0 else 0.0,
        "time_avg_pharm_queue": area_pharm_queue / makespan if makespan > 0 else 0.0,
        "area_doc_queue": area_doc_queue,
        "area_pharm_queue": area_pharm_queue,
        **costs,
    }

    return {
        "summary": summary,
        "patients": {
            "assigned_doc": assigned_doc,
            "doc_wait": doc_wait,
            "med_wait": med_wait,
            "departure_time": departure_time,
            "total_time": total_time,
        },
        "trajectory_q_doc": traj_q_doc,
        "trace": trace_rows,
    }
