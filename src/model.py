"""
model.py
Discrete-event simulation model 
(start by writing the simplest version to ensure it runs).
"""

import simpy


class Clinic:
    """Outpatient Clinic Simulation"""
    
    def __init__(self, env, n_doctors, n_pharmacy):
        self.env = env
        self.doctors = simpy.Resource(env, capacity=n_doctors)
        self.pharmacy = simpy.Resource(env, capacity=n_pharmacy)
        self.records = []
    
    def patient(self, pid):
        """Workflow for an individual patient (left blank for now; Tung will fill in the actual logic later)"""
        arrive = self.env.now
        self.records.append({
            'pid': pid,
            'arrive': arrive,
        })
        yield self.env.timeout(0)
    
    def run(self, n_patients):
        """Generate n_patients patients"""
        for i in range(n_patients):
            yield self.env.timeout(0.1)
            self.env.process(self.patient(i))


def run_simulation(n_doctors, n_pharmacy, n_patients):
    """Run a simulation and return the log."""
    env = simpy.Environment()
    clinic = Clinic(env, n_doctors, n_pharmacy)
    env.process(clinic.run(n_patients))
    env.run()
    return clinic.records


if __name__ == '__main__':
    records = run_simulation(n_doctors=2, n_pharmacy=1, n_patients=200)
    print(f"Finish, {len(records)} patients overall.")
    print(f"First: {records[0]}")
    print(f"Last: {records[-1]}")