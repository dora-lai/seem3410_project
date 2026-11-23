"""A7: compare model P90 against dataset staffing scenarios."""
import model
from config import SEED_BASE

dataset_p90 = {2: 370.0, 3: 8.0, 4: 0.0}
for n in [2, 3, 4]:
    pts = model.make_patients(SEED_BASE)
    rec = model.run_one(pts, n_doctors=n, n_pharmacy=1)
    p90 = model.run_p90(rec)
    print(f"{n} doctors: model P90 = {p90:.1f} min, dataset P90 = {dataset_p90[n]:.1f} min")