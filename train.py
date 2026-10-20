"""Train on the official MovieLens u1.base/u1.test split and save factors.

    python train.py     # ~1-2 minutes
"""
import json
import numpy as np
from recsys import data, evaluate, mf

titles = data.load_titles()
train = data.load_ratings("data/u1.base")
test = data.load_ratings("data/u1.test", train[3], train[4])
n_users = max(train[3].values()) + 1
n_items = max(max(train[4].values()), max(test[4].values())) + 1
print(f"users={n_users} items={n_items} train={len(train[2])} test={len(test[2])}")

model = mf.MatrixFactorization(n_users, n_items, k=24)
model.fit(train[0], train[1], train[2], epochs=25)

report = {
    "rmse_test": round(evaluate.rmse(model, test[0], test[1], test[2]), 4),
    "rmse_baseline_global_mean": round(float(np.sqrt(np.mean((test[2] - train[2].mean()) ** 2))), 4),
}
p, r = evaluate.precision_recall_at_k(model, train, test, k=10)
report["precision_at_10"] = round(p, 4)
report["recall_at_10"] = round(r, 4)
print(json.dumps(report, indent=2))

np.savez("recsys/factors.npz", P=model.P, Q=model.Q, bu=model.bu, bi=model.bi, mu=model.mu)
with open("recsys/titles.json", "w") as f:
    json.dump({str(k): v for k, v in data.load_titles().items()}, f)
with open("recsys/id_maps.json", "w") as f:
    json.dump({"users": train[3], "items": train[4]}, f)
with open("docs/eval-report.json", "w") as f:
    json.dump(report, f, indent=2)
print("artifacts saved")
