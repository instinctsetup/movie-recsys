"""FastAPI server: rate a few movies, fold yourself into the trained model,
get personal top-10s with 'because you liked X' explanations."""
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from recsys.mf import MatrixFactorization

ROOT = Path(__file__).parent
z = np.load(ROOT / "recsys" / "factors.npz")
titles = {int(k): v for k, v in json.load(open(ROOT / "recsys" / "titles.json")).items()}
maps = json.load(open(ROOT / "recsys" / "id_maps.json"))
i_map = {int(k): v for k, v in maps["items"].items()}      # MovieLens id -> internal
internal_to_ml = {v: k for k, v in i_map.items()}
report = json.load(open(ROOT / "docs" / "eval-report.json"))

model = MatrixFactorization(z["P"].shape[0], z["Q"].shape[0], k=z["P"].shape[1])
model.P, model.Q, model.bu, model.bi, model.mu = z["P"], z["Q"], z["bu"], z["bi"], float(z["mu"])

# Normalized item factors for "similar to what you liked" explanations.
Qn = model.Q / (np.linalg.norm(model.Q, axis=1, keepdims=True) + 1e-9)

# Rating UI shows the most-rated movies so the user recognizes them.
# Counts come from the actual ratings file, not the id map.
from recsys import data as _data
_tr = _data.load_ratings("data/u1.base")      # same file/order as train.py -> identical maps
_internal_popular = np.argsort(-np.bincount(_tr[1], minlength=model.Q.shape[0]))
POPULAR = [int(internal_to_ml[i]) for i in _internal_popular if int(i) in internal_to_ml]

app = FastAPI(title="Movie RecSys")


class Ratings(BaseModel):
    ratings: dict[int, float]          # MovieLens item id -> 1..5


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/catalog")
def catalog():
    return {"movies": [{"id": m, **titles.get(m, {"title": str(m), "year": ""})} for m in POPULAR[:60]],
            "eval": report}


@app.post("/api/recommend")
def recommend(body: Ratings):
    rated = {i_map[m]: float(r) for m, r in body.ratings.items() if m in i_map}
    if len(rated) < 3:
        return {"error": "rate at least 3 movies so your taste vector is meaningful"}
    scores, _ = model.fold_in(rated)
    for i in rated:
        scores[i] = -np.inf              # never recommend what you already rated
    top = np.argpartition(-scores, 10)[:10]
    top = top[np.argsort(-scores[top])]

    liked = [i for i, r in rated.items() if r >= 4.0]
    recs = []
    for j in top:
        why = None
        if liked:
            sims = Qn[j] @ Qn[liked].T   # cosine to each liked item's factor
            why = liked[int(np.argmax(sims))]
        ml = internal_to_ml[int(j)]
        recs.append({"id": ml, **titles.get(ml, {"title": str(ml), "year": ""}),
                     "score": round(float(scores[j]), 2),
                     "because": titles.get(internal_to_ml.get(why, -1), {}).get("title") if why is not None else None})
    return {"recommendations": recs, "eval": report}
