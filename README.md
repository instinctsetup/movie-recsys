# Movie RecSys - matrix factorization from scratch

A movie recommender with **no ML libraries**: biased matrix factorization
trained with hand-written SGD in NumPy, served behind FastAPI with a UI
where you rate a few movies and instantly fold into the model.

- MovieLens 100k, official u1.base/u1.test split (80k/20k)
- **Test RMSE 0.934** vs 1.154 global-mean baseline (-19%)
- Precision@10 0.143, Recall@10 0.051 (honest top-N numbers, see docs/eval-report.json)
- Cold-start fold-in: rate >=3 movies, get top-10 with
  "because you liked X" explanations from item-factor cosine similarity

## Run it

```bash
pip install numpy fastapi uvicorn pytest
python train.py            # ~20s, retrains and writes recsys/factors.npz
uvicorn server:app --port 8000
# open http://localhost:8000
```

## Test it

```bash
python -m pytest tests/ -q
```

## Layout

    recsys/data.py      MovieLens loader (shared index space across splits)
    recsys/mf.py        biased MF: SGD fit, predict, fold_in
    recsys/evaluate.py  RMSE, Precision@K, Recall@K
    train.py            reproduce the numbers in docs/eval-report.json
    server.py           FastAPI: catalog, fold-in recommend, explanations
    static/index.html   rating UI + recommendations
    docs/MATH.md        every update rule, in plain English
    docs/VIVA.md        viva questions with honest answers

## Honest notes

- Uses the MovieLens 100k dataset (GroupLens, grouplens.org/datasets/movielens).
- Train RMSE (~0.79) is well below test RMSE (0.934): mild overfitting,
  controlled by L2 regularization and lr decay; test error still beats
  every non-factor baseline we tried (global mean 1.154).
- Precision@10 is low in absolute terms - that is typical for explicit
  top-N on this dataset; we report it instead of cherry-picking K.
- A bug we actually hit and fixed: the train/test loaders originally built
  separate id maps, so test indices pointed at different users - test RMSE
  looked worse than the mean baseline. Shared maps fixed it; there is a
  regression test.
