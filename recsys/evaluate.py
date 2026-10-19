"""Recommender metrics from scratch: RMSE, Precision@K, Recall@K."""
import numpy as np


def rmse(model, users, items, ratings) -> float:
    preds = np.array([model.predict_one(u, i) for u, i in zip(users, items)])
    return float(np.sqrt(np.mean((preds - ratings) ** 2)))


def precision_recall_at_k(model, train, test, k=10, relevant=4.0, max_users=300):
    """Per user: recommend top-K among movies they did NOT rate in train;
    count how many of the movies they LIKED in test (rating >= `relevant`)
    appear in that list. Returns (precision@k, recall@k) averaged over
    users with at least one liked test movie."""
    tr_u, tr_i = train[0], train[1]
    te_u, te_i, te_r = test[0], test[1], test[2]
    seen = {}
    for u, i in zip(tr_u, tr_i):
        seen.setdefault(u, set()).add(i)
    likes = {}
    for u, i, r in zip(te_u, te_i, te_r):
        if r >= relevant:
            likes.setdefault(u, set()).add(i)
    precisions, recalls = [], []
    for u, liked in list(likes.items())[:max_users]:
        scores = model.predict_user_all(u)
        for i in seen.get(u, ()):
            scores[i] = -np.inf           # never recommend what they saw
        topk = set(np.argpartition(-scores, k)[:k].tolist())
        hit = len(topk & liked)
        precisions.append(hit / k)
        recalls.append(hit / len(liked))
    return float(np.mean(precisions)), float(np.mean(recalls))
