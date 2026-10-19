import numpy as np
from recsys import data, evaluate, mf


def small_model():
    m = mf.MatrixFactorization(5, 6, k=4, seed=0)
    return m


def test_data_files_share_index_space():
    train = data.load_ratings("data/u1.base")
    test = data.load_ratings("data/u1.test", train[3], train[4])
    assert max(test[0]) <= max(train[0])          # same users in u1 split
    assert max(test[1]) <= max(max(train[4].values()), max(test[4].values()))
    assert len(train[2]) == 80000 and len(test[2]) == 20000


def test_predict_shapes():
    m = small_model()
    assert isinstance(m.predict_one(0, 0), float)
    assert m.predict_user_all(0).shape == (6,)


def test_fit_reduces_train_error():
    rng = np.random.default_rng(0)
    m = small_model()
    users = rng.integers(0, 5, 400)
    items = rng.integers(0, 6, 400)
    ratings = np.clip(3 + m.P[users] @ m.Q[items].T.diagonal() * 0, 1, 5)  # arbitrary signal
    ratings = (users % 5 + 1).astype(float)     # learnable pattern: user id -> rating
    before = m.predict_one(0, 0)
    m.fit(users, items, ratings, epochs=15, lr=0.05)
    after = np.mean([abs(m.predict_one(u, i) - r) for u, i, r in zip(users[:50], items[:50], ratings[:50])])
    assert after < 0.5 and m.predict_one(0, 0) != before


def test_fold_in_picks_up_obvious_taste():
    # Two camps: users who love items 0-2 and hate 3-5, and vice versa.
    m = mf.MatrixFactorization(20, 6, k=2, seed=1)
    users, items, ratings = [], [], []
    for u in range(20):
        camp = u % 2
        for i in range(6):
            like = (i < 3) == (camp == 0)
            users.append(u); items.append(i); ratings.append(5.0 if like else 1.0)
    m.fit(np.array(users), np.array(items), np.array(ratings), epochs=20, lr=0.05)
    scores, _ = m.fold_in({0: 5.0, 1: 5.0, 2: 5.0})
    assert scores[2] > scores[4]                 # camp-0 item beats camp-1 item


def test_metrics_perfect_model():
    m = small_model()
    # Make item i score exactly i so ranking is deterministic.
    m.mu, m.bu[:] = 0.0, 0.0
    m.bi = np.arange(6, dtype=float)
    m.Q[:] = 0.0; m.P[:] = 0.0
    train = (np.array([0, 0]), np.array([0, 1]), np.array([1.0, 1.0]), {0: 0}, {0: 0, 1: 1})
    test = (np.array([0, 0]), np.array([4, 5]), np.array([5.0, 5.0]))
    p, r = evaluate.precision_recall_at_k(m, train, test, k=2)
    assert p == 1.0 and r == 1.0                 # top-2 unseen are exactly the liked ones


def test_rmse_global_mean_baseline():
    ratings = np.array([1.0, 2.0, 4.0, 5.0])
    m = small_model()
    m.mu, m.bu[:], m.bi[:], m.P[:], m.Q[:] = ratings.mean(), 0, 0, 0, 0
    r = evaluate.rmse(m, np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]), ratings)
    assert abs(r - np.sqrt(np.mean((ratings - ratings.mean()) ** 2))) < 1e-9
