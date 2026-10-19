"""Matrix factorization trained with plain SGD - the algorithm that won
the Netflix Prize era. Every update rule is derived in docs/MATH.md.

Model:  r_hat(u, i) = mu + b_u[u] + b_i[i] + P[u] . Q[i]

mu       global mean rating
b_u/b_i  per-user and per-item bias ("some people rate high", "some
         movies are just good")
P, Q     k-dimensional taste vectors - learned latent factors
"""
import numpy as np


class MatrixFactorization:
    def __init__(self, n_users, n_items, k=24, seed=42):
        rng = np.random.default_rng(seed)
        self.P = rng.normal(0, 0.1, (n_users, k))
        self.Q = rng.normal(0, 0.1, (n_items, k))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.mu = 0.0

    def predict_one(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]

    def predict_user_all(self, u):
        """Scores for every item for one user - used for top-N lists."""
        return self.mu + self.bu[u] + self.bi + self.Q @ self.P[u]

    def fit(self, users, items, ratings, epochs=25, lr=0.01, reg=0.05, seed=1):
        """SGD over observed ratings only. We never touch the unknowns -
        missing does not mean 0."""
        self.mu = float(ratings.mean())
        n = len(ratings)
        rng = np.random.default_rng(seed)
        for epoch in range(epochs):
            order = rng.permutation(n)
            err_sum = 0.0
            for idx in order:
                u, i, r = users[idx], items[idx], ratings[idx]
                e = r - self.predict_one(u, i)
                err_sum += e * e
                # Gradient of squared error + L2 regularization, per parameter.
                pu, qi = self.P[u].copy(), self.Q[i].copy()
                self.P[u] += lr * (e * qi - reg * pu)
                self.Q[i] += lr * (e * pu - reg * qi)
                self.bu[u] += lr * (e - reg * self.bu[u])
                self.bi[i] += lr * (e - reg * self.bi[i])
            if epoch % 5 == 4:
                lr *= 0.9
            if epoch % 5 == 0:
                print(f"epoch {epoch:2d}  train RMSE {np.sqrt(err_sum / n):.4f}")
        return self

    def fold_in(self, rated_items: dict[int, float], k_epochs=60, lr=0.02, reg=0.05):
        """A brand-new user rates a few movies; learn ONLY their vector.

        Item factors stay frozen (they encode the catalog), we optimize a
        fresh p_new against them. This is how 'rate 10 movies and get
        personal picks' works in real products - no retraining.
        """
        p = np.zeros(self.P.shape[1])
        b = 0.0
        items = list(rated_items)
        for _ in range(k_epochs):
            for i in items:
                e = rated_items[i] - (self.mu + b + self.bi[i] + p @ self.Q[i])
                p += lr * (e * self.Q[i] - reg * p)
                b += lr * (e - reg * b)
        scores = self.mu + b + self.bi + self.Q @ p
        return scores, p
