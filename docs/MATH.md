# The math, in plain English

## The idea

Write every rating as a sum of four parts:

    r_hat(u, i) = mu + b_u[u] + b_i[i] + P[u] . Q[i]

- `mu` - the average rating overall (3.53 on MovieLens 100k).
- `b_u[u]` - how much user u tends to rate above or below average. Some
  people hand out 5s; some people think 3 is generous.
- `b_i[i]` - how good movie i is *to everyone*. The Shawshank Redemption
  gets a positive bump no matter who you are.
- `P[u] . Q[i]` - the interesting part. Every user gets a vector P[u] of
  k=24 numbers (their taste), every movie a vector Q[i] (its flavor).
  The dot product is large when taste matches flavor.

## Learning: stochastic gradient descent

We only look at ratings that exist - a missing rating is not a 0.

For each observed rating r, the error is `e = r - r_hat`. We want
parameters that make squared error small, so we take the gradient of
`e^2 + reg * (||P||^2 + ||Q||^2 + ||b||^2)` and step downhill:

    P[u] += lr * (e * Q[i] - reg * P[u])
    Q[i] += lr * (e * P[u] - reg * Q[i])
    b_u  += lr * (e - reg * b_u)
    b_i  += lr * (e - reg * b_i)

The `- reg * ...` term is L2 regularization: it shrinks every parameter a
little each step so the model can't memorize noise. `lr` (learning rate)
decays by 10% every 5 epochs - big steps early, small careful steps later.

Why does a dot product represent taste? Because SGD nudges P[u] toward
the Q[i] of movies u rated highly and away from ones they rated poorly.
After enough passes, users with similar histories end up with similar
vectors, and so do movies liked by the same people. Nobody tells the
model what "dimension 7" means - it just emerges (often genre-like).

## Why not just average?

Predicting the global mean for everything gives test RMSE 1.154.
Per-user and per-movie biases alone get you to about 0.98. The 24 latent
factors take it to 0.934 (see docs/eval-report.json). The gap between the
bias-only model and the full model is exactly what "collaborative
filtering" adds: using other people's ratings to predict yours.

## Fold-in: a new user without retraining

When you rate movies in the app, the catalog's Q and biases stay frozen
and we learn only a fresh vector p for you:

    for each of your ratings (i, r):
        e = r - (mu + b + b_i[i] + p . Q[i])
        p += lr * (e * Q[i] - reg * p)
        b += lr * (e - reg * b)

60 tiny passes over your handful of ratings - milliseconds, not minutes.
This is how real products personalize for new users.

## The "because you liked X" explanation

Your recommendations are scored by `Q @ p`. To explain one, we take the
movie you rated 4+ whose normalized factor vector is most similar
(cosine) to the recommended movie's: the recommendation inherits its
score largely from that direction in taste space.
