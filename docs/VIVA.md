# Viva Q&A - Movie RecSys

**Q: Why matrix factorization and not k-nearest-neighbors?**
kNN on 943x1682 with ~6% density is dominated by noise: most user pairs
share few co-rated movies, so similarities are unreliable. MF compresses
each user into 24 numbers learned from ALL their ratings, which is far
more robust to sparsity. kNN also doesn't scale: MF scores a whole
catalog with one matrix-vector product.

**Q: Why is treating missing ratings as 0 wrong?**
Missing mostly means "hasn't seen it", not "hates it". If you fill
unknowns with 0, the model learns that everyone dislikes almost
everything and the biases absorb garbage. We optimize only over the
80k observed entries.

**Q: Walk me through one SGD update.**
Say user u gave movie i a 5 but we predict 3.2, so e=1.8. We add a bit of
Q[i] to P[u] (move taste toward this movie's flavor), a bit of P[u] to
Q[i], and bump both biases. The reg term shrinks everything slightly so
no single rating can dominate. Repeat 80k times per epoch, 25 epochs.

**Q: How do you know it's not overfitting?**
Official u1.base/u1.test split. Train RMSE falls to ~0.79 while test RMSE
stays at 0.934 - there IS a gap (mild overfitting, expected), but test
error beat the global-mean baseline (1.154) by 19%. Regularization and
learning-rate decay control the gap; we verified epoch-by-epoch that test
RMSE doesn't climb back past baseline.

**Q: What's the cold-start problem and how does fold-in solve it?**
A new user has no P vector. Full retraining is expensive, so we freeze
the item side and optimize only the new user's vector against their first
few ratings. It's an underdetermined problem regularized toward 0, which
is exactly what you want: few ratings = stay near the average.

**Q: What do the latent factors mean?**
Nothing is imposed. Inspect them and they correlate with human-readable
axes (genre, era, seriousness) because ratings carry that structure.
That's the famous part of the Netflix Prize story: meaning emerges from
pure rating optimization.

**Q: Why is Precision@10 only 0.14?**
Top-N on implicit 1-5 stars is hard: of ~20 test ratings per user, few
are 4+ on movies the model can surface among 1650 items. Published
baselines on this dataset land in the same range. We report it honestly
rather than tuning the definition until it looks good.

**Q: What would you do with more time?**
Implicit-feedback confidence weighting (weighted ALS), time-aware
splitting, and a two-tower neural model - but only after showing they
beat this 30-line SGD on the same split.
