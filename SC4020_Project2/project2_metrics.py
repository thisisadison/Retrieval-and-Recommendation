"""Shared evaluation for Tasks 2 and 3.

Predicted-score ties are broken uniformly at random *in expectation*, never by
row order. True-score ties count as multiple correct best targets. Ligands with
no variation in true scores are excluded from recommendation/macro-CI metrics.
Pooled CI also compares observations belonging to different ligands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def n_distinct_pairs(x):
    x = np.asarray(x)
    _, counts = np.unique(x, return_counts=True)
    return (len(x) * (len(x) - 1) - np.sum(counts * (counts - 1))) / 2


def concordance_index(y, yhat):
    """Half credit for tied predictions; 0.5 convention for no comparable y."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if y.shape != yhat.shape or not (np.isfinite(y).all() and np.isfinite(yhat).all()):
        raise ValueError("True and predicted scores must be equally shaped and finite.")
    a, b = n_distinct_pairs(y), n_distinct_pairs(yhat)
    if a == 0 or b == 0:
        return 0.5
    tau = kendalltau(y, yhat, variant="b").statistic
    return float(np.clip((tau * np.sqrt(b / a) + 1) / 2, 0, 1))


def _hit_probability(n, n_best, draws):
    """Probability of at least one best item in a uniform sample without replacement."""
    draws = min(max(int(draws), 0), n)
    if draws == 0 or n_best == 0:
        return 0.0
    if draws > n - n_best:
        return 1.0
    return float(1 - np.prod([(n - n_best - i) / (n - i) for i in range(draws)]))


def ranking_metrics(df, ks=(1, 5)):
    """One row per rankable ligand, with exact expected rank and expected hits.

The first predicted-score tie block containing a truly best target determines
the first correct rank. If a rows precede it, m items tie and b are correct,
E[first rank] = a + (m+1)/(b+1). Hit@k is its hypergeometric hit probability.
    """
    if any(int(k) != k or k < 1 for k in ks):
        raise ValueError("Ranking cutoffs must be positive integers.")
    rows = []
    for lig, g in df.groupby("ligand_key", sort=True):
        y, p = g["score"].to_numpy(float), g["pred"].to_numpy(float)
        if not (np.isfinite(y).all() and np.isfinite(p).all()):
            raise ValueError("Ranking requires finite true and predicted scores.")
        if len(np.unique(y)) < 2:
            continue
        best = y == y.max()
        best_prediction = p[best].max()
        above = int(np.sum(p > best_prediction))
        tied = p == best_prediction
        m, b = int(tied.sum()), int(np.sum(tied & best))
        row = {"ligand_key": lig, "n_candidates": len(g),
               "n_true_best": int(best.sum()), "rank_of_best": above + (m + 1) / (b + 1),
               "ci": concordance_index(y, p)}
        for k in ks:
            row[f"hit@{k}"] = _hit_probability(m, b, k - above)
            row[f"random hit@{k}"] = _hit_probability(len(g), int(best.sum()), k)
        rows.append(row)
    columns = ["ligand_key", "n_candidates", "n_true_best", "rank_of_best", "ci"]
    columns += [c for k in ks for c in (f"hit@{k}", f"random hit@{k}")]
    return pd.DataFrame(rows, columns=columns)


def summarize_predictions(df):
    if df.empty:
        raise ValueError("Cannot evaluate an empty partition.")
    rk = ranking_metrics(df)
    y, p = df["score"].to_numpy(float), df["pred"].to_numpy(float)
    return {"MSE": float(np.mean((y - p) ** 2)), "CI": concordance_index(y, p),
            "macro CI": float(rk["ci"].mean()),
            "Hit@1": float(rk["hit@1"].mean()), "Hit@5": float(rk["hit@5"].mean()),
            "random Hit@1": float(rk["random hit@1"].mean()),
            "random Hit@5": float(rk["random hit@5"].mean()),
            "mean rank of best": float(rk["rank_of_best"].mean()),
            "ranked ligands": len(rk),
            "excluded ligands": int(df["ligand_key"].nunique() - len(rk))}


def paired_bootstrap(df, prediction_columns, reference, cluster="ligand_key", n_boot=1000, seed=42):
    """Paired percentile 95% intervals, conditional on this fitted model/split.

Resamples whole clusters with replacement using identical draws for every
model. MSE is pair-weighted within each resample. Ligand clustering also reports
macro CI and expected Hits, excluding unrankable ligands. Target clustering
reports MSE only: resampling targets changes the recommendation candidate set.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive.")
    cols = list(dict.fromkeys([reference, *prediction_columns]))
    keys = pd.Index(sorted(df[cluster].unique()))
    if len(keys) == 0:
        raise ValueError("Cannot bootstrap empty data.")
    counts = df.groupby(cluster).size().reindex(keys).to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    samples, estimates = {}, {}
    for col in cols:
        p = df[col].to_numpy(float)
        y = df["score"].to_numpy(float)
        if not np.isfinite(p).all():
            raise ValueError("Predictions must be finite.")
        sse = df.assign(_sse=(y - p) ** 2).groupby(cluster)["_sse"].sum().reindex(keys).to_numpy()
        samples[(col, "MSE")] = sse[draws].sum(axis=1) / counts[draws].sum(axis=1)
        estimates[(col, "MSE")] = float(sse.sum() / counts.sum())
        if cluster == "ligand_key":
            rk = ranking_metrics(df.assign(pred=p)).set_index("ligand_key").reindex(keys)
            for metric, field in [("macro CI", "ci"), ("Hit@1", "hit@1"), ("Hit@5", "hit@5")]:
                values = rk[field].to_numpy(float)
                valid_counts = np.isfinite(values[draws]).sum(axis=1)
                sums = np.nansum(values[draws], axis=1)
                samples[(col, metric)] = np.divide(sums, valid_counts, out=np.full(n_boot, np.nan), where=valid_counts > 0)
                estimates[(col, metric)] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
    result = []
    metrics = ["MSE"] + (["macro CI", "Hit@1", "Hit@5"] if cluster == "ligand_key" else [])
    for col in cols:
        for metric in metrics:
            sample = samples[(col, metric)]
            delta_sample = sample - samples[(reference, metric)]
            finite = sample[np.isfinite(sample)]
            delta_finite = delta_sample[np.isfinite(delta_sample)]
            lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
            dlo, dhi = np.quantile(delta_finite, [0.025, 0.975]) if len(delta_finite) else (np.nan, np.nan)
            result.append({"model": col, "reference": reference, "metric": metric,
                           "estimate": estimates[(col, metric)], "ci_low": lo, "ci_high": hi,
                           "delta": estimates[(col, metric)] - estimates[(reference, metric)],
                           "delta_ci_low": dlo, "delta_ci_high": dhi, "cluster": cluster,
                           "n_clusters": len(keys), "n_boot": n_boot})
    return pd.DataFrame(result)
