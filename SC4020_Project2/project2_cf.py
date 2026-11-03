"""Neighbourhood CF (Task 2): split, fit, predict, select on validation, save/load.

The original notebook formulas and seed-42 pair split are preserved. Model
selection uses only validation MSE; test partitions never select settings.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from project2_metrics import summarize_predictions

KS = (2, 3, 5, 10, 20, 40, 80)
SCALES = {
    "davis": {"score_name": "pKd", "units": "pKd = 9 - log10(Kd [nM])", "screening_limit": 5.0},
    "kiba": {"score_name": "KIBA score", "units": "supplied KIBA scale (higher = stronger)", "screening_limit": None},
}

def build_index(df):
    ligands = np.array(sorted(df["ligand_key"].unique()))
    targets = np.array(sorted(df["target_key"].unique()))
    return ligands, targets

def to_idx(keys, vocab):
    pos = {k: i for i, k in enumerate(vocab)}
    keys = list(keys)
    missing = sorted(set(keys) - set(pos))
    if missing:
        raise ValueError(f"Unknown IDs (this model only supports training IDs): {missing[:5]}")
    return np.array([pos[k] for k in keys], dtype=int)

def fit_baseline(l, t, y, n_l, n_t):
    # slides 18-22
    mu = y.mean()
    b_l = np.bincount(l, y, n_l) / np.maximum(np.bincount(l, minlength=n_l), 1) - mu
    b_t = np.bincount(t, y, n_t) / np.maximum(np.bincount(t, minlength=n_t), 1) - mu
    return mu, b_l, b_t

def centred_cosine(R):
    # slide 9
    C = np.nan_to_num(R - np.nanmean(R, axis=1, keepdims=True))
    norms = np.linalg.norm(C, axis=1)
    # An effectively constant profile has no informative centred direction.
    # Floating-point mean/subtraction can otherwise turn ~1e-15 residuals into
    # large cosine similarities after normalization (observed in sparse KIBA).
    scale = np.maximum(1.0, np.max(np.nan_to_num(np.abs(R)), axis=1))
    count = np.sum(np.isfinite(R), axis=1)
    floor = 32 * np.finfo(float).eps * scale * np.sqrt(count)
    constant = norms <= floor
    C[constant] = 0.0
    norms[constant] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(np.outer(norms, norms) > 0, (C @ C.T) / np.outer(norms, norms), 0.0)
    np.fill_diagonal(S, 0.0)                         # an entity is not its own neighbour
    return S

def fit_cf(train, ligands, targets, method="item", k=10):
    if train.empty or not np.isfinite(train["score"].to_numpy()).all():
        raise ValueError("Training data must contain finite observations.")
    if train.duplicated(["ligand_key", "target_key"]).any():
        raise ValueError("Duplicate ligand-target pairs must be resolved before fitting.")
    if method not in {"global", "ligand_mean", "baseline", "item", "user"}:
        raise ValueError(f"Unknown CF method: {method}")
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer.")
    l = to_idx(train["ligand_key"], ligands)
    t = to_idx(train["target_key"], targets)
    y = train["score"].values
    mu, b_l, b_t = fit_baseline(l, t, y, len(ligands), len(targets))
    R = np.full((len(ligands), len(targets)), np.nan)    
    R[l, t] = y
    E = R - (mu + b_l[:, None] + b_t[None, :])           # deviations from the baseline
    S = None
    if method in ("item", "user"):
        S = centred_cosine(R.T if method == "item" else R)
    return {"method": method, "k": k, "mu": mu, "b_l": b_l, "b_t": b_t,
            "E": E, "S": S, "ligands": ligands, "targets": targets}

def predict(model, ligand_keys, target_keys):
    l = to_idx(ligand_keys, model["ligands"])
    t = to_idx(target_keys, model["targets"])
    base = model["mu"] + model["b_l"][l] + model["b_t"][t]
    method = model["method"]
    if method == "global":
        return np.full(len(l), model["mu"]), np.zeros(len(l), int)
    if method == "ligand_mean":
        return model["mu"] + model["b_l"][l], np.zeros(len(l), int)
    if method == "baseline":
        return base, np.zeros(len(l), int)

    E, S, k = model["E"], model["S"], int(model["k"])
    query, other = (t, l) if method == "item" else (l, t)
    Eq = E if method == "item" else E.T              # rows indexed by `other`, columns by neighbour entity
    pred, n_used = base.copy(), np.zeros(len(l), int)
    for o in np.unique(other):
        rows = np.where(other == o)[0]
        rated = np.where(~np.isnan(Eq[o]))[0]
        if len(rated) == 0:
            continue
        sims = S[np.ix_(query[rows], rated)]         
        sims = np.where(sims > 0, sims, 0.0)
        if len(rated) > k:                            # keep top-k neighbours per query
            cut = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            keep = np.zeros_like(sims, dtype=bool)
            np.put_along_axis(keep, cut, True, axis=1)
            sims = np.where(keep, sims, 0.0)
        wsum = sims.sum(axis=1)
        ok = wsum > 0
        pred[rows[ok]] += (sims[ok] @ Eq[o, rated]) / wsum[ok]
        n_used[rows] = (sims > 0).sum(axis=1)
    return pred, n_used


def split_pairs(df, seed=42, fracs=(0.70, 0.15, 0.15)):
    """Random 70/15/15 pair split; evaluation pairs with an ID unseen in training move to train."""
    if len(fracs) != 3 or not np.isclose(sum(fracs), 1) or min(fracs) <= 0:
        raise ValueError("Three positive split fractions must sum to one.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_tr, n_va = int(fracs[0] * len(df)), int(fracs[1] * len(df))
    split = np.empty(len(df), dtype=object)
    split[order[:n_tr]] = "train"
    split[order[n_tr:n_tr + n_va]] = "val"
    split[order[n_tr + n_va:]] = "test"
    out = df.assign(split=split)
    moved = 0
    while True:
        tr = out[out["split"] == "train"]
        unseen = (out["split"] != "train") & (~out["ligand_key"].isin(tr["ligand_key"]) |
                                               ~out["target_key"].isin(tr["target_key"]))
        if not unseen.any():
            break
        moved += int(unseen.sum())
        out.loc[unseen, "split"] = "train"
    validate_split(out)
    return out, moved


def validate_split(df):
    if df.duplicated(["ligand_key", "target_key"]).any():
        raise ValueError("Duplicate ligand-target keys in split input.")
    if set(df["split"]) != {"train", "val", "test"}:
        raise ValueError("All three partitions must be nonempty.")
    tr, ev = df[df["split"] == "train"], df[df["split"] != "train"]
    if not (ev["ligand_key"].isin(tr["ligand_key"]).all() and ev["target_key"].isin(tr["target_key"]).all()):
        raise ValueError("Evaluation IDs must be represented in training.")


def evaluate(model, df):
    pred, n_used = predict(model, df["ligand_key"], df["target_key"])
    scored = df.assign(pred=pred, n_neighbours=n_used)
    out = summarize_predictions(scored)
    if model["method"] in ("item", "user"):
        out["fallback %"] = float(100 * np.mean(n_used == 0))
    return out, scored


def select_model(split_df, ks=KS):
    """Fit only training observations and choose direction/k by validation MSE."""
    validate_split(split_df)
    tr, val = split_df[split_df["split"] == "train"], split_df[split_df["split"] == "val"]
    ligands, targets = build_index(split_df)
    rows, fitted = [], {}
    for method in ("item", "user"):
        model = fit_cf(tr, ligands, targets, method)
        fitted[method] = model
        for k in ks:
            model["k"] = int(k)
            pred, _ = predict(model, val["ligand_key"], val["target_key"])
            rows.append({"method": method, "k": int(k), "val MSE": float(np.mean((val["score"].to_numpy() - pred) ** 2))})
    search = pd.DataFrame(rows).sort_values("val MSE", kind="stable").reset_index(drop=True)
    best = search.iloc[0]
    model = fitted[best["method"]].copy()
    model["k"] = int(best["k"])
    return model, search


def save_model(model, path, scale, seed=42):
    """Save numeric/string arrays only, so loading never requires pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, method=model["method"], k=model["k"], mu=model["mu"],
                        b_l=model["b_l"], b_t=model["b_t"], E=model["E"],
                        S=model["S"] if model["S"] is not None else np.empty((0, 0)),
                        ligands=np.asarray(model["ligands"], dtype=str),
                        targets=np.asarray(model["targets"], dtype=str),
                        scale=json.dumps(scale), seed=int(seed), format_version=2)


def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        model = {key: z[key] for key in ["b_l", "b_t", "E", "S", "ligands", "targets"]}
        model.update(method=str(z["method"]), k=int(z["k"]), mu=float(z["mu"]),
                     scale=json.loads(str(z["scale"])), seed=int(z["seed"]))
    if model["S"].size == 0:
        model["S"] = None
    return model
