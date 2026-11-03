"""Cold-target experiments and inference for a previously unseen protein sequence.

The extension was motivated by the original test error analysis. All settings
below are selected on validation labels; repeated splits are robustness checks,
not independent external confirmation. Missing affinities are never zero-filled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project2_metrics import summarize_predictions, paired_bootstrap

SEEDS = (42, 7, 2026)
KS = (1, 2, 3, 5, 10, 20, 40)
GLOBAL_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
RAMPS = tuple((lo, hi) for lo in (0.0, 0.1, 0.2, 0.3)
              for hi in (0.3, 0.5, 0.75, 1.0) if hi > lo)
METHODS = ("global", "fallback", "neighbours", "global_shrinkage", "similarity_shrinkage")


def kmer_set(sequence, k=3):
    return {sequence[i:i + k] for i in range(max(0, len(sequence) - k + 1))}


def sequence_similarity(targets, k=3):
    """Exact set Jaccard. Vocabulary uses sequences only, never affinity labels."""
    if "target_key" in targets.columns:
        targets = targets.set_index("target_key")
    sets = [kmer_set(s, k) for s in targets["sequence"]]
    vocab = {frag: i for i, frag in enumerate(sorted(set().union(*sets)))}
    binary = np.zeros((len(sets), len(vocab)), dtype=float)
    for i, fragments in enumerate(sets):
        binary[i, [vocab[f] for f in fragments]] = 1.0
    inter = binary @ binary.T
    size = binary.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return pd.DataFrame(sim, index=targets.index, columns=targets.index)


def cold_target_split(df, seed=42, fracs=(0.70, 0.15, 0.15)):
    """Split *targets* 70/15/15; every pair of a val/test target is held out.

    Targets were deduplicated by sequence in Task 1, so no cold target has an
    exact sequence copy in training. Pairs whose ligand has no training pair are dropped.
    """
    if len(fracs) != 3 or not np.isclose(sum(fracs), 1) or min(fracs) <= 0:
        raise ValueError("fracs must contain three positive fractions summing to 1")
    keys = np.array(sorted(df["target_key"].unique()))
    order = np.random.default_rng(seed).permutation(len(keys))
    n_tr, n_va = int(fracs[0] * len(keys)), int(fracs[1] * len(keys))
    if n_tr == 0 or n_va == 0 or n_tr + n_va == len(keys):
        raise ValueError("Not enough targets for nonempty train/validation/test sets")
    role = {keys[idx]: "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
            for i, idx in enumerate(order)}
    out = df.assign(split=df["target_key"].map(role)).copy()
    training_ligands = set(out.loc[out["split"] == "train", "ligand_key"])
    unknown = (out["split"] != "train") & ~out["ligand_key"].isin(training_ligands)
    dropped = int(unknown.sum())
    return out.loc[~unknown].reset_index(drop=True), role, dropped


def fit(train):
    if train.empty or train.duplicated(["ligand_key", "target_key"]).any():
        raise ValueError("Training pairs must be nonempty and unique")
    mu = float(train["score"].mean())
    return {"mu": mu,
            "b_u": train.groupby("ligand_key")["score"].mean() - mu,
            "R": train.pivot(index="ligand_key", columns="target_key", values="score")}


def neighbour_components(model, df, similarity, k=5):
    """Return fallback, neighbour estimate, count and nearest *observed* similarity."""
    if k < 1:
        raise ValueError("k must be positive")
    unknown = set(df["ligand_key"]) - set(model["b_u"].index)
    if unknown:
        raise ValueError(f"Unsupported ligand IDs (no training measurements): {sorted(unknown)[:5]}")
    frame = df.reset_index(drop=True)
    base = model["mu"] + frame["ligand_key"].map(model["b_u"]).to_numpy()
    pred, used, nearest = base.copy(), np.zeros(len(frame), int), np.zeros(len(frame))
    for ligand, group in frame.groupby("ligand_key", sort=False):
        scores = model["R"].loc[ligand].dropna()
        if scores.empty:
            continue
        sims = similarity.loc[group["target_key"].to_numpy(), scores.index].to_numpy().copy()
        sims = np.maximum(sims, 0)
        rows = group.index.to_numpy()
        nearest[rows] = sims.max(axis=1)
        if sims.shape[1] > k:
            cut = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            keep = np.zeros_like(sims, dtype=bool)
            np.put_along_axis(keep, cut, True, axis=1)
            sims = np.where(keep, sims, 0)
        total = sims.sum(axis=1)
        ok = total > 0
        deviations = scores.to_numpy() - (model["mu"] + model["b_u"].loc[ligand])
        pred[rows[ok]] += (sims[ok] @ deviations) / total[ok]
        used[rows] = (sims > 0).sum(axis=1)
    return base, pred, used, nearest


def confidence_weights(nearest, method, alpha=1.0, low=0.0, high=1.0):
    if method == "global_shrinkage":
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        return np.full(len(nearest), alpha)
    if method == "similarity_shrinkage":
        if not 0 <= low < high <= 1:
            raise ValueError("Require 0 <= low < high <= 1")
        return np.clip((nearest - low) / (high - low), 0, 1)
    if method == "neighbours":
        return np.ones(len(nearest))
    raise ValueError(f"Unknown neighbour method: {method}")


def predict(model, df, method="neighbours", S=None, k=5, alpha=1.0, low=0.0, high=1.0):
    """Original notebook-compatible prediction interface, with optional shrinkage."""
    base = model["mu"] + df["ligand_key"].map(model["b_u"]).to_numpy()
    if np.isnan(base).any():
        raise ValueError("Unsupported ligand ID: no training measurements")
    if method == "global":
        return np.full(len(df), model["mu"]), np.zeros(len(df), int)
    if method == "fallback":
        return base, np.zeros(len(df), int)
    if S is None:
        raise ValueError("Sequence similarities are required")
    base, raw, used, nearest = neighbour_components(model, df, S, k)
    weight = confidence_weights(nearest, method, alpha, low, high)
    return base + weight * (raw - base), used


def select_parameters(model, validation, similarity):
    """MSE-only selection, never consults test data. Grid fixed before experiments."""
    y = validation["score"].to_numpy()
    records = []
    for k in KS:
        base, raw, used, nearest = neighbour_components(model, validation, similarity, k)
        variants = [("neighbours", 1.0, 0.0, 1.0)]
        variants += [("global_shrinkage", alpha, 0.0, 1.0) for alpha in GLOBAL_WEIGHTS]
        variants += [("similarity_shrinkage", 1.0, lo, hi) for lo, hi in RAMPS]
        for method, alpha, low, high in variants:
            weight = confidence_weights(nearest, method, alpha, low, high)
            error = float(np.mean((y - base - weight * (raw - base)) ** 2))
            records.append(dict(method=method, k=k, alpha=alpha, low=low, high=high, val_MSE=error))
    search = pd.DataFrame(records)
    selected = {}
    for method in ("neighbours", "global_shrinkage", "similarity_shrinkage"):
        row = search.loc[search["method"] == method].sort_values("val_MSE", kind="stable").iloc[0]
        selected[method] = {"method": method, "k": int(row["k"]),
                            "alpha": float(row["alpha"]), "low": float(row["low"]), "high": float(row["high"])}
    return selected, search


def save_model(model, path, training_sequences, parameters, dataset, seed):
    """Store only training affinities/sequences; no pickled Python objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = model["R"].columns
    seq = training_sequences["sequence"] if isinstance(training_sequences, pd.DataFrame) else training_sequences
    np.savez_compressed(path, mu=model["mu"],
                        ligands=model["R"].index.to_numpy(dtype=str), targets=ids.to_numpy(dtype=str),
                        R=model["R"].to_numpy(), b_u=model["b_u"].reindex(model["R"].index).to_numpy(),
                        sequences=seq.loc[ids].to_numpy(dtype=str),
                        metadata=json.dumps({"parameters": parameters, "dataset": dataset, "seed": int(seed),
                                             "kmer_length": 3, "score_scale": "pKd" if dataset == "davis" else "KIBA",
                                             "version": 1}))


def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        ids, ligands = z["targets"].astype(str), z["ligands"].astype(str)
        return {"mu": float(z["mu"]), "b_u": pd.Series(z["b_u"], index=ligands),
                "R": pd.DataFrame(z["R"], index=ligands, columns=ids),
                "sequences": pd.Series(z["sequences"].astype(str), index=ids),
                "metadata": json.loads(str(z["metadata"]))}


def predict_new_target(model_or_path, sequence, ligand_ids):
    """Predict a supplied protein sequence without needing a pre-existing target ID.

    Example: predict_new_target('saved_models/davis_cold.npz', sequence, ['11338033']).
    Ligands must have training measurements; cold-ligand prediction is a different task.
    """
    model = load_model(model_or_path) if isinstance(model_or_path, (str, Path)) else model_or_path
    if not isinstance(sequence, str):
        raise ValueError("sequence must be a protein string")
    sequence = "".join(sequence.split()).upper()
    if len(sequence) < 3 or not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWYBXZJUO"):
        raise ValueError("Supply at least three valid amino-acid letters")
    ligand_ids = [str(x) for x in ligand_ids]
    query = kmer_set(sequence)
    scores = []
    for seq in model["sequences"]:
        other = kmer_set(seq)
        union = query | other
        scores.append(len(query & other) / len(union) if union else 0.0)
    similarity = pd.DataFrame([scores], index=["__new_target__"], columns=model["sequences"].index)
    frame = pd.DataFrame({"ligand_key": ligand_ids, "target_key": "__new_target__"})
    params = model["metadata"]["parameters"]
    base, raw, used, nearest = neighbour_components(model, frame, similarity, params["k"])
    weight = confidence_weights(nearest, params["method"], params["alpha"], params["low"], params["high"])
    return pd.DataFrame({"ligand_key": ligand_ids, "prediction": base + weight * (raw - base),
                         "fallback": base, "n_neighbours": used, "nearest_observed_similarity": nearest,
                         "confidence_weight": weight})


def _similarity_breakdown(frame, columns, confidence_column, dataset, seed):
    bins = pd.cut(frame[confidence_column], [-1e-12, 0.15, 0.25, 0.4, 1.0], include_lowest=True)
    rows = []
    for band, group in frame.groupby(bins, observed=True):
        for method in columns:
            rows.append({"dataset": dataset, "seed": seed, "similarity_definition": confidence_column,
                         "similarity_band": str(band), "method": method,
                         "targets": group["target_key"].nunique(), "pairs": len(group),
                         "MSE": float(np.mean((group["score"] - group[method]) ** 2))})
    return rows


def run_cold_experiments(clean_dir="data/clean", output_dir="results/cold", figure_dir="figures",
                         model_dir="saved_models", seeds=SEEDS, n_boot=1000):
    clean_dir, output_dir, figure_dir, model_dir = map(Path, (clean_dir, output_dir, figure_dir, model_dir))
    for folder in (output_dir, figure_dir, model_dir):
        folder.mkdir(parents=True, exist_ok=True)
    protocol = {"seeds": list(seeds), "k_grid": list(KS), "global_weights": list(GLOBAL_WEIGHTS),
                "similarity_ramps": list(RAMPS), "selection_metric": "validation MSE",
                "similarity_confidence": "max similarity among this ligand's observed training targets",
                "cold_split_unit": "target (targets deduplicated by sequence in Task 1)",
                "bootstrap_replicates": n_boot,
                "interpretation": "Exploratory extension motivated by prior test analysis; repeated splits are robustness checks.",
                "data_dir": str(clean_dir)}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    evaluations, selections, searches, intervals, breakdowns, split_counts, reloads = [], [], [], [], [], [], []
    for dataset in ("davis", "kiba"):
        pairs = pd.read_csv(clean_dir / f"{dataset}_pairs.csv", dtype={"ligand_key": str, "target_key": str})
        targets = pd.read_csv(clean_dir / f"{dataset}_targets.csv", dtype=str).set_index("target_key")
        similarity = sequence_similarity(targets)
        for seed in seeds:
            print(f"Cold-target {dataset}, seed {seed}: validation selection", flush=True)
            split, roles, dropped = cold_target_split(pairs, seed)
            split[["ligand_key", "target_key", "split"]].to_csv(output_dir / f"{dataset}_seed{seed}_splits.csv.gz", index=False)
            train, validation, test = [split.loc[split["split"] == part].copy() for part in ("train", "val", "test")]
            for part, frame in (("train", train), ("val", validation), ("test", test)):
                split_counts.append({"dataset": dataset, "seed": seed, "split": part, "pairs": len(frame),
                                     "targets": frame["target_key"].nunique(),
                                     "dropped_unseen_ligand_pairs": dropped})
            model = fit(train)
            selected, search = select_parameters(model, validation, similarity)
            searches.append(search.assign(dataset=dataset, seed=seed))
            for method, params in selected.items():
                val_score = search.loc[search["method"] == method, "val_MSE"].min()
                selections.append({"dataset": dataset, "seed": seed, **params, "val_MSE": val_score})
            scored = test[["ligand_key", "target_key", "score"]].copy()
            for method in METHODS:
                params = selected.get(method, {"method": method})
                pred, used = predict(model, test, S=similarity, **params)
                scored[method] = pred
                metrics = summarize_predictions(scored.assign(pred=pred))
                evaluations.append({"dataset": dataset, "seed": seed, "method": method, **metrics,
                                    "no_neighbour_pct": float(100 * np.mean(used == 0)) if method not in ("global", "fallback") else np.nan})
            _, _, _, nearest = neighbour_components(model, test, similarity, selected["neighbours"]["k"])
            scored["nearest_observed_similarity"] = nearest
            scored["nearest_training_similarity"] = test["target_key"].map(similarity.loc[:, model["R"].columns].max(axis=1)).to_numpy()
            scored.to_csv(output_dir / f"{dataset}_seed{seed}_predictions.csv.gz", index=False)
            for confidence_column in ("nearest_observed_similarity", "nearest_training_similarity"):
                breakdowns.extend(_similarity_breakdown(scored, METHODS[1:], confidence_column, dataset, seed))
            # MSE uncertainty resamples whole cold targets (all pairs of a target together).
            table = paired_bootstrap(scored, list(METHODS[1:]), "neighbours", cluster="target_key",
                                     n_boot=n_boot, seed=seed)
            intervals.append(table.assign(dataset=dataset, seed=seed))
            if seed != 42:
                print(f"Cold-target {dataset}, seed {seed}: complete", flush=True)
                continue
            # Only the seed-42 model is saved; check that reloading and the new-sequence
            # interface reproduce the batch predictions exactly.
            checkpoint = model_dir / f"{dataset}_cold.npz"
            save_model(model, checkpoint, targets, selected["similarity_shrinkage"], dataset, seed)
            restored = load_model(checkpoint)
            restored_sim = sequence_similarity(pd.concat([
                pd.DataFrame({"sequence": restored["sequences"]}),
                targets.loc[test["target_key"].unique(), ["sequence"]]]))
            after, _ = predict(restored, test, S=restored_sim, **selected["similarity_shrinkage"])
            before = scored["similarity_shrinkage"].to_numpy()
            if not np.allclose(before, after, atol=1e-12, rtol=0):
                raise AssertionError("Checkpoint reload changed predictions")
            # The direct sequence interface is checked on one genuinely held-out target.
            key = test.iloc[0]["target_key"]
            query = test.loc[test["target_key"] == key].head(5)
            direct = predict_new_target(restored, targets.loc[key, "sequence"], query["ligand_key"].tolist())
            expected, _ = predict(restored, query, S=restored_sim, **selected["similarity_shrinkage"])
            if not np.allclose(direct["prediction"], expected, atol=1e-12, rtol=0):
                raise AssertionError("New-sequence interface changed predictions")
            reloads.append({"dataset": dataset, "seed": seed, "checkpoint": str(checkpoint),
                            "reload_max_absolute_error": float(np.max(np.abs(before - after))),
                            "new_sequence_max_absolute_error": float(np.max(np.abs(direct["prediction"] - expected)))})
            print(f"Cold-target {dataset}, seed {seed}: complete", flush=True)
    tables = {"test_metrics": pd.DataFrame(evaluations), "selected_parameters": pd.DataFrame(selections),
              "validation_search": pd.concat(searches, ignore_index=True),
              "paired_bootstrap": pd.concat(intervals, ignore_index=True), "similarity_breakdown": pd.DataFrame(breakdowns),
              "split_counts": pd.DataFrame(split_counts), "reload_checks": pd.DataFrame(reloads)}
    tables["seed_summary"] = tables["test_metrics"].groupby(["dataset", "method"])[["MSE", "Hit@1", "Hit@5", "macro CI"]].agg(["mean", "std"])
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=name == "seed_summary")
    _plot_results(tables, figure_dir)
    return tables


def _plot_results(tables, figure_dir):
    import matplotlib.pyplot as plt
    methods = list(METHODS[1:])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, dataset in zip(axes, ("davis", "kiba")):
        data = tables["test_metrics"].query("dataset == @dataset and method != 'global'")
        for seed, group in data.groupby("seed"):
            ax.plot(range(len(methods)), group.set_index("method").loc[methods, "MSE"], marker="o", label=f"seed {seed}")
        ax.set_xticks(range(len(methods)), ["Fallback", "Sequence", "Global\nshrinkage", "Similarity\nshrinkage"])
        ax.set_ylabel("Cold-target test MSE (lower is better)")
        ax.set_title(dataset.upper())
        ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "cold_robustness.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bands = tables["similarity_breakdown"]
    for ax, dataset in zip(axes, ("davis", "kiba")):
        subset = bands.query("dataset == @dataset and seed == 42 and similarity_definition == 'nearest_observed_similarity'")
        for method, group in subset.groupby("method", sort=False):
            ax.plot(group["similarity_band"], group["MSE"], marker="o", label=method.replace("_", " "))
        ax.set_xlabel("Nearest observed training-target similarity")
        ax.set_ylabel("Test MSE")
        ax.set_title(dataset.upper() + ": seed 42")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "cold_similarity_bands.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--output-dir", default="results/cold")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--model-dir", default="saved_models")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    run_cold_experiments(**vars(args))


if __name__ == "__main__":
    main()
