"""Task 1 data loading, cleaning and deduplication, shared with Tasks 2 and 3.

Pairs with the same canonical SMILES and supplied protein sequence are averaged
(in pKd space for DAVIS). Valid zero/extreme scores are kept and flagged.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(name, data_dir=Path("data")):
    folder = Path(data_dir) / name
    ligands = json.loads((folder / "ligands_can.txt").read_text(encoding="utf-8"))
    proteins = json.loads((folder / "proteins.txt").read_text(encoding="utf-8"))
    # These are the locally supplied benchmark files, not downloaded objects.
    with (folder / "Y").open("rb") as handle:
        scores = np.asarray(pickle.load(handle, encoding="latin1"), dtype=float)
    if scores.shape != (len(ligands), len(proteins)):
        raise ValueError(f"{name}: matrix dimensions do not match the ID dictionaries")
    return ligands, proteins, scores


def to_long(ligands, proteins, scores):
    ligand_ids, target_ids = list(ligands), list(proteins)
    rows, cols = np.where(np.isfinite(scores))
    return pd.DataFrame({
        "ligand_id": [ligand_ids[i] for i in rows],
        "target_id": [target_ids[j] for j in cols],
        "smiles": [ligands[ligand_ids[i]] for i in rows],
        "sequence": [proteins[target_ids[j]] for j in cols],
        "score": scores[rows, cols],
    })


def validate(frame, name):
    frame = frame[np.isfinite(frame["score"])].copy()
    if name.lower() == "davis":
        frame = frame[frame["score"] > 0].copy()
        frame["kd_nM"] = frame["score"]
        frame["score"] = 9 - np.log10(frame["kd_nM"])
    return frame


def flag_records(frame, name):
    frame = frame.copy()
    if name == "davis":
        frame["at_screening_limit"] = np.isclose(frame["score"], 5.0)
        return frame, ["at_screening_limit"]
    q1, q3 = frame["score"].quantile([0.25, 0.75])
    frame["extreme_low"] = frame["score"] < q1 - 3 * (q3 - q1)
    frame["extreme_high"] = frame["score"] > q3 + 3 * (q3 - q1)
    return frame, ["extreme_low", "extreme_high"]


def deduplicate(frame, extra_flags=()):
    """Average comparable labels; retain source IDs, spread and flag fractions.

    Input order and representative IDs (first in file order) match the original notebook.
    """
    ligands = (frame.groupby("smiles", sort=False)["ligand_id"]
               .agg(lambda s: list(dict.fromkeys(s))).rename("ligand_ids").reset_index())
    ligands["ligand_key"] = ligands["ligand_ids"].str[0]
    targets = (frame.groupby("sequence", sort=False)["target_id"]
               .agg(lambda s: list(dict.fromkeys(s))).rename("target_ids").reset_index())
    targets["target_key"] = targets["target_ids"].str[0]
    mapped = frame.merge(ligands[["smiles", "ligand_key"]], on="smiles").merge(
        targets[["sequence", "target_key"]], on="sequence")
    aggregation = {"score": ["mean", "std", "min", "max", "count"]}
    aggregation.update({flag: "mean" for flag in extra_flags})
    pairs = mapped.groupby(["ligand_key", "target_key"], sort=False).agg(aggregation)
    pairs.columns = ["score", "score_std", "score_min", "score_max", "n_merged",
                     *[f"frac_{flag}" for flag in extra_flags]]
    pairs = pairs.reset_index()
    ligands["ligand_ids"] = ligands["ligand_ids"].str.join("|")
    targets["target_ids"] = targets["target_ids"].str.join("|")
    return pairs, ligands, targets


def prepare_data(base_dir=Path(".")):
    """Write data/clean/: validated, flagged and deduplicated pairs plus ID lookups."""
    base_dir = Path(base_dir)
    folder = base_dir / "data" / "clean"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("davis", "kiba"):
        ligands_in, proteins, scores = load_dataset(name, base_dir / "data")
        raw, flags = flag_records(validate(to_long(ligands_in, proteins, scores), name), name)
        pairs, ligands, targets = deduplicate(raw, flags)
        assert not pairs.duplicated(["ligand_key", "target_key"]).any()
        assert int(pairs["n_merged"].sum()) == len(raw)
        pairs.to_csv(folder / f"{name}_pairs.csv", index=False)
        ligands[["ligand_key", "ligand_ids", "smiles"]].to_csv(folder / f"{name}_ligands.csv", index=False)
        targets[["target_key", "target_ids", "sequence"]].to_csv(folder / f"{name}_targets.csv", index=False)
    return folder


def similarity_vs_profile(similarity, matrix, representations, min_overlap, similarity_column="sequence_similarity"):
    """Correlate profiles, removing actual identical input strings (not a score proxy)."""
    ids = similarity.index
    matrix = matrix.loc[:, ids]
    corr = matrix.corr(min_periods=min_overlap)
    finite = matrix.notna().to_numpy(dtype=np.float64)
    overlap = finite.T @ finite
    i, j = np.triu_indices(len(ids), 1)
    rep = np.array([representations[k] for k in ids], dtype=object)
    different = rep[i] != rep[j]
    frame = pd.DataFrame({similarity_column: similarity.to_numpy()[i, j],
        "profile_correlation": corr.to_numpy()[i, j],
        "overlap": overlap[i, j].astype(int), "left_id": ids.to_numpy()[i], "right_id": ids.to_numpy()[j]})
    frame = frame.loc[different].dropna(subset=[similarity_column, "profile_correlation"])
    frame = frame[frame["overlap"] >= min_overlap].copy()
    frame["similarity_bin"] = pd.cut(frame[similarity_column], np.linspace(0, 1, 11), include_lowest=True)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    print("Wrote", prepare_data(parser.parse_args().base_dir))


if __name__ == "__main__":
    main()
