# SC4020 Project 2: DAVIS and KIBA Drug–Target Binding

## Folder structure
- `data/davis/`, `data/kiba/`: original datasets from DeepDTA (Öztürk et al., 2018)
- `data/clean/`: cleaned, deduplicated data created by Task 1 (used by Tasks 2 and 3)
- `figures/`: figures saved by the notebooks for the report
- `saved_models/`: trained CF models from Task 2 (`davis_cf.npz`, `kiba_cf.npz`)
- `task1_analysis.ipynb`: Task 1, data loading, cleaning, visualisation and analysis
- `task2_cf.ipynb`: Task 2, collaborative filtering recommender (lecture 11.3 method)

## How to run
1. Open the `SC4020_Project2` folder in VS Code.
2. Run `task1_analysis.ipynb` first (Run All). It creates `data/clean/` and `figures/`.
3. Run `task2_cf.ipynb` (Run All, about 1–2 minutes). It creates `saved_models/` and more figures.

Required libraries: pandas, numpy, matplotlib, seaborn, scipy (install with `%pip install ...` in a notebook cell if missing).

## Task 1 summary
- DAVIS scores converted to pKd = 9 − log10(Kd); KIBA scores used as supplied (higher = stronger for both).
- Pairs with identical SMILES and sequence merged by averaging their scores.
- Screening-limit (DAVIS pKd = 5) and extreme KIBA values are flagged, not removed.
- Missing KIBA entries are not treated as zero (missing does not mean no binding).

## Task 2 summary
- Ligands = users, targets = items.
- Split: 70 / 15 / 15 with seed 42; every validation/test ligand and target also appears in training.
- Models: global mean, ligand mean, baseline (mu + b_u + b_i), and neighbourhood CF on top of the baseline with centred cosine similarity (lecture 11.3).
- Hyperparameters (item- vs user-based, k) chosen on validation; test set evaluated once.
- Metrics: MSE, Concordance Index, Hit@1, Hit@5.
- Final models: DAVIS item-based k = 10, KIBA user-based k = 10. Reloading the saved models reproduces identical predictions.

## Task 3 summary


## Notes
- `data/kiba/kiba_drug_sim.zip` must be unzipped locally if needed (file too large for GitHub web upload).
- `saved_models/kiba_cf.npz` is about 25 MB; upload it through Git (VS Code Sync), not the GitHub website.
- Random seed 42 is used throughout so results are reproducible.

