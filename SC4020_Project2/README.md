# SC4020 Project 2: DAVIS and KIBA Drug–Target Binding

## Folder structure
- `data/davis/`, `data/kiba/`: original datasets from DeepDTA provided in the slides
- `data/clean/`: cleaned, deduplicated data created by Task 1 (used by Tasks 2 and 3)
- `figures/`: figures saved by the notebooks for the report
- `saved_models/`: trained CF models from Task 2 (`davis_cf.npz`, `kiba_cf.npz`)
- `task1_analysis.ipynb`: Task 1, data loading, cleaning, visualisation and analysis
- `task2_cf.ipynb`: Task 2, collaborative filtering recommender using lecture 11.3 method
- `task3_cold_target.ipynb`: Task 3, cold-target generalisation using sequence similarity

## How to run
1. Open the `SC4020_Project2` folder in VS Code.
2. Run `task1_analysis.ipynb` first (Run All). It creates `data/clean/` and `figures/`.
3. Run `task2_cf.ipynb` (Run All). It creates `saved_models/` and more figures.
4. Run `task3_cold_target.ipynb` (Run All). It saves its figures to `figures/`.

## Requirements
Running the first cell of `task1_analysis.ipynb` installs all required libraries (pandas, numpy, matplotlib, seaborn, scipy). Run Task 1 first.

## Task 1 summary
- DAVIS scores converted to pKd = 9 − log10(Kd). KIBA scores are used as supplied (higher = stronger for both).
- Pairs with identical SMILES and sequence merged by averaging their scores.
- Screening-limit (DAVIS pKd = 5) and extreme KIBA values are flagged instead of being removed.
- Missing KIBA entries are not treated as zero (missing does not mean no binding).

## Task 2 summary
- Ligands = users, targets = items.
- Split: 70 / 15 / 15 with seed 42, where every validation/test ligand and target also appears in training.
- Models: global mean, ligand mean, baseline (mu + b_u + b_i) and neighbourhood CF on top of the baseline with centred cosine similarity used in lecture 11.3.
- Hyperparameters (item- vs user-based, k) chosen on validation. The test set is evaluated once.
- Metrics: MSE, Concordance Index, Hit@1 and Hit@5.
- Final models: DAVIS item-based k = 10 and KIBA user-based k = 10. Reloading the saved models reproduces identical predictions.

## Task 3 summary
- Problem: CF cannot predict for a target with no training scores. It falls back to the ligand mean (mu + b_u), giving every new target the same score.
- Cold-target split: targets (not pairs) split 70 / 15 / 15 with seed 42. Validation and test targets have no training scores.
- Method: borrow scores from the training targets with the most similar protein sequences using the lecture 11.3 neighbour formula with sequence similarity instead of score similarity.
- Sequence similarity: Jaccard similarity on sets of 3-letter fragments (3-mers) of each sequence. It uses no binding scores so it works for any new target.
- k chosen on cold validation targets (DAVIS k = 5, KIBA k = 3).
- Results vs. the CF fallback on cold test targets:
  - DAVIS: MSE 0.460 -> 0.293, Hit@1 0.02 -> 0.48
  - KIBA: MSE 0.394 -> 0.365, Hit@1 0.13 -> 0.35

- Limitation: the gain depends on having a similar training target. Targets with no close relative see little to no improvement.

## Notes
- `data/kiba/kiba_drug_sim.zip` must be unzipped locally if needed (the file is too large to be uploaded on its own).
- - `saved_models/kiba_cf.npz` is about 26 MB, which is above the GitHub website's 25 MB upload limit. Commit and sync it from VS Code instead of uploading it on the website.
- Random seed 42 is used throughout so that the results are reproducible.

