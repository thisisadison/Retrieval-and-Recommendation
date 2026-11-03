# SC4020 Project 2: DAVIS and KIBA Drug–Target Binding

## Folder structure
- `data/davis/`, `data/kiba/`: original datasets from DeepDTA provided
- `data/clean/`: cleaned, deduplicated data created by Task 1 (used by Tasks 2 and 3)
- `figures/`: figures saved by the notebooks for the report
- `saved_models/`: trained CF models from Task 2 (`davis_cf.npz`, `kiba_cf.npz`) and cold-target models from Task 3 (`davis_cold.npz`, `kiba_cold.npz`)
- `task1_analysis.ipynb`: Task 1, data loading, cleaning, visualisation and analysis
- `task2_cf.ipynb`: Task 2, collaborative filtering recommender using lecture 11.3 method
- `task3_cold_target.ipynb`: Task 3, cold-target generalisation using sequence similarity
- `project2_data.py`, `project2_metrics.py`, `project2_cf.py`, `project2_cold.py`: shared code used by the notebooks (cleaning, metrics, CF model, cold-target model)
- `results/cold/`: Task 3 results over three splits (seeds 42, 7, 2026), created by `project2_cold.py`
- `tests/`: checks for the metrics, CF, cold-target model and cleaning

## How to run
1. Open the `SC4020_Project2` folder in VS Code and install the requirements (below).
2. Run `task1_analysis.ipynb` first (Run All). It creates `data/clean/` and `figures/`.
3. Run `task2_cf.ipynb` (Run All). It creates `saved_models/` and more figures.
4. Run `python project2_cold.py` once. It creates `results/cold/` and the Task 3 saved models.
5. Run `task3_cold_target.ipynb` (Run All). It saves its figures to `figures/`.

Optional: `python -m unittest discover -s tests` runs the checks.

## Requirements
Validated with Python 3.13.7. Install the pinned libraries once before running the notebooks:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Task 1 summary
We started by making sure both datasets are in the same shape. DAVIS gives binding as Kd in nM, where lower means stronger. Hence we converted it to pKd = 9 − log10(Kd) to make higher mean stronger, matching KIBA's scale. Some IDs turned out to share the exact same SMILES or protein sequence, so we merged those pairs and averaged their scores.

We did not delete any unusual values. About 70% of DAVIS sits at the screening limit (pKd = 5), and KIBA has a handful of extreme scores, so we flagged them and discussed them instead. Missing KIBA entries were left empty since a pair that was never measured does not necessarily mean that it does not bind.

## Task 2 summary
We treated ligands as users and targets as items, so the recommender answers "which targets is this ligand likely to bind?". The data was split 70 / 15 / 15 with seed 42, making sure every ligand and target in the validation and test sets also appears in training.

The main model is the neighbourhood CF from lecture 11.3. It starts from the baseline (mu + b_u + b_i) and adjusts it using similar targets or ligands measured with centred cosine similarity. We compared it against a global mean, a ligand mean and the baseline itself using MSE, Concordance Index, Hit@1 and Hit@5. Validation picked item-based CF with k = 10 for DAVIS and user-based CF with k = 10 for KIBA, and the test set was only used once at the end. Both models are saved in `saved_models/`.

## Task 3 summary
CF However has a blind spot. If a target has no training scores, it can only fall back to the ligand's average (mu + b_u), so every new target gets the same prediction. To test this properly, we split by target instead of by pair (70 / 15 / 15, seed 42), so validation and test targets have no scores in training at all.

Our fix borrows scores from the training targets whose protein sequences look most similar. Each sequence is broken into overlapping 3-letter fragments, and two targets are compared by how many fragments they share (Jaccard similarity). This requires no binding data so it works for any new target. The prediction itself uses the same neighbour formula as Task 2, with sequence similarity in place of score similarity. We picked k on the validation targets, where k = 5 for DAVIS and k = 3 for KIBA.

Compared with the CF fallback on the cold test targets:
- DAVIS: MSE dropped from 0.460 to 0.293, and Hit@1 rose from 0.02 to 0.48
- KIBA: MSE dropped from 0.394 to 0.365, and Hit@1 rose from 0.14 to 0.35

The method works well only when a new target has a similar protein in the training set. If no similar protein exists, its predictions are no better than the fallback.

**Extension: similarity-dependent shrinkage.** Because weak neighbours can make predictions worse, we blend the neighbour estimate with the ligand's average, trusting the neighbours more when the closest protein *measured for that ligand* is more similar:

```text
prediction = ligand_mean + weight * (neighbour_estimate - ligand_mean)
weight     = clip((s - low) / (high - low), 0, 1)
```

`low`, `high` and k are chosen on validation targets. A constant-weight blend is also tested to check that the similarity is what helps. Since this idea came from looking at the seed-42 results, we repeated the experiment on three splits (seeds 42, 7, 2026):

| Dataset | Original MSE, seed 42 | Extension MSE, seed 42 | Mean MSE over 3 splits: original -> extension |
|---|---:|---:|---:|
| DAVIS | 0.2926 | 0.2687 | 0.3024 -> 0.2628 |
| KIBA | 0.3647 | 0.3028 | 0.3388 -> 0.3016 |

MSE improves on all six dataset/split combinations, and five of six paired bootstrap intervals exclude zero (KIBA seed 7 does not). Ranking does not consistently improve.

The saved model can predict for a protein sequence that has no training measurements:

```python
from project2_cold import predict_new_target
predictions = predict_new_target("saved_models/davis_cold.npz", protein_sequence, ["11338033"])
```

## Evaluation notes
- Hit@1/5 and the rank of the true best target are averaged over all orderings of tied predictions, so results do not depend on row order. Every target tied for the highest true score counts as correct.
- Pooled CI compares all pairs, including pairs from different ligands, so a ligand-mean model scores about 0.75 without ranking anything within a ligand. We also report macro CI (CI within each ligand, averaged), which is 0.5 for such a model.
- In CF, ligand or target profiles that are constant up to rounding error get zero similarity, otherwise rounding noise creates large false similarities in KIBA.

## Notes
- `data/kiba/kiba_drug_sim.zip` must be unzipped locally if needed (the file is too large to be uploaded on its own).
- `saved_models/kiba_cf.npz` is about 25 MB, at the GitHub website's 25 MB upload limit. Commit and sync it from VS Code instead of uploading it on the website.
- Random seed 42 is used throughout so that the results are reproducible. Task 3 additionally reports seeds 7 and 2026.
