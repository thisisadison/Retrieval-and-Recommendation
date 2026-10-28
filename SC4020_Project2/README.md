# SC4020 Project 2: DAVIS and KIBA Drug–Target Binding

## Folder structure
- `data/davis/`, `data/kiba/`: original datasets from DeepDTA (Öztürk et al., 2018)
- `data/clean/`: cleaned, deduplicated data created by Task 1 (used by Tasks 2 and 3)
- `figures/`: figures saved by the notebooks for the report
- `task1_analysis.ipynb`: Task 1, data loading, cleaning, visualisation and analysis

## How to run
1. Open the `SC4020_Project2` folder in VS Code.
2. Run `task1_analysis.ipynb` first (Run All). It creates `data/clean/` and `figures/`.

## Notes
- `data/kiba/kiba_drug_sim.zip` must be unzipped locally if needed (file too large for GitHub upload).
- DAVIS scores are converted to pKd = 9 − log10(Kd); KIBA scores are used as supplied.
- Screening-limit and extreme values are flagged, not removed.