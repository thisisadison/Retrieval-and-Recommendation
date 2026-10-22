# SC4020 Project 1 — Similarity Search

Benchmarks three approximate nearest-neighbour (ANN) methods from FAISS, **LSH**, **PQ** and **HNSW**, against exact search (Flat) on three datasets:

| dataset | description | size |
|---|---|---|
| random | Gaussian vectors, normalised (no structure) | 100,000 × 384 |
| clustered | 6 Gaussian clusters with correlated dimensions, normalised | 100,000 × 384 |
| FinanceBench | 500-character passages from 84 financial filings, embedded with `all-MiniLM-L6-v2`; the 150 analyst questions are the queries | 104,169 × 384 |

Every vector has unit length, and the ground truth is the exact top-10 under L2 distance. Each method is measured on recall@10, query latency, speedup vs Flat, index size and build time.

---

## Project structure

```text
SC4020_Project1/
├── requirements.txt
├── README.md
├── data/                      # created on first run, not submitted (gitignored)
└── src/
    ├── py/
    │   ├── main.py            # runs the main comparison and both ablations, writes src/txt/results.csv
    │   ├── methods.py         # index builders: Flat, LSH, PQ, IVF-PQ, HNSW
    │   ├── metrics.py         # recall@k, latency, index size, build time, results table
    │   ├── dataset.py         # random, clustered and FinanceBench data
    │   ├── plots.py           # results figures (fig1–fig7) from results.csv
    │   ├── ablation_plots.py  # IVF ablation figure (fig9) from results.csv
    │   ├── opq.py             # improvement: OPQ vs plain PQ, writes opq_results.csv and fig8
    │   ├── cases.py           # FinanceBench per-question success and failure cases (fig10)
    │   ├── lsh_diagnostics.py # why learned LSH thresholds help clustered data but hurt FinanceBench (prints only)
    │   └── eda.py             # dataset figures (eda1–eda6)
    ├── txt/
    │   ├── results.csv        # one row per configuration (used by plots.py and ablation_plots.py)
    │   ├── results.txt        # raw console log of the last full run (main comparison)
    │   ├── opq_results.csv    # OPQ vs PQ, kept apart from results.csv
    │   ├── cases.txt          # per-question summary and worked examples
    │   └── cases_per_query.csv  # recall@10 of every method for each of the 150 questions
    ├── figures/               # all generated figures
    └── md/
        ├── findings.md        # results tables and observations
        ├── lsh.md, pq.md, hnsw.md   # theory notes for each method
```

---

## Setup

Tested with Python 3.13.9.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run every command below from the project root**, the `SC4020_Project1/` folder containing this `README.md` (not the repository root above it). The data paths are relative to it.

---

## Data

**Synthetic data** (random, clustered) is generated automatically, with fixed seeds.

**FinanceBench** needs a one-off download of the 84 source filings (PDFs):

```bash
python -c "import sys; sys.path.insert(0, 'src/py'); from dataset import download_financebench_pdfs; download_financebench_pdfs()"
```

The PDFs are saved to `data/financebench_pdfs/`. The download resumes if interrupted.

The first time `main.py` runs, it chunks and embeds the filings (about 20 minutes on a laptop CPU) and caches the vectors in `data/`. Later runs load the cache instantly.

---

## How to run

**1. Experiments** (about 45 minutes, mostly PQ training). This runs the main comparison and the two ablations:

```bash
python -u src/py/main.py | tee src/txt/results.txt
```

This writes `src/txt/results.csv`, one row per configuration.

To run only part of it:

```bash
python -u -c "import sys; sys.path.insert(0, 'src/py'); import main; main.run_all_synthetic(); main.save_results(main.RESULTS_PATH)"
python -u -c "import sys; sys.path.insert(0, 'src/py'); import main; main.run_all_finance(); main.save_results(main.RESULTS_PATH)"
python -u -c "import sys; sys.path.insert(0, 'src/py'); import main; main.run_lsh_ablation(); main.run_ivf_ablation(); main.save_results(main.RESULTS_PATH)"
```

Rows for the same dataset and method are replaced, so the other results in `results.csv` are kept.

**2. Results figures** (seconds, reads `results.csv`):

```bash
python src/py/plots.py
```

**3. IVF ablation figure** (seconds, reads `results.csv`):

```bash
python src/py/ablation_plots.py
```

**4. Dataset figures** (about a minute; includes FinanceBench only once its cache exists):

```bash
python src/py/eda.py
```

Synthetic and FinanceBench get separate `eda1`, `eda3` and `eda4` figures (`_synthetic`, `_financebench`); `eda2`, `eda5` and `eda6` compare all three datasets.

**5. Improvement: OPQ vs PQ** (writes `src/txt/opq_results.csv` and `fig8`):

```bash
python -u src/py/opq.py                      # all three datasets
python -u src/py/opq.py random clustered     # or only the ones named
```

**6. FinanceBench success and failure cases** (needs the PDFs and the embedding cache):

```bash
python -u src/py/cases.py
```

This writes `src/txt/cases.txt`, `src/txt/cases_per_query.csv` and `fig10`. The first run downloads the FinanceBench questions from Hugging Face and rebuilds the passage texts from the PDFs into `data/financebench_texts.json`.

**7. LSH threshold diagnostics** (needs the FinanceBench cache; prints to the console only):

```bash
python -u src/py/lsh_diagnostics.py
```

All figures are saved to `src/figures/`.

---

## Parameters tested

| method | parameters |
|---|---|
| LSH | nbits ∈ {256, 512, 1024, 2048, 4096} |
| PQ | m = 32 with nbits ∈ {6, 8, 10}; nbits = 10 with m ∈ {8, 16, 64, 192} |
| HNSW | M = 32 with efSearch ∈ {16, 32, 64, 128, 256}; efSearch = 64 with M ∈ {8, 16, 64} |

**Ablations**

| ablation | what changes | settings |
|---|---|---|
| LSH thresholds | each bit split at the data's median instead of at 0 | nbits ∈ {256, 512, 1024, 2048, 4096} |
| IVF | PQ with an inverted file in front (only the nearest cells are searched) vs plain PQ | m = 192, nbits = 8, nlist = 1024, nprobe ∈ {1, 8, 32, 128} |

**Improvement**

| improvement | what changes | settings |
|---|---|---|
| OPQ | a learned rotation is applied before PQ, compared with plain PQ at the same code size | m ∈ {16, 32, 64, 192}, nbits = 8 |

**Success and failure cases** use one mid-range setting per method, so recall varies between questions: LSH nbits = 1024; PQ m = 32, nbits = 8; HNSW M = 32, efSearch = 64.

---

## Notes

- Latency is the median of 10 runs. Speedup is measured against the Flat index built in the same run, so it doesn't depend on machine load between runs.
- LSH and PQ recall reproduce exactly. HNSW recall can vary by about ±0.03 between runs, because FAISS builds the graph in parallel. `cases.py` builds its HNSW graph on one thread, so its per-question results reproduce exactly.
- Timings are from a laptop CPU and will differ on other machines; the relative results should hold.
