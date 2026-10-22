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
    │   ├── main.py            # runs every experiment, writes src/txt/results.csv
    │   ├── methods.py         # index builders: Flat, LSH, PQ, HNSW
    │   ├── metrics.py         # recall@k, latency, index size, build time, results table
    │   ├── dataset.py         # random, clustered and FinanceBench data
    │   ├── plots.py           # results figures (fig1–fig7) from results.csv
    │   └── eda.py             # dataset figures (eda1–eda6)
    ├── txt/
    │   ├── results.csv        # one row per configuration (used by plots.py)
    │   └── results.txt        # raw console log of the last full run (main comparison)
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

**Run every command below from the project root** (the folder containing `README.md`). The data paths are relative to it.

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

**3. Dataset figures** (about a minute; includes FinanceBench only once its cache exists):

```bash
python src/py/eda.py
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

---

## Notes

- Latency is the median of 10 runs. Speedup is measured against the Flat index built in the same run, so it doesn't depend on machine load between runs.
- LSH and PQ recall reproduce exactly. HNSW recall can vary by about ±0.03 between runs, because FAISS builds the graph in parallel.
- Timings are from a laptop CPU and will differ on other machines; the relative results should hold.
