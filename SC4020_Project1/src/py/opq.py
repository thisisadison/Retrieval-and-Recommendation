import os
import sys

import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods import *
from metrics import *
from main import run_flat_baseline, D, SYNTHETIC_NB, SYNTHETIC_NQ
from dataset import generate_random_data, generate_clustered_data, generate_financebench_data
from plots import style_axes, save, tint, COLOURS, DATASET_LABEL, INK, INK_MUTED



HERE = os.path.dirname(os.path.abspath(__file__))
OPQ_RESULTS_PATH = os.path.join(HERE, "..", "txt", "opq_results.csv")  # kept apart from results.csv

M_VALUES = [16, 32, 64, 192]  # each divides 384, so every subspace has a whole number of dimensions
NBITS = 8                     # byte-aligned codes, the fastest pq setting on every dataset
MSE_SAMPLE = 20000            # vectors used to measure reconstruction error



def build_opq_index(d: int, m: int, nbits: int, xb: np.ndarray):
    """
    Build OPQ index: learns a d x d rotation before product quantization, so that
    correlated dimensions are spread evenly across the m subspaces instead of
    being cut apart by the fixed split of plain PQ

    :param d: dimension of vectors
    :param m: number of subspaces
    :param nbits: bits per subquantizer code
    :param xb: vector db
    :return: trained OPQ index (rotation + PQ)
    """
    opq = faiss.OPQMatrix(d, m)  # rotation and pq codebooks are learned together, alternating
    pq = faiss.IndexPQ(d, m, nbits)
    index = faiss.IndexPreTransform(opq, pq)  # every vector, db or query, is rotated before it reaches the pq
    index.train(xb)

    add_vectors(index, xb)

    return index



def reconstruction_mse(index, xb: np.ndarray, n=MSE_SAMPLE) -> float:
    """
    Mean squared error between vectors and their decoded pq codes, i.e. how much
    information the compression throws away

    :param index: trained PQ or OPQ index
    :param xb: vector db
    :param n: number of vectors to sample
    :return: mean squared reconstruction error per vector
    """
    rng = np.random.default_rng(0)
    x = xb[rng.choice(len(xb), size=min(n, len(xb)), replace=False)]
    x_rec = index.sa_decode(index.sa_encode(x))  # encode then decode, through the rotation for opq

    return float(((x - x_rec) ** 2).sum(axis=1).mean())



def load_datasets(names):
    """
    Load the requested datasets with the same settings as the main runs.
    FinanceBench is skipped if its embedding cache doesn't exist yet

    :param names: dataset names to load
    :return: dict of dataset name -> (xb, xq)
    """
    loaders = {
        "random": lambda: generate_random_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, d=D),
        "clustered": lambda: generate_clustered_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, n_features=D),
        "financebench": lambda: generate_financebench_data()[:2],  # labels not needed
    }

    data = {}
    for name in names:
        if name == "financebench" and not os.path.exists(os.path.join("data", "financebench_xb.npy")):
            print("skipped financebench: no embedding cache in data/, see the README")
            continue
        data[name] = loaders[name]()

    return data



def run_opq_comparison(names=("random", "clustered", "financebench")):
    """
    Improvement: OPQ against plain PQ at the same code size (m bytes per vector).
    Both are built in the same run against the same flat baseline
    """
    for dataset, (xb, xq) in load_datasets(names).items():
        d = xb.shape[1]
        nq = xq.shape[0]
        gt_ids = run_flat_baseline(dataset, xb, xq)

        for m in M_VALUES:
            for method, build_fn in [("PQ", build_pq_index), ("OPQ", build_opq_index)]:

                print(f"\n========== {method} m={m}, nbits={NBITS} ({dataset}) ==========")

                index = measure_build_time(build_fn, d=d, m=m, nbits=NBITS, xb=xb)
                _, pred_ids = search_index(index, noNeighbors=10, noQueries=nq, xq=xq)
                mse = reconstruction_mse(index, xb)
                print(f"reconstruction mse: {mse:.5f}")
                get_metrics(pred_ids, gt_ids, index, xq, k=10, dataset=dataset, method=method, m=m, nbits=NBITS, mse=mse)



def plot_opq(path=OPQ_RESULTS_PATH):
    """
    Recall against m, plain PQ vs OPQ (top row), and how much OPQ cuts the reconstruction
    error relative to PQ (bottom row), one column per dataset
    """
    df = pd.read_csv(path)
    datasets = [d for d in ["random", "clustered", "financebench"] if d in set(df["dataset"])]
    styles = {"PQ": dict(color=tint(COLOURS["PQ"]), linestyle="--", marker="s"),
              "OPQ": dict(color=COLOURS["PQ"], linestyle="-", marker="o")}
    line = dict(linewidth=2, markersize=6, markeredgecolor="white", markeredgewidth=0.8)

    fig, axes = plt.subplots(2, len(datasets), figsize=(3.6 * len(datasets), 6.2), squeeze=False)

    for col, dataset in enumerate(datasets):
        rows = {method: df[(df["dataset"] == dataset) & (df["method"] == method)].sort_values("m").set_index("m")
                for method in ["PQ", "OPQ"]}
        xs = rows["PQ"].index.astype(int).tolist()

        for method in ["PQ", "OPQ"]:
            axes[0, col].plot(xs, rows[method]["recall"], label=method, **line, **styles[method])

        # raw mse differs ~60x between datasets, so a shared scale would flatten clustered to a line at 0
        reduction = (1 - rows["OPQ"]["mse"] / rows["PQ"]["mse"]) * 100
        axes[1, col].plot(xs, reduction, **line, **styles["OPQ"])

        for row, ylabel in [(0, "recall@10"), (1, "MSE reduction vs PQ (%)")]:
            ax = axes[row, col]
            style_axes(ax, "m (nbits=8)", ylabel if col == 0 else "",
                       DATASET_LABEL[dataset] if row == 0 else "", logx=True)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
            ax.minorticks_off()

        axes[1, col].set_ylim(0, 60)  # style_axes fixes y to recall's 0-1

    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left", handlelength=3.5)
    fig.suptitle("OPQ vs plain PQ at equal code size (m bytes per vector), d=384", fontsize=11, color=INK, y=1.01)
    fig.tight_layout()
    save(fig, "fig8_opq_vs_pq.png")



def main():
    names = sys.argv[1:] or ["random", "clustered", "financebench"]
    run_opq_comparison(names)
    save_results(OPQ_RESULTS_PATH)
    plot_opq()



if __name__ == "__main__":
    main()
