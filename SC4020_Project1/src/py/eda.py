import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dataset import generate_random_data, generate_clustered_data, generate_financebench_data
from main import D, SYNTHETIC_NB, SYNTHETIC_NQ  # same config as the experiments



HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "figures")
FINANCE_CACHE = os.path.join("data", "financebench_xb.npy")  # relative to the project root, like main.py

SAMPLE = 5000  # database points used for projections, norms and pca
K = 10         # neighbours per query, matching recall@10

LABEL = {"random": "random", "clustered": "clustered", "financebench": "FinanceBench"}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
DARK = "#3b3b38"   # primary series
LIGHT = "#b8b7b0"  # comparison series

rng = np.random.default_rng(42)



def load_datasets():
    """
    Load every dataset at the experiment configuration. FinanceBench is only
    loaded from its cache, so this never triggers the 20 minute embedding run

    :return: dict of name -> (xb, xq)
    """
    data = {
        "random": generate_random_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, d=D),
        "clustered": generate_clustered_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, n_features=D),
    }

    if os.path.exists(FINANCE_CACHE):
        xb, xq, _ = generate_financebench_data()
        data["financebench"] = (xb, xq)
    else:
        print(f"skipping financebench: {FINANCE_CACHE} not found, run main.py from the project root first")

    return data



def sample_rows(x, n=SAMPLE):
    """
    Random subset of rows, so plots stay readable and fast
    """
    idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
    return x[idx]



def unit(x):
    """
    L2-normalise rows without modifying the input
    """
    return x / np.linalg.norm(x, axis=1, keepdims=True)



def style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
    ax.set_title(title, fontsize=10, color=INK, pad=6)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)



def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, name)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)



def plot_projections(data, tag=""):
    """
    2-D PCA projection of a database sample: shows whether a dataset has visible structure
    """
    fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data), 3.8))

    for ax, (name, (xb, _)) in zip(np.atleast_1d(axes), data.items()):
        x = sample_rows(xb)
        pca = PCA(n_components=2, random_state=42).fit(x)
        p = pca.transform(x)
        kept = pca.explained_variance_ratio_.sum()

        ax.scatter(p[:, 0], p[:, 1], s=3, color=DARK, alpha=0.35, linewidths=0)
        style_axes(ax, "PC 1", "PC 2", f"{LABEL[name]} — 2 PCs keep {kept:.0%} of variance")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"2-D PCA projection of {SAMPLE:,} database vectors (d={D})", fontsize=11, color=INK, y=1.03)
    save(fig, f"eda1_projections{tag}.png")



def plot_norms(data):
    """
    Distribution of vector lengths: unit-norm data makes L2 and cosine rank identically
    """
    fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data), 3.2))

    for ax, (name, (xb, _)) in zip(np.atleast_1d(axes), data.items()):
        norms = np.linalg.norm(sample_rows(xb), axis=1)
        spread = norms.std() / norms.mean()

        if spread < 1e-3:  # all norms equal, a histogram would be a single bar of zero width
            ax.axvline(norms.mean(), color=DARK, linewidth=3)
            ax.set_xlim(norms.mean() - 0.5, norms.mean() + 0.5)
            ax.set_yticks([])
            note = "every vector has length 1"
        else:
            ax.hist(norms, bins=60, color=DARK, alpha=0.85)
            note = f"coefficient of variation {spread:.2f}"

        style_axes(ax, "vector length ‖x‖", "count", f"{LABEL[name]} — {note}")

    fig.suptitle("Vector lengths", fontsize=11, color=INK, y=1.03)
    save(fig, "eda2_norms.png")



def plot_similarity(data, tag=""):
    """
    Cosine similarity of each query to its 10 true neighbours vs to random database
    points. The gap between the two is what every ANN method has to exploit
    """
    fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data), 3.4), sharey=False)

    for ax, (name, (xb, xq)) in zip(np.atleast_1d(axes), data.items()):
        xb_u, xq_u = unit(xb), unit(xq)
        sims = xq_u @ xb_u.T  # cosine of every query to every database vector

        neighbours = -np.sort(-sims, axis=1)[:, :K].ravel()
        random_pairs = sims[:, rng.choice(sims.shape[1], size=1000, replace=False)].ravel()

        bins = np.linspace(min(random_pairs.min(), neighbours.min()), 1.0, 70)
        ax.hist(random_pairs, bins=bins, color=LIGHT, density=True, label="random pairs")
        ax.hist(neighbours, bins=bins, color=DARK, alpha=0.85, density=True, label=f"{K} nearest neighbours")

        gap = np.median(neighbours) - np.median(random_pairs)
        style_axes(ax, "cosine similarity to query", "density", f"{LABEL[name]} — median gap {gap:.2f}")
        ax.set_yticks([])

    # one legend under the panels, so it never covers the bars
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_MUTED,
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.suptitle("How much do true neighbours stand out? (cosine similarity, per query)",
                 fontsize=11, color=INK, y=1.03)
    save(fig, f"eda3_neighbour_similarity{tag}.png")



def plot_intrinsic_dimension(data, tag=""):
    """
    Cumulative explained variance: how many directions the data really uses out of 384
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    styles = {"random": "--", "clustered": "-", "financebench": "-"}  # same line styles as the sweep figures
    markers = {"random": "s", "clustered": "o", "financebench": "o"}

    for i, (name, (xb, _)) in enumerate(data.items()):
        pca = PCA(random_state=42).fit(sample_rows(xb))
        cum = np.cumsum(pca.explained_variance_ratio_)
        n90 = int(np.searchsorted(cum, 0.9)) + 1

        ax.plot(np.arange(1, len(cum) + 1), cum, color=DARK, linestyle=styles[name], linewidth=2)
        ax.annotate(f"{LABEL[name]}: {n90} dims for 90%", (n90, 0.9), textcoords="offset points",
                    xytext=(6, -14 - 12 * i), fontsize=8, color=INK)
        ax.plot([n90], [0.9], marker=markers[name], color=DARK, markersize=5)

    ax.axhline(0.9, color=GRID, linewidth=1)
    style_axes(ax, "number of principal components", "cumulative variance explained",
               f"Intrinsic dimensionality (all datasets nominally d={D})")
    ax.set_ylim(0, 1.02)
    save(fig, f"eda4_intrinsic_dimension{tag}.png")



def plot_metric_comparison(data):
    """
    L2 vs cosine on the vectors as indexed, as two figures. First: L2 distance against
    cosine similarity for query-database pairs; unit-length data falls on the line
    ||x-y||^2 = 2 - 2cos. Second: overlap between the top-10 neighbours chosen by each metric
    """
    names = list(data)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6))
    overlaps = {}

    for ax, name in zip(np.atleast_1d(axes), names):
        xb, xq = data[name]

        # squared l2 via ||q||^2 + ||x||^2 - 2 q.x, avoiding a (nq, nb, d) array
        l2 = (xq ** 2).sum(1)[:, None] + (xb ** 2).sum(1)[None, :] - 2 * xq @ xb.T
        cos = unit(xq) @ unit(xb).T

        top_l2 = np.argsort(l2, axis=1)[:, :K]
        top_cos = np.argsort(-cos, axis=1)[:, :K]
        overlaps[name] = np.mean([len(set(a) & set(b)) / K for a, b in zip(top_l2, top_cos)])

        pick = rng.choice(xb.shape[0], size=300, replace=False)  # 300 random database points per query
        ax.scatter(cos[:, pick].ravel(), l2[:, pick].ravel(), s=2, color=DARK, alpha=0.15, linewidths=0)

        c = np.linspace(cos[:, pick].min(), 1, 100)  # every dataset is unit length, so every panel follows this line
        ax.plot(c, 2 - 2 * c, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.text(0.97, 0.95, "‖x−y‖² = 2 − 2cos", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK_MUTED)

        style_axes(ax, "cosine similarity", "squared L2 distance", LABEL[name])

    fig.suptitle("L2 distance against cosine similarity on unit-length vectors", fontsize=11, color=INK, y=1.03)
    save(fig, "eda5_l2_vs_cosine.png")

    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar([LABEL[n] for n in names], [overlaps[n] for n in names], color=DARK, width=0.55)
    for bar, n in zip(bars, names):
        ax.text(bar.get_x() + bar.get_width() / 2, overlaps[n] + 0.02, f"{overlaps[n]:.2f}",
                ha="center", fontsize=8, color=INK)
    style_axes(ax, "", "fraction shared", f"Top-{K} neighbours shared by L2 and cosine")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="x", visible=False)
    save(fig, "eda6_topk_overlap.png")

    for n in names:
        print(f"top-{K} overlap, L2 vs cosine, {n}: {overlaps[n]:.3f}")




def main():
    data = load_datasets()

    # experiment 1 (synthetic) and experiment 2 (FinanceBench) get separate figures,
    # so each section of the report shows only its own data
    groups = {"_synthetic": {n: data[n] for n in ("random", "clustered")}}
    if "financebench" in data:
        groups["_financebench"] = {"financebench": data["financebench"]}

    for tag, subset in groups.items():
        plot_projections(subset, tag)
        plot_similarity(subset, tag)
        plot_intrinsic_dimension(subset, tag)

    # appendix figures compare all datasets side by side
    plot_norms(data)
    plot_metric_comparison(data)



if __name__ == "__main__":
    main()
