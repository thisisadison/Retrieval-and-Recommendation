import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch



HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "..", "txt", "results.csv")  # written by main.py
FIGDIR = os.path.join(HERE, "..", "figures")

METHODS = ["LSH", "PQ", "HNSW"]
COLOURS = {"LSH": "#2a78d6", "PQ": "#eb6834", "HNSW": "#1baf7a"}  # validated colourblind-safe palette, fixed order
DATASETS = ["random", "clustered", "financebench"]
DATASET_LABEL = {"random": "random", "clustered": "clustered", "financebench": "FinanceBench"}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
FLAT = "#8a8a84"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential, light to dark

NB = 100000  # synthetic database size, used to split index size into its parts



def load_results(path=RESULTS_PATH):
    """
    Load the results table written by main.py

    :param path: path to results.csv
    :return: dataframe with one row per configuration, plus speedup vs the same-run flat index
    """
    df = pd.read_csv(path)
    df["speedup"] = df["flat_latency"] / df["latency"]  # same-run flat, so database size and query count cancel out

    return df



def select(df, method, dataset):
    """
    Rows for one method on one dataset
    """
    return df[(df["method"] == method) & (df["dataset"] == dataset)]



def frontier(rows, key, higher_is_better):
    """
    Keep only configurations not beaten on both cost and recall, sorted by cost

    :param rows: result rows
    :param key: cost column, e.g. "size", "build", "speedup"
    :param higher_is_better: True for speedup, False for size and build time
    :return: frontier rows
    """
    rows = rows.sort_values([key, "recall"], ascending=[not higher_is_better, False])
    best_so_far = rows["recall"].cummax().shift(fill_value=-1)  # best recall among cheaper configs

    return rows[rows["recall"] > best_so_far].sort_values(key)



def style_axes(ax, xlabel, ylabel, title, logx=False):
    """
    Apply a consistent, recessive axis style
    """
    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
    ax.set_title(title, fontsize=10, color=INK, pad=6)
    ax.set_ylim(0, 1.05)

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



def plot_parameter_sweeps(df, dataset="clustered"):
    """
    Report section 2: recall against each method's own parameter, one dataset
    """
    lsh = select(df, "LSH", dataset)
    pq = select(df, "PQ", dataset)
    hnsw = select(df, "HNSW", dataset)

    panels = [
        ("LSH — nbits", "LSH", "nbits", lsh),
        ("PQ — nbits (m=32)", "PQ", "nbits", pq[pq["m"] == 32]),
        ("PQ — m (nbits=10)", "PQ", "m", pq[pq["nbits"] == 10]),
        ("HNSW — efSearch (M=32)", "HNSW", "efSearch", hnsw[hnsw["M"] == 32]),
        ("HNSW — M (efSearch=64)", "HNSW", "M", hnsw[hnsw["efSearch"] == 64]),
    ]

    fig, grid = plt.subplots(2, 3, figsize=(10.5, 6.4), sharey=True)  # two rows so the figure fits a portrait page
    axes = grid.ravel()
    axes[-1].axis("off")  # five sweeps, six slots

    for ax, (title, method, key, rows) in zip(axes, panels):
        rows = rows.sort_values(key)
        xs = rows[key].astype(int).tolist()

        ax.plot(xs, rows["recall"], color=COLOURS[method], linewidth=2,
                marker="o", markersize=6, markeredgecolor="white", markeredgewidth=0.8)

        logx = max(xs) / min(xs) > 8
        style_axes(ax, key, "recall@10" if ax in grid[:, 0] else "", title, logx=logx)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()

    fig.suptitle(f"Parameter sensitivity — {DATASET_LABEL[dataset]} data, d=384",
                 fontsize=11, color=INK, y=1.01)
    fig.tight_layout()
    save(fig, "fig1_parameter_sweeps.png")



def plot_sweeps_by_method(df, datasets=("random", "clustered")):
    """
    Report section 4.2: one figure per method, one panel per hyperparameter sweep,
    one line per synthetic dataset (solid = clustered, dashed = random)
    """
    styles = {"random": "--", "clustered": "-"}
    markers = {"random": "s", "clustered": "o"}  # second cue besides line style, so overlapping lines stay readable
    groups = {
        "LSH": [("nbits", "nbits", lambda r: r)],
        "PQ": [("nbits (m=32)", "nbits", lambda r: r[r["m"] == 32]),
               ("m (nbits=10)", "m", lambda r: r[r["nbits"] == 10])],
        "HNSW": [("efSearch (M=32)", "efSearch", lambda r: r[r["M"] == 32]),
                 ("M (efSearch=64)", "M", lambda r: r[r["efSearch"] == 64])],
    }

    for method, panels in groups.items():
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), sharey=True, squeeze=False)
        axes = axes[0]

        for ax, (title, key, pick) in zip(axes, panels):
            for dataset in datasets:
                rows = pick(select(df, method, dataset)).sort_values(key)
                xs = rows[key].astype(int).tolist()
                ax.plot(xs, rows["recall"], color=COLOURS[method], linestyle=styles[dataset], linewidth=2,
                        marker=markers[dataset], markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                        label=DATASET_LABEL[dataset])

            logx = max(xs) / min(xs) > 8
            style_axes(ax, key, "recall@10" if ax is axes[0] else "", f"{method} — {title}", logx=logx)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
            ax.minorticks_off()

        axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left", handlelength=3.5)
        fig.tight_layout()
        save(fig, f"fig1_sweeps_{method.lower()}.png")



TRADEOFF_AXES = [
    ("size", "index size (MB, log)", False),
    ("speedup", "speedup vs Flat (×, log)", True),
    ("build", "build time (s, log)", False),
]


def draw_tradeoff(ax, df, dataset, key, xlabel, higher_better):
    """
    One trade-off panel: recall against a single cost, all methods on one dataset
    """
    flat = df[df["dataset"] == dataset].iloc[0]  # every row carries its run's flat baseline
    if key == "speedup":
        ax.axvline(1, color=FLAT, linewidth=1.2, linestyle=":")
        ax.text(1, 0.02, " Flat", fontsize=7, color=INK_MUTED, va="bottom")
    elif key == "size":
        ax.axvline(flat["flat_size"], color=FLAT, linewidth=1.2, linestyle=":")
        ax.text(flat["flat_size"], 0.02, " Flat", fontsize=7, color=INK_MUTED, va="bottom", ha="right")

    for method in METHODS:
        rows = select(df, method, dataset)

        ax.scatter(rows[key], rows["recall"], s=16, color=COLOURS[method], alpha=0.35, linewidths=0, zorder=2)

        front = frontier(rows, key, higher_better)
        ax.plot(front[key], front["recall"], color=COLOURS[method],
                linewidth=2, marker="o", markersize=6, markeredgecolor="white",
                markeredgewidth=0.8, label=method, zorder=3)

        best = rows.loc[rows["recall"].idxmax()]
        ax.annotate(method, (best[key], best["recall"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=8, color=INK)

    style_axes(ax, xlabel, "recall@10", "", logx=True)



def plot_tradeoffs(df, dataset="clustered"):
    """
    Report section 4.3: recall against memory, speed and build cost, all methods on one dataset.
    Saves the three panels stacked in one figure and as three separate figures
    """
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 10.5))  # stacked vertically, one cost per row
    for ax, (key, xlabel, higher_better) in zip(axes, TRADEOFF_AXES):
        draw_tradeoff(ax, df, dataset, key, xlabel, higher_better)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_MUTED)
    fig.suptitle(f"Method trade-offs — {DATASET_LABEL[dataset]} data\n"
                 f"(faint points: all configs; line: best achievable)", fontsize=11, color=INK, y=1.0)
    fig.tight_layout(h_pad=2.0)
    save(fig, "fig2_method_tradeoffs.png")

    for key, xlabel, higher_better in TRADEOFF_AXES:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        draw_tradeoff(ax, df, dataset, key, xlabel, higher_better)
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                  labelcolor=INK_MUTED)  # above the plot area so no point is hidden
        ax.set_title(f"{DATASET_LABEL[dataset]} data (faint points: all configs; line: best achievable)",
                     fontsize=9, color=INK_MUTED, pad=24)
        save(fig, f"fig2_tradeoff_{key}.png")



def plot_dataset_effect(df):
    """
    Report section 4: recall against speedup on every dataset
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)

    for ax, dataset in zip(axes, DATASETS):
        ax.axvline(1, color=FLAT, linewidth=1.2, linestyle=":")
        ax.text(1, 0.02, " Flat", fontsize=7, color=INK_MUTED, va="bottom")

        for method in METHODS:
            rows = select(df, method, dataset)

            ax.scatter(rows["speedup"], rows["recall"], s=16, color=COLOURS[method], alpha=0.35, linewidths=0, zorder=2)

            front = frontier(rows, "speedup", True)
            ax.plot(front["speedup"], front["recall"], color=COLOURS[method],
                    linewidth=2, marker="o", markersize=6, markeredgecolor="white",
                    markeredgewidth=0.8, label=method, zorder=3)

            best = rows.loc[rows["recall"].idxmax()]
            ax.annotate(method, (best["speedup"], best["recall"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=8, color=INK)

        style_axes(ax, "speedup vs Flat (×, log)", "recall@10" if ax is axes[0] else "",
                   DATASET_LABEL[dataset], logx=True)

    axes[0].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_MUTED)
    fig.suptitle("Effect of data — recall against speedup, normalised vectors, d=384", fontsize=11, color=INK, y=1.03)
    save(fig, "fig3_dataset_effect.png")



def tint(colour, amount=0.55):
    """
    Lighter version of a colour, used for the secondary part of a stacked bar
    """
    r, g, b = to_rgb(colour)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)



def best_rows(df):
    """
    Highest-recall configuration of every method on every dataset
    """
    return df.loc[df.groupby(["dataset", "method"])["recall"].idxmax()]



def plot_summary_heatmap(df):
    """
    Best recall of every method on every dataset, with the speedup it runs at
    """
    best = best_rows(df).set_index(["method", "dataset"])
    recall = np.array([[best.loc[(m, d), "recall"] for d in DATASETS] for m in METHODS])
    speedup = np.array([[best.loc[(m, d), "speedup"] for d in DATASETS] for m in METHODS])

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    mesh = ax.pcolormesh(recall, cmap=cmap, vmin=0, vmax=1, edgecolors="white", linewidth=3)

    for i in range(len(METHODS)):
        for j in range(len(DATASETS)):
            ink = "white" if recall[i, j] > 0.55 else INK
            ax.text(j + 0.5, i + 0.56, f"{recall[i, j]:.2f}", ha="center", va="center", fontsize=13, color=ink, weight="bold")
            ax.text(j + 0.5, i + 0.28, f"{speedup[i, j]:.1f}× Flat speed", ha="center", va="center", fontsize=7.5, color=ink)

    ax.set_xticks(np.arange(len(DATASETS)) + 0.5)
    ax.set_xticklabels([DATASET_LABEL[d] for d in DATASETS], fontsize=9, color=INK)
    ax.set_yticks(np.arange(len(METHODS)) + 0.5)
    ax.set_yticklabels(METHODS, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    bar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03)
    bar.set_label("best recall@10", fontsize=8, color=INK_MUTED)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)

    ax.set_title("Best recall@10 of each method (largest configuration tested)", fontsize=10, color=INK, pad=8)
    save(fig, "fig4_summary_heatmap.png")



def plot_bubbles(df, datasets=DATASETS, tag=""):
    """
    Recall against speedup, bubble area proportional to index size, one panel per dataset
    """
    scale = 2.5  # points^2 per MB
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets) + 0.6, 4.2), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, dataset in zip(axes, datasets):
        flat = df[df["dataset"] == dataset].iloc[0]
        ax.scatter([1], [1], s=flat["flat_size"] * scale, color=FLAT, alpha=0.35, edgecolors="white", linewidths=1.5, zorder=2)
        ax.annotate("Flat", (1, 1), textcoords="offset points", xytext=(0, -17), ha="center", fontsize=7, color=INK_MUTED)

        for method in METHODS:
            rows = select(df, method, dataset)
            ax.scatter(rows["speedup"], rows["recall"], s=rows["size"] * scale, color=COLOURS[method],
                       alpha=0.6, edgecolors="white", linewidths=1.2, zorder=3)

            best = rows.loc[rows["recall"].idxmax()]
            ax.annotate(method, (best["speedup"], best["recall"]), textcoords="offset points",
                        xytext=(8, 6), fontsize=8, color=INK, weight="bold")

        style_axes(ax, "speedup vs Flat (×, log)", "recall@10" if ax is axes[0] else "", DATASET_LABEL[dataset], logx=True)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlim(0.1, 300)

    method_keys = [Line2D([], [], marker="o", linestyle="", markersize=8, color=COLOURS[m], alpha=0.7, label=m) for m in METHODS]
    size_keys = [plt.scatter([], [], s=mb * scale, color=FLAT, alpha=0.35, label=f"{mb} MB") for mb in (5, 50, 150)]
    ncol = 6 if len(datasets) > 1 else 3
    fig.legend(handles=method_keys + size_keys, loc="lower center", ncol=ncol, frameon=False, fontsize=8,
               labelcolor=INK_MUTED, bbox_to_anchor=(0.5, -0.1 if ncol == 6 else -0.2), labelspacing=1.5,
               handletextpad=1.2, columnspacing=2)
    fig.suptitle("Recall, speed and memory together — bubble area = index size", fontsize=11, color=INK, y=1.03)
    save(fig, f"fig5_bubbles{tag}.png")



def plot_memory_breakdown(df, dataset="clustered"):
    """
    Where each method's memory goes, at its highest-recall configuration
    """
    best = best_rows(df[df["dataset"] == dataset]).set_index("method")
    flat_size = best["flat_size"].iloc[0]
    mb = lambda nbytes: nbytes * NB / 1024 ** 2

    lsh, pq, hnsw = best.loc["LSH"], best.loc["PQ"], best.loc["HNSW"]
    lsh_codes = mb(lsh["nbits"] / 8)
    pq_codes = mb(pq["m"] * pq["nbits"] / 8)

    bars = [  # label, [(part, size, colour)]
        ("Flat (exact)", [("raw vectors", flat_size, FLAT)]),
        (f"LSH {int(lsh['nbits'])} bits", [("hash codes", lsh_codes, COLOURS["LSH"]),
                                        ("hyperplanes", lsh["size"] - lsh_codes, tint(COLOURS["LSH"]))]),
        (f"PQ m={int(pq['m'])}, nbits={int(pq['nbits'])}", [("codes", pq_codes, COLOURS["PQ"]),
                                                            ("codebooks", pq["size"] - pq_codes, tint(COLOURS["PQ"]))]),
        (f"HNSW M={int(hnsw['M'])}", [("raw vectors", flat_size, FLAT),
                                      ("graph links", hnsw["size"] - flat_size, tint(COLOURS["HNSW"]))]),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for y, (label, parts) in enumerate(bars):
        left = 0
        for part, size, colour in parts:
            ax.barh(y, size, left=left, height=0.6, color=colour, edgecolor="white", linewidth=2)
            if size > 12:  # name the part inside the bar only when it fits
                ax.text(left + size / 2, y, part, ha="center", va="center", fontsize=7.5,
                        color="white" if colour in (FLAT, COLOURS["LSH"], COLOURS["PQ"]) else INK)
            left += size

        total = left
        if label.startswith("Flat"):
            ratio = ""
        elif total < flat_size:
            ratio = f"  ({flat_size / total:.0f}× smaller than Flat)"
        else:
            ratio = f"  ({total / flat_size:.2f}× Flat)"
        ax.text(total + 3, y, f"{total:.1f} MB{ratio}", va="center", fontsize=8, color=INK)

        small = [f"{part} {size:.1f} MB" for part, size, _ in parts if size <= 12]
        if small:  # parts too thin to label inside the bar
            ax.text(total + 3, y + 0.32, ", ".join(small), va="center", fontsize=7, color=INK_MUTED)

    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 260)
    ax.set_xlabel("index size (MB)", fontsize=9, color=INK_MUTED)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", labelsize=8, colors=INK_MUTED, length=3)
    ax.tick_params(axis="y", length=0)

    ax.set_title(f"Where the memory goes — highest-recall configuration, {DATASET_LABEL[dataset]} data (100,000 vectors)",
                 fontsize=10, color=INK, pad=8)
    save(fig, "fig6_memory_breakdown.png")



def plot_build_time(df):
    """
    Build time of each method on each dataset: line spans every configuration
    tested, the large marker is the highest-recall configuration
    """
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    offset = 0.24
    best = best_rows(df).set_index(["method", "dataset"])

    for i, method in enumerate(METHODS):
        for j, dataset in enumerate(DATASETS):
            x = j + (i - 1) * offset
            builds = select(df, method, dataset)["build"].unique()
            value = best.loc[(method, dataset), "build"]

            ax.plot([x, x], [builds.min(), builds.max()], color=COLOURS[method], linewidth=2, alpha=0.5, zorder=2)
            ax.scatter(np.full(len(builds), x), builds, s=16, color=COLOURS[method], linewidths=0, zorder=3)
            ax.scatter([x], [value], s=70, color=COLOURS[method], edgecolors="white", linewidths=1.5, zorder=4,
                       label=method if j == 0 else None)
            ax.annotate(f"{value:.0f}s" if value >= 10 else f"{value:.1f}s", (x, value), textcoords="offset points",
                        xytext=(7, -3), fontsize=7, color=INK)

    ax.set_yscale("log")
    ax.set_ylim(0.05, 1000)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABEL[d] for d in DATASETS], fontsize=9, color=INK)
    ax.set_ylabel("build time (s, log)", fontsize=9, color=INK_MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)

    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left", ncol=3)
    ax.set_title("Build time — line: every configuration tested, large marker: highest-recall configuration",
                 fontsize=10, color=INK, pad=8)
    save(fig, "fig7_build_time.png")




def main():
    df = load_results()
    print(f"loaded {len(df)} configurations")

    plot_parameter_sweeps(df)
    plot_sweeps_by_method(df)  # section 4.2: one figure per method, random and clustered
    plot_tradeoffs(df)
    plot_dataset_effect(df)
    plot_summary_heatmap(df)
    plot_bubbles(df, ("random", "clustered"), "_synthetic")
    plot_bubbles(df, ("financebench",), "_financebench")  # section 5.2
    plot_memory_breakdown(df)
    plot_build_time(df)



if __name__ == "__main__":
    main()
