import matplotlib.pyplot as plt

from plots import load_results, style_axes, save, tint, COLOURS, DATASET_LABEL, DATASETS, FLAT, INK, INK_MUTED



def plot_ivf(df):
    """
    IVF ablation: recall against speedup vs Flat, plain PQ (one point) and IVF-PQ (one point per nprobe),
    one panel per dataset
    """
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.6 * len(DATASETS), 3.6), sharey=True)

    for ax, dataset in zip(axes, DATASETS):
        pq = df[(df["dataset"] == dataset) & (df["method"] == "PQ-noIVF")]
        ivf = df[(df["dataset"] == dataset) & (df["method"] == "IVF-PQ")].sort_values("nprobe")

        ax.axvline(1, color=FLAT, linewidth=0.8, linestyle=":")  # right of this line is faster than flat
        ax.plot(ivf["speedup"], ivf["recall"], color=COLOURS["PQ"], linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, label="IVF-PQ")
        ax.plot(pq["speedup"], pq["recall"], color=tint(COLOURS["PQ"]), linestyle="none", marker="s", markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, label="PQ (no IVF)")

        for _, row in ivf.iterrows():
            ax.annotate(f"nprobe={int(row['nprobe'])}", (row["speedup"], row["recall"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=INK_MUTED)

        style_axes(ax, "speedup vs Flat (×)", "recall@10" if ax is axes[0] else "", DATASET_LABEL[dataset], logx=True)
        ax.set_xlim(0.5, 200)
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.minorticks_off()

    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper right")
    fig.suptitle("PQ with and without IVF (m=192, nbits=8, nlist=1024)", fontsize=11, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "fig9_ivf_ablation.png")



def main():
    plot_ivf(load_results())



if __name__ == "__main__":
    main()
