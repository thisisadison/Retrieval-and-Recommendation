import os
import json

import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from dataset import generate_financebench_data, chunk_text
from methods import build_flat_index, build_lsh_index, build_pq_index, build_hnsw_index, search_index
from plots import style_axes, save, COLOURS, INK, INK_MUTED



HERE = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(HERE, "..", "txt", "cases.txt")
PER_QUERY_PATH = os.path.join(HERE, "..", "txt", "cases_per_query.csv")
TEXTS_PATH = os.path.join("data", "financebench_texts.json")  # gitignored, rebuilt from the pdfs if missing

# mid-range settings from the main sweeps, so per-query recall varies between questions
SETTINGS = {
    "LSH": "nbits=1024",
    "PQ": "m=32, nbits=8",
    "HNSW": "M=32, efSearch=64",
}



def load_texts(labels, pdf_dir="data/financebench_pdfs", chunk_size=500, overlap=100):
    """
    Passage text for every vector in the cached FinanceBench embeddings. The cache only keeps
    (doc_name, page), so the passages are rebuilt with exactly the chunking used in dataset.py

    :param labels: (doc_name, page) for each cached vector, used to check the order matches
    :return: list of passage strings, aligned with xb
    """
    if os.path.exists(TEXTS_PATH):
        return json.load(open(TEXTS_PATH, encoding="utf8"))

    from pypdf import PdfReader

    texts, rebuilt_labels, seen = [], [], set()
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.endswith(".pdf"):
            continue
        doc_name = fname[:-4]
        try:
            reader = PdfReader(os.path.join(pdf_dir, fname))
        except Exception:
            continue

        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            for chunk in chunk_text(page_text, chunk_size, overlap):
                if chunk in seen:
                    continue
                seen.add(chunk)
                texts.append(chunk)
                rebuilt_labels.append((doc_name, page_num))

    assert len(texts) == len(labels), "rebuilt passages don't match the cached embeddings"
    assert all(tuple(a) == tuple(b) for a, b in zip(rebuilt_labels, labels)), "passage order differs from the cache"

    json.dump(texts, open(TEXTS_PATH, "w", encoding="utf8"))
    return texts



def per_query_recall(pred_ids: np.ndarray, gt_ids: np.ndarray, k=10) -> np.ndarray:
    return np.array([len(set(p[:k]) & set(g[:k])) / k for p, g in zip(pred_ids, gt_ids)])



def build_indexes(xb, d):
    lsh = build_lsh_index(d=d, nbits=1024, xb=xb)
    pq = build_pq_index(d=d, m=32, nbits=8, xb=xb)

    threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)  # parallel insertion makes the graph differ between runs; one thread reproduces it exactly
    hnsw = build_hnsw_index(d=d, M=32, xb=xb)
    faiss.omp_set_num_threads(threads)
    hnsw.hnsw.efSearch = 64

    return {"LSH": lsh, "PQ": pq, "HNSW": hnsw}



def plot_cases(gap, recalls, rho):
    """
    Per-query recall against how clearly the true top-10 stand out from the next 90 neighbours
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharey=True)

    for ax, method in zip(axes, ["LSH", "PQ", "HNSW"]):
        jitter = np.random.default_rng(0).uniform(-0.015, 0.015, len(gap))  # recall takes 11 values, so points overlap
        ax.scatter(gap, recalls[method] + jitter, s=14, color=COLOURS[method], alpha=0.7, edgecolor="white", linewidth=0.4)
        style_axes(ax, "similarity gap, 10th − 100th neighbour", "recall@10" if ax is axes[0] else "",
                   f"{method} ({SETTINGS[method]})")
        ax.text(0.97, 0.05, f"mean recall {recalls[method].mean():.2f}\nSpearman ρ = {rho[method]:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK_MUTED)

    fig.suptitle("Per-question recall@10 on FinanceBench (150 questions)", fontsize=11, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "fig10_cases_financebench.png")



def describe(q, question, ids, sims, texts, labels, n=3, width=220):
    lines = []
    for rank in range(n):
        i = ids[rank]
        doc, page = labels[i]
        snippet = texts[i][:width].replace("\n", " ")
        lines.append(f"    {rank + 1}. [{doc}, p.{page + 1}] (cos {sims[rank]:.3f}) {snippet}…")
    return "\n".join(lines)



def main():
    from datasets import load_dataset
    xb, xq, labels = generate_financebench_data()
    texts = load_texts(labels)
    ds = load_dataset("PatronusAI/financebench", split="train")
    d, nq = xb.shape[1], xq.shape[0]

    # exact top-100: top-10 is the ground truth, ranks 10 and 100 measure how much the neighbours stand out
    flat = build_flat_index(xb=xb, d=d)
    dist, gt = search_index(flat, noNeighbors=100, noQueries=nq, xq=xq)
    sim = 1 - dist / 2  # unit vectors: ||x - y||^2 = 2 - 2 cos
    gap = sim[:, 9] - sim[:, 99]

    indexes = build_indexes(xb, d)
    preds, recalls, rho = {}, {}, {}
    for method, index in indexes.items():
        _, preds[method] = search_index(index, noNeighbors=10, noQueries=nq, xq=xq)
        recalls[method] = per_query_recall(preds[method], gt)
        rho[method] = spearmanr(gap, recalls[method]).statistic

    plot_cases(gap, recalls, rho)

    # does the exact top-10 contain the page that holds the answer?
    def evidence_hit(q):
        pages = {(e["doc_name"], e["evidence_page_num"]) for e in ds[q]["evidence"]}
        return any(tuple(labels[i]) in pages for i in gt[q, :10])
    evidence = np.array([evidence_hit(q) for q in range(nq)])

    out = [f"per-query recall@10 on FinanceBench, {nq} questions",
           f"gap = cosine similarity of the 10th minus the 100th exact neighbour (median {np.median(gap):.3f})", ""]
    for method in recalls:
        r = recalls[method]
        low, high = gap <= np.median(gap), gap > np.median(gap)
        out.append(f"{method:5s} ({SETTINGS[method]}): mean {r.mean():.3f} | perfect {np.mean(r == 1):.0%} | "
                   f"recall <= 0.5: {np.mean(r <= 0.5):.0%} | low-gap half {r[low].mean():.3f} vs high-gap half {r[high].mean():.3f} | "
                   f"Spearman rho {rho[method]:.2f}")
    out.append(f"\nexact top-10 contains an evidence page for {evidence.mean():.0%} of questions")

    pd.DataFrame({"question": ds["question"][:nq], "company": ds["company"][:nq], "gap": gap,
                  "cos_top1": sim[:, 0], "cos_10th": sim[:, 9], "cos_100th": sim[:, 99], "evidence_in_exact_top10": evidence,
                  **{f"recall_{m}": recalls[m] for m in recalls}}).to_csv(PER_QUERY_PATH, index_label="query")

    # worked examples: highest-gap question LSH and HNSW both get right with the answer page in the exact top-10
    # (PQ at m=32 is rarely perfect, so it isn't required), lowest-gap question both LSH and HNSW miss, and
    # HNSW's worst question
    all_good = np.where((recalls["LSH"] >= 0.9) & (recalls["HNSW"] == 1.0) & evidence)[0]
    both_bad = np.where((recalls["LSH"] <= 0.3) & (recalls["HNSW"] <= 0.7))[0]
    examples = []
    if len(all_good):
        examples.append(("SUCCESS", all_good[np.argmax(gap[all_good])]))
    if len(both_bad):
        examples.append(("FAILURE (no clear neighbours)", both_bad[np.argmin(gap[both_bad])]))
    worst_hnsw = int(np.argmin(recalls["HNSW"]))
    examples.append(("FAILURE (HNSW graph search)", worst_hnsw))

    # does a longer candidate list let the greedy walk escape? (search-time only, same graph)
    hnsw = indexes["HNSW"]
    escape = []
    for ef in [64, 128, 256, 512]:
        hnsw.hnsw.efSearch = ef
        _, ids = search_index(hnsw, noNeighbors=10, noQueries=nq, xq=xq)
        escape.append(f"efSearch={ef}: {per_query_recall(ids[worst_hnsw:worst_hnsw + 1], gt[worst_hnsw:worst_hnsw + 1])[0]:.1f}")
    hnsw.hnsw.efSearch = 64
    hnsw_ids = preds["HNSW"][worst_hnsw]
    hnsw_docs = sorted({labels[i][0] for i in hnsw_ids})
    true_docs = sorted({labels[i][0] for i in gt[worst_hnsw, :10]})

    lsh_sims = {}
    for tag, q in examples:
        out += ["", "=" * 100, f"{tag}: question {q} ({ds[q]['company']}, {ds[q]['doc_name']})",
                f"  Q: {ds[q]['question']}",
                f"  gap {gap[q]:.3f} | cos top-1 {sim[q, 0]:.3f}, 10th {sim[q, 9]:.3f}, 100th {sim[q, 99]:.3f} | "
                f"evidence page in exact top-10: {evidence[q]}",
                "  recall@10: " + ", ".join(f"{m} {recalls[m][q]:.1f}" for m in recalls),
                "  exact top-3:", describe(q, ds[q]["question"], gt[q], sim[q], texts, labels)]
        lsh_ids = preds["LSH"][q]
        lsh_sim = xb[lsh_ids] @ xq[q]
        missed = [i for i in gt[q, :10] if i not in set(lsh_ids)]
        out += [f"  LSH top-3 (true cosine to the question):", describe(q, None, lsh_ids, lsh_sim, texts, labels),
                f"  LSH missed {len(missed)} of the exact top-10"]
        if q == worst_hnsw:
            out += [f"  HNSW recall for this question as efSearch grows: " + ", ".join(escape),
                    f"  HNSW top-3 at efSearch=64 (true cosine):",
                    describe(q, None, hnsw_ids, xb[hnsw_ids] @ xq[q], texts, labels),
                    f"  filings in exact top-10: {true_docs}", f"  filings in HNSW top-10: {hnsw_docs}"]

    os.makedirs(os.path.dirname(CASES_PATH), exist_ok=True)
    open(CASES_PATH, "w", encoding="utf8").write("\n".join(out) + "\n")
    print("\n".join(out))



if __name__ == "__main__":
    main()
