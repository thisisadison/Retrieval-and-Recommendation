import faiss
import numpy as np

from main import D, SYNTHETIC_NB, SYNTHETIC_NQ
from dataset import generate_random_data, generate_clustered_data, generate_financebench_data
from metrics import recall_at_k



def build_lsh(xb: np.ndarray, nbits: int, learned: bool):
    """
    Same LSH index as the ablation: split at 0, or at the median projection when learned=True
    """
    index = faiss.IndexLSH(xb.shape[1], nbits, True, learned)
    if learned:
        index.train(xb)
    index.add(xb)

    return index



def bit_balance(xb: np.ndarray, nbits: int, learned: bool, n=20000) -> float:
    """
    How evenly the hyperplanes split the database

    :return: mean |p - 0.5| over bits, p = share of vectors with bit 1 (0 = every bit splits the data in half)
    """
    index = build_lsh(xb, nbits, learned)
    proj = index.rrot.apply_py(xb[:n])  # projection onto every hyperplane normal
    thresholds = faiss.vector_to_array(index.thresholds) if learned else np.zeros(nbits, dtype="float32")
    p = (proj > thresholds).mean(axis=0)

    return float(np.abs(p - 0.5).mean())



def lsh_recall(xb: np.ndarray, xq: np.ndarray, nbits: int, learned: bool) -> float:
    flat = faiss.IndexFlatL2(xb.shape[1])
    flat.add(xb)
    _, gt_ids = flat.search(xq, 10)
    _, pred_ids = build_lsh(xb, nbits, learned).search(xq, 10)

    return recall_at_k(pred_ids, gt_ids, 10)



def main():
    """
    Diagnostics for the learned-threshold ablation: why median thresholds help clustered data
    but hurt FinanceBench
    """
    xb_fin, xq_fin, _ = generate_financebench_data()
    data = {
        "random": generate_random_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, d=D),
        "clustered": generate_clustered_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, n_features=D),
        "financebench": (xb_fin, xq_fin),
    }

    print("\n== bit balance, nbits=1024: mean |p - 0.5| ==")
    for name, (xb, _) in data.items():
        print(f"{name:13s} zero split {bit_balance(xb, 1024, False):.3f} | learned {bit_balance(xb, 1024, True):.3f}")

    # questions are not passages: compare against held-out passages used as queries, removed from the db
    # so they can't match themselves
    rng = np.random.default_rng(0)
    held = rng.choice(len(xb_fin), size=len(xq_fin), replace=False)
    keep = np.ones(len(xb_fin), dtype=bool)
    keep[held] = False

    mean_q, mean_b = xq_fin.mean(axis=0), xb_fin.mean(axis=0)
    print(f"\ncosine(mean question, mean passage) = {mean_q @ mean_b / np.linalg.norm(mean_q) / np.linalg.norm(mean_b):.3f}")

    print("\n== financebench recall@10: question queries vs passage queries ==")
    for nbits in [1024, 4096]:
        for qname, xb, xq in [("questions", xb_fin, xq_fin), ("passages", xb_fin[keep], xb_fin[held])]:
            zero, learned = lsh_recall(xb, xq, nbits, False), lsh_recall(xb, xq, nbits, True)
            print(f"nbits={nbits:<5} {qname:9s} zero {zero:.3f} | learned {learned:.3f} | change {learned - zero:+.3f}")



if __name__ == "__main__":
    main()
