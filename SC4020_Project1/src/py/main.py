import os

from methods import *
from dataset import *
from metrics import *



D = 384                # vector dimension for every dataset, set by the all-MiniLM-L6-v2 embedding size
SYNTHETIC_NB = 100000  # synthetic vector db size
SYNTHETIC_NQ = 100     # synthetic query count
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "txt", "results.csv")



def run_lsh_clustered():
    """
    Use clustered data for lsh search
    """
    xb_clustered, xq_clustered = generate_clustered_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, n_features=D)

    print(f"\n========== Flat (clustered data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_clustered, d=D)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
    flat_latency = measure_latency(flat_index, xq_clustered, k=10)
    flat_size = measure_index_size(flat_index)
    set_baseline(flat_latency, flat_size)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [256, 512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits (clustered data) ==========")

        lsh_index = measure_build_time(build_lsh_index, d=D, nbits=nbits, xb=xb_clustered)
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_clustered, k=10, dataset="clustered", method="LSH", nbits=nbits)



def run_lsh_random():
    """
    Use random data for lsh search
    """
    xb_random, xq_random = generate_random_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, d=D)

    print(f"\n========== Flat (random data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_random, d=D)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_random)
    flat_latency = measure_latency(flat_index, xq_random, k=10)
    flat_size = measure_index_size(flat_index)
    set_baseline(flat_latency, flat_size)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [256, 512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits (random data) ==========")

        lsh_index = measure_build_time(build_lsh_index, d=D, nbits=nbits, xb=xb_random)
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_random)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_random, k=10, dataset="random", method="LSH", nbits=nbits)



def run_pq_random():
    """
    Use random data for pq search
    """
    xb_random, xq_random = generate_random_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, d=D)

    print(f"\n========== Flat (random data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_random, d=D)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_random)
    flat_latency = measure_latency(flat_index, xq_random, k=10)
    flat_size = measure_index_size(flat_index)
    set_baseline(flat_latency, flat_size)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [6, 8, 10]:

        print(f"\n========== PQ m=32, nbits={nbits} (random data) ==========")

        pq_index = measure_build_time(build_pq_index, d=D, m=32, nbits=nbits, xb=xb_random)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_random)
        get_metrics(pred_ids, gt_ids, pq_index, xq_random, k=10, dataset="random", method="PQ", m=32, nbits=nbits)

    # m=32 at nbits=10 already measured in the loop above
    for m in [8, 16, 64, 192]:

        print(f"\n========== PQ m={m}, nbits=10 (random data) ==========")

        pq_index = measure_build_time(build_pq_index, d=D, m=m, nbits=10, xb=xb_random)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_random)
        get_metrics(pred_ids, gt_ids, pq_index, xq_random, k=10, dataset="random", method="PQ", m=m, nbits=10)



def run_pq_clustered():
    """
    Use clustered data for pq search
    """
    xb_clustered, xq_clustered = generate_clustered_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, n_features=D)

    print(f"\n========== Flat (clustered data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_clustered, d=D)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
    flat_latency = measure_latency(flat_index, xq_clustered, k=10)
    flat_size = measure_index_size(flat_index)
    set_baseline(flat_latency, flat_size)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [6, 8, 10]:

        print(f"\n========== PQ m=32, nbits={nbits} (clustered data) ==========")

        pq_index = measure_build_time(build_pq_index, d=D, m=32, nbits=nbits, xb=xb_clustered)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
        get_metrics(pred_ids, gt_ids, pq_index, xq_clustered, k=10, dataset="clustered", method="PQ", m=32, nbits=nbits)

    # m=32 at nbits=10 already measured in the loop above
    for m in [8, 16, 64, 192]:

        print(f"\n========== PQ m={m}, nbits=10 (clustered data) ==========")

        pq_index = measure_build_time(build_pq_index, d=D, m=m, nbits=10, xb=xb_clustered)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
        get_metrics(pred_ids, gt_ids, pq_index, xq_clustered, k=10, dataset="clustered", method="PQ", m=m, nbits=10)



def run_hnsw_random():
    """
    Use random data for hnsw search
    """
    xb_random, xq_random = generate_random_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, d=D)

    print(f"\n========== Flat (random data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_random, d=D)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_random)
    flat_latency = measure_latency(flat_index, xq_random, k=10)
    flat_size = measure_index_size(flat_index)
    set_baseline(flat_latency, flat_size)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    # efSearch is a search-time knob, so one index serves the whole sweep
    hnsw_index = measure_build_time(build_hnsw_index, d=D, M=32, xb=xb_random)

    for efSearch in [16, 32, 64, 128, 256]:

        print(f"\n========== HNSW M=32, efSearch={efSearch} (random data) ==========")

        hnsw_index.hnsw.efSearch = efSearch
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_random)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_random, k=10, dataset="random", method="HNSW", M=32, efSearch=efSearch)

    # M=32 at efSearch=64 already measured in the loop above
    for M in [8, 16, 64]:

        print(f"\n========== HNSW M={M}, efSearch=64 (random data) ==========")

        hnsw_index = measure_build_time(build_hnsw_index, d=D, M=M, xb=xb_random)
        hnsw_index.hnsw.efSearch = 64
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_random)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_random, k=10, dataset="random", method="HNSW", M=M, efSearch=64)



def run_hnsw_clustered():
    """
    Use clustered data for hnsw search
    """
    xb_clustered, xq_clustered = generate_clustered_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, n_features=D)

    print(f"\n========== Flat (clustered data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_clustered, d=D)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
    flat_latency = measure_latency(flat_index, xq_clustered, k=10)
    flat_size = measure_index_size(flat_index)
    set_baseline(flat_latency, flat_size)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    # efSearch is a search-time knob, so one index serves the whole sweep
    hnsw_index = measure_build_time(build_hnsw_index, d=D, M=32, xb=xb_clustered)

    for efSearch in [16, 32, 64, 128, 256]:

        print(f"\n========== HNSW M=32, efSearch={efSearch} (clustered data) ==========")

        hnsw_index.hnsw.efSearch = efSearch
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_clustered, k=10, dataset="clustered", method="HNSW", M=32, efSearch=efSearch)

    # M=32 at efSearch=64 already measured in the loop above
    for M in [8, 16, 64]:

        print(f"\n========== HNSW M={M}, efSearch=64 (clustered data) ==========")

        hnsw_index = measure_build_time(build_hnsw_index, d=D, M=M, xb=xb_clustered)
        hnsw_index.hnsw.efSearch = 64
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_clustered, k=10, dataset="clustered", method="HNSW", M=M, efSearch=64)



def run_pq_finance():
    """
    Use financebench filing chunks for pq search
    """
    xb_fin, xq_fin, _ = generate_financebench_data()
    d = xb_fin.shape[1]  # 384 for all-MiniLM-L6-v2
    nq = xq_fin.shape[0]  # all 150 questions


    print(f"\n========== Flat (financebench) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_fin, d=d)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=nq, xq=xq_fin)
    flat_latency = measure_latency(flat_index, xq_fin, k=10)
    flat_size = measure_index_size(flat_index)
    set_baseline(flat_latency, flat_size)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [6, 8, 10]:

        print(f"\n========== PQ m=32, nbits={nbits} (financebench) ==========")

        pq_index = measure_build_time(build_pq_index, d=d, m=32, nbits=nbits, xb=xb_fin)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=nq, xq=xq_fin)
        get_metrics(pred_ids, gt_ids, pq_index, xq_fin, k=10, dataset="financebench", method="PQ", m=32, nbits=nbits)

    # m=32 at nbits=10 already measured in the loop above
    for m in [8, 16, 64, 192]:

        print(f"\n========== PQ m={m}, nbits=10 (financebench) ==========")

        pq_index = measure_build_time(build_pq_index, d=d, m=m, nbits=10, xb=xb_fin)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=nq, xq=xq_fin)
        get_metrics(pred_ids, gt_ids, pq_index, xq_fin, k=10, dataset="financebench", method="PQ", m=m, nbits=10)



def run_lsh_finance():
    """
    Use financebench filing chunks for lsh search
    """
    xb_fin, xq_fin, _ = generate_financebench_data()
    d = xb_fin.shape[1]  # 384 for all-MiniLM-L6-v2
    nq = xq_fin.shape[0]  # all 150 questions


    print(f"\n========== Flat (financebench) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_fin, d=d)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=nq, xq=xq_fin)
    flat_latency = measure_latency(flat_index, xq_fin, k=10)
    flat_size = measure_index_size(flat_index)
    set_baseline(flat_latency, flat_size)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [256, 512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits (financebench) ==========")

        lsh_index = measure_build_time(build_lsh_index, d=d, nbits=nbits, xb=xb_fin)
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=nq, xq=xq_fin)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_fin, k=10, dataset="financebench", method="LSH", nbits=nbits)



def run_hnsw_finance():
    """
    Use financebench filing chunks for hnsw search
    """
    xb_fin, xq_fin, _ = generate_financebench_data()
    d = xb_fin.shape[1]  # 384 for all-MiniLM-L6-v2
    nq = xq_fin.shape[0]  # all 150 questions


    print(f"\n========== Flat (financebench) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_fin, d=d)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=nq, xq=xq_fin)
    flat_latency = measure_latency(flat_index, xq_fin, k=10)
    flat_size = measure_index_size(flat_index)
    set_baseline(flat_latency, flat_size)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    # efSearch is a search-time knob, so one index serves the whole sweep
    hnsw_index = measure_build_time(build_hnsw_index, d=d, M=32, xb=xb_fin)

    for efSearch in [16, 32, 64, 128, 256]:

        print(f"\n========== HNSW M=32, efSearch={efSearch} (financebench) ==========")

        hnsw_index.hnsw.efSearch = efSearch
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=nq, xq=xq_fin)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_fin, k=10, dataset="financebench", method="HNSW", M=32, efSearch=efSearch)

    # M=32 at efSearch=64 already measured in the loop above
    for M in [8, 16, 64]:

        print(f"\n========== HNSW M={M}, efSearch=64 (financebench) ==========")

        hnsw_index = measure_build_time(build_hnsw_index, d=d, M=M, xb=xb_fin)
        hnsw_index.hnsw.efSearch = 64
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=nq, xq=xq_fin)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_fin, k=10, dataset="financebench", method="HNSW", M=M, efSearch=64)



def load_all_datasets():
    """
    Load every dataset for the ablations, same settings as the main runs

    :return: dict of dataset name -> (xb, xq)
    """
    return {
        "random": generate_random_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, d=D),
        "clustered": generate_clustered_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, n_features=D),
        "financebench": generate_financebench_data()[:2],  # labels not needed
    }



def run_flat_baseline(dataset: str, xb: np.ndarray, xq: np.ndarray):
    """
    Exact search on one dataset: gives the ground truth and the baseline every row is compared against

    :return: gt_ids, the true top-10 of every query
    """
    d = xb.shape[1]
    nq = xq.shape[0]

    print(f"\n========== Flat ({dataset}) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb, d=d)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=nq, xq=xq)
    flat_latency = measure_latency(flat_index, xq, k=10)
    flat_size = measure_index_size(flat_index)
    set_baseline(flat_latency, flat_size)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    return gt_ids



def run_lsh_ablation():
    """
    Ablation: lsh with each bit's threshold learned from the data (the median projection)
    instead of fixed at 0. The standard lsh runs give the other half of the comparison
    """
    for dataset, (xb, xq) in load_all_datasets().items():
        d = xb.shape[1]
        nq = xq.shape[0]
        gt_ids = run_flat_baseline(dataset, xb, xq)

        for nbits in [256, 512, 1024, 2048, 4096]:

            print(f"\n========== LSH-trained {nbits} bits ({dataset}) ==========")

            lsh_index = measure_build_time(build_lsh_index, d=d, nbits=nbits, xb=xb, train_thresholds=True)
            _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=nq, xq=xq)
            get_metrics(pred_ids, gt_ids, lsh_index, xq, k=10, dataset=dataset, method="LSH-trained", nbits=nbits)



def run_ivf_ablation():
    """
    Ablation: pq with and without an inverted file (ivf) in front of it, at the same
    code size. Without ivf every code is scanned; with ivf only the nprobe cells
    nearest the query are, and codes store each vector's offset from its cell centre
    """
    m, nbits, nlist = 192, 8, 1024

    for dataset, (xb, xq) in load_all_datasets().items():
        d = xb.shape[1]
        nq = xq.shape[0]
        gt_ids = run_flat_baseline(dataset, xb, xq)

        print(f"\n========== PQ m={m}, nbits={nbits}, no IVF ({dataset}) ==========")

        pq_index = measure_build_time(build_pq_index, d=d, m=m, nbits=nbits, xb=xb)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=nq, xq=xq)
        get_metrics(pred_ids, gt_ids, pq_index, xq, k=10, dataset=dataset, method="PQ-noIVF", m=m, nbits=nbits)

        # nprobe is a search-time knob, so one index serves the whole sweep
        ivf_index = measure_build_time(build_ivfpq_index, d=d, nlist=nlist, m=m, nbits=nbits, xb=xb)

        for nprobe in [1, 8, 32, 128]:

            print(f"\n========== IVF-PQ nlist={nlist}, nprobe={nprobe}, m={m}, nbits={nbits} ({dataset}) ==========")

            ivf_index.nprobe = nprobe
            _, pred_ids = search_index(ivf_index, noNeighbors=10, noQueries=nq, xq=xq)
            get_metrics(pred_ids, gt_ids, ivf_index, xq, k=10, dataset=dataset, method="IVF-PQ", m=m, nbits=nbits, nlist=nlist, nprobe=nprobe)



def run_all_synthetic():
    # every dataset arrives unit length, and every method uses l2 for ground truth and search
    run_lsh_random()
    run_lsh_clustered()

    run_pq_random()
    run_pq_clustered()

    run_hnsw_random()
    run_hnsw_clustered()



def run_all_finance():
    # every dataset arrives unit length, and every method uses l2 for ground truth and search
    run_lsh_finance()
    run_pq_finance()
    run_hnsw_finance()



def main():
    run_all_synthetic()
    run_all_finance()
    run_lsh_ablation()
    run_ivf_ablation()
    save_results(RESULTS_PATH)



if __name__ == "__main__":
    main()