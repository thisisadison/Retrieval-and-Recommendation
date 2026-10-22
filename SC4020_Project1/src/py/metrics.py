import os
import pandas as pd
import numpy as np
import time
import faiss
import statistics

from methods import *

VERBOSE = False  # set True to see the per-step prints

RESULTS = []  # one dict per configuration, written to csv by save_results
LAST = {"build": None, "flat_latency": None, "flat_size": None, "flat_build": None}  # latest build time and flat baseline



# Precision@k = (how many of your k guesses were correct) / (k, the number of guesses you made)
def precision_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """
    Get precision@k

    :param pred_ids: ids of predicted vectors
    :return gt_ids: ids of ground truth vectors
    :param k: number of nearest neighbors
    :return: precision@k 
    """
    if VERBOSE:
        print("\n=== Get Precision@k ===")

    n_queries = gt_ids.shape[0]
    precision_arr = []

    for i in range (n_queries):
        # Calculate precision for each query
        set_pred = set(pred_ids[i][:k])
        set_gt = set(gt_ids[i][:k])
        count = len(set_pred.intersection(set_gt))
        precision = count / k

        precision_arr.append(precision)

    precision = np.mean(precision_arr)

    if VERBOSE:
        print(f"\nprecision@{k}: {precision:.4f}")

    return precision



# Recall@k = (how many of your k guesses were correct) / (the total number of correct answers that exist)
def recall_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """
    Get recall@k
    
    :param pred_ids: ids of predicted vectors
    :return gt_ids: ids of ground truth vectors
    :param k: number of nearest neighbors
    :return: recall@k 
    """
    if VERBOSE:
        print("\n=== Get Recall@k ===")

    n_queries = gt_ids.shape[0]
    recall_arr = []

    for i in range (n_queries):
        # Calculate recall for each query
        set_pred = set(pred_ids[i][:k])
        set_gt = set(gt_ids[i][:k])
        count = len(set_pred.intersection(set_gt))
        recall = count / k

        recall_arr.append(recall)

    recall = np.mean(recall_arr)

    if VERBOSE:
        print(f"\nrecall@{k}: {recall:.4f}")

    return recall



def measure_latency(index, xq: np.ndarray, k: int, n_repeats=10)-> float:
    """
    Measure latency of FAISS index search

    :param index: FAISS index
    :param xq: query dimensions, np array
    :param k: number of nearest neighbors
    :param n_repeats: number of repeats for averaging
    :return: average latency in seconds
    """
    if VERBOSE:
        print("\n=== Get Latency ===")

    time_list = []
    for _ in range(n_repeats):
        start_time = time.perf_counter() # record start time
        index.search(xq, k) # search k nearest neighbors for xq
        end_time = time.perf_counter() # record end time
        time_list.append(end_time - start_time)

    median_val = statistics.median(time_list) # get the median time 

    if VERBOSE:
        print(f"\nMedian latency for {n_repeats} repeats: {median_val:.6f} seconds")

    return median_val



def measure_index_size(index) -> int:
    """
    Measure size of FAISS index in bytes

    :param index: FAISS index
    :return: size of index in bytes
    """
    if VERBOSE:
        print("\n=== Get Index Size ===")

    serialized_index = faiss.serialize_index(index) # serialize index to numpy byte array
    size_in_bytes = serialized_index.nbytes
    size_in_mb = size_in_bytes / (1024 * 1024)

    if VERBOSE:
        print("\nIndex size (MB):", size_in_mb)  

    return size_in_mb



def measure_build_time(build_fn, **kwargs):
    """
    Measure time taken to build a FAISS index

    :param build_fn: index builder function from methods.py
    :param kwargs: arguments passed to the builder
    :return: whatever the builder returns
    """
    if VERBOSE:
        print("\n=== Get Build Time ===")

    start_time = time.perf_counter() # record start time
    result = build_fn(**kwargs) # build index, includes training where applicable
    end_time = time.perf_counter() # record end time

    build_time = end_time - start_time
    LAST["build"] = build_time  # picked up by get_metrics, so reused indexes keep their build time

    print(f"build time: {build_time:.6f} seconds")

    return result



def set_baseline(latency: float, size: float):
    """
    Remember the flat (exact) baseline of the current run, so every result row
    can be compared against the flat index built on the same data

    :param latency: flat search latency in seconds
    :param size: flat index size in MB
    """
    LAST["flat_latency"] = latency
    LAST["flat_size"] = size
    LAST["flat_build"] = LAST["build"]



def get_metrics(pred_ids: np.ndarray, gt_ids: np.ndarray, index, xq: np.ndarray, k: int, n_repeats=10, dataset=None, method=None, **params):
    """
    Print metrics for search and index performance, and store them as one row of RESULTS

    :param dataset: dataset name, e.g. "random"
    :param method: method name, e.g. "PQ"
    :param params: method parameters, e.g. m=32, nbits=8
    """
    recall = recall_at_k(pred_ids, gt_ids, k)
    latency = measure_latency(index, xq, k, n_repeats)
    size = measure_index_size(index)

    print(f"recall@{k}: {recall:.4f} | latency: {latency:.6f}s | index size: {size:.4f}MB\n")

    if dataset is not None:
        RESULTS.append({"dataset": dataset, "method": method, **params,
                        "recall": recall, "latency": latency, "size": size, "build": LAST["build"],
                        "flat_latency": LAST["flat_latency"], "flat_size": LAST["flat_size"], "flat_build": LAST["flat_build"]})

    return recall, latency, size



def save_results(path: str):
    """
    Write RESULTS to csv. Rows already in the file for the same dataset and method
    are replaced, so parts of main can be rerun on their own

    :param path: csv file path
    """
    new = pd.DataFrame(RESULTS)

    if os.path.exists(path):
        old = pd.read_csv(path)
        rerun = old.set_index(["dataset", "method"]).index.isin(new.set_index(["dataset", "method"]).index)
        new = pd.concat([old[~rerun], new], ignore_index=True)

    new.to_csv(path, index=False)
    print(f"saved {len(RESULTS)} results to {path}")



# def main():
#     test_metrics()

# if __name__ == "__main__":
#     main()