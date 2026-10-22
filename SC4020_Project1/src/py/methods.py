import faiss
import numpy as np



def add_vectors(index, xb: np.ndarray):
    """
    Populate FAISS index with vectors

    :param index: FAISS index
    :param xb: vector db dimensions, np array
    """
    index.add(xb)



def search_index(index, noNeighbors: int, noQueries: int, xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for nearest neighbors

    :param index: FAISS index
    :param noNeighbors: no of nearest neighbors
    :param noQueries: no of queries
    :param xq: query dimensions, np array
    :return: d distance between query and neighbors, i index of neighbors
    """
    d, i = index.search(xq[:noQueries], noNeighbors)

    return d, i



def build_flat_index(xb: np.ndarray, d: int):
    """
    Build FAISS flat index for vector search using squared l2 distance

    :param xb: index numpy array
    :param d: dimension of vectors
    :return: vector db
    """
    # print("\n=== Build Flat Index ===")
    index = faiss.IndexFlatL2(d);  # build index
    # print("\nTrained:", index.is_trained)

    add_vectors(index, xb)

    return index



def build_lsh_index(d: int, nbits: int, xb: np.ndarray, train_thresholds=False):
    """
    Build LSH index: generates nbits random hyperplanes and hashes each
    vector into an nbits-length binary code based on which side of each
    hyperplane it falls on.

    :param d: dimension of vectors
    :param nbits: number of random hash functions (hyperplanes) / bits per code
    :param train_thresholds: False splits every hyperplane at 0 (standard LSH),
                             True learns each split point from the data (the median projection)
    :return: empty LSH index, ready for add_vectors
    """
    # print("\n=== Build LSH Index ===")
    index = faiss.IndexLSH(d, nbits, True, train_thresholds)
    # print("\nTrained:", index.is_trained)

    if train_thresholds:
        index.train(xb)  # one median per bit, so each hyperplane splits the data in half

    add_vectors(index, xb)

    return index



def build_pq_index(d: int, m: int, nbits: int, xb: np.ndarray):
    """
    Build compressed vector for Product Quantization: splits each vector into m
    subspaces, learns nbits-worth of centroids per subspace via k-means,
    and stores compressed centroid-ID codes instead of full vectors.

    :param d: dimension of vectors
    :param m: number of subspaces
    :param nbits: bits per subquantizer code
    :param xb: vector db
    :return: trained, empty PQ index
    """
    index = faiss.IndexPQ(d, m, nbits)  # build index, l2 by default
    index.train(xb) # k-means performed, create codebook

    add_vectors(index, xb)

    return index



def build_ivfpq_index(d: int, nlist: int, m: int, nbits: int, xb: np.ndarray):
    """
    Build IVF-PQ index: k-means splits the vector db into nlist cells, each vector is
    stored as a PQ code of its offset (residual) from its cell centre, and a search
    only scans the nprobe cells nearest the query instead of every code

    :param d: dimension of vectors
    :param nlist: number of cells
    :param m: number of subspaces
    :param nbits: bits per subquantizer code
    :param xb: vector db
    :return: trained IVF-PQ index, set index.nprobe before searching
    """
    quantizer = faiss.IndexFlatL2(d)  # holds the cell centres, finds the nearest cells for a vector
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    index.train(xb)  # k-means for the cell centres, then k-means for the pq codebooks

    add_vectors(index, xb)

    return index



def build_hnsw_index(d: int, M: int, xb: np.ndarray, efConstruction=40):
    """
    Build HNSW index: builds a multi-layer proximity graph where each vector
    links to M neighbours per layer, and search greedily traverses the graph
    from an entry point.

    :param d: dimension of vectors
    :param M: number of graph links per node
    :param xb: vector db
    :param efConstruction: candidate list size during graph construction
    :return: HNSW index
    """
    index = faiss.IndexHNSWFlat(d, M)  # l2 by default
    index.hnsw.efConstruction = efConstruction

    add_vectors(index, xb)

    return index