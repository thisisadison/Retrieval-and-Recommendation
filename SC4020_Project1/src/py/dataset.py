import numpy as np

from sklearn.datasets._samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

VERBOSE = False  # set True to see the per-step prints



# get fake vector db and query for search
# xb is for vector db
# xq is for query
def generate_random_data(nb=150000, nq=100, d=64):
    """
    Generate numpy array for vector db and query
    
    :param nb: no of vectors in vector db
    :param nq: no of queries
    :param d: dimension of vectors 
    :return: xb numpy array for vector db, xq numpy array for query
    """
    if VERBOSE:
        print("\n=== Generate Random Data ===")

    # float32 for faiss
    np.random.seed(1234)  # make reproducible

    xb_raw = np.random.randn(nb, d)
    xq_raw = np.random.randn(nq, d)

    xb = normalize(xb_raw, norm='l2').astype('float32')
    xq = normalize(xq_raw, norm='l2').astype('float32')

    return xb, xq



def generate_clustered_data(nb=128000, nq=100, centers=6, n_features=64, random_state=42):
    """
    Create clustered data for vector search benchmarking.
    
    :param nb: Number of database/index vectors
    :param nq: Number of query vectors
    :param centers: Number of cluster centers
    :param n_features: Number of features (dimensions) per vector
    :param random_state: Seed for reproducibility
    :return: xb (database array), xq (query array)
    """
    if VERBOSE:
        print("\n=== Generate Clustered Data ===")

    total_samples = nb + nq
    X_raw, y = make_blobs(n_samples=total_samples, centers=centers, n_features=n_features, random_state=random_state)

    np.random.seed(random_state)
    transformation = np.random.randn(n_features, n_features)
    X_correlated = np.dot(X_raw, transformation) # Stretches the perfect spheres into ellipses

    xb, xq = train_test_split(X_correlated, train_size=nb, test_size=nq, shuffle=True, random_state=random_state)

    # unit length, like the random and financebench vectors, so l2 ranks neighbours the same as cosine
    xb = normalize(xb, norm='l2').astype('float32')
    xq = normalize(xq, norm='l2').astype('float32')

    return xb, xq



def chunk_text(text: str, chunk_size=500, overlap=100):
    """
    Split a long text into overlapping fixed-length passages

    :param text: text to split
    :param chunk_size: characters per chunk
    :param overlap: characters shared between consecutive chunks
    :return: list of chunks
    """
    text = " ".join(text.split()) # reduce space and newlines to single space

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    return chunks



def generate_financebench_data(pdf_dir="data/financebench_pdfs", chunk_size=500, overlap=100,
                               model_name="all-MiniLM-L6-v2", cache_dir="data"):
    """
    Generate real embedding vectors from the full FinanceBench filings. Vectors are returned
    unit length: the encoder is trained with a cosine objective, so only direction carries
    meaning, and on unit vectors l2 ranks neighbours the same as cosine.

    :param pdf_dir: folder holding the downloaded filings
    :param chunk_size: characters per chunk
    :param overlap: characters shared between consecutive chunks
    :param model_name: sentence-transformers model to convert text to vectors
    :param cache_dir: folder to save embeddings in, so later runs skip re-embedding
    :return: xb filing chunk vectors, xq question vectors, labels (doc_name, page) for each chunk in xb
    """
    import os
    from datasets import load_dataset  # kept local so dataset.py imports without these installed
    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer

    print("\n=== Generate FinanceBench Data ===")

    xb_path = os.path.join(cache_dir, "financebench_xb.npy")
    xq_path = os.path.join(cache_dir, "financebench_xq.npy")
    labels_path = os.path.join(cache_dir, "financebench_labels.npy")

    if all(os.path.exists(p) for p in (xb_path, xq_path, labels_path)):
        print("\nloading cached embeddings")
        return np.load(xb_path), np.load(xq_path), np.load(labels_path, allow_pickle=True)

    texts, labels, seen = [], [], set()

    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.endswith(".pdf"):
            continue

        doc_name = fname[:-4]
        try:
            reader = PdfReader(os.path.join(pdf_dir, fname))
        except Exception as e:
            print(f"skipped {doc_name}: {e}")  # one unreadable filing shouldn't stop the whole run
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
                labels.append((doc_name, page_num))

        print(f"{doc_name}: {len(reader.pages)} pages")

    print(f"\ntexts to embed: {len(texts)}")

    model = SentenceTransformer(model_name)  # maps each string to a fixed length vector

    xb = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype("float32")

    ds = load_dataset("PatronusAI/financebench", split="train")
    xq = model.encode(ds["question"], batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype("float32")  # the 150 real analyst questions are the queries

    labels = np.array(labels, dtype=object)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(xb_path, xb)
    np.save(xq_path, xq)
    np.save(labels_path, labels)

    print(f"\nxb shape: {xb.shape} | xq shape: {xq.shape}")

    return xb, xq, labels



def download_financebench_pdfs(out_dir="data/financebench_pdfs"):
    """
    Download the source filings referenced by the 150 open-source FinanceBench questions

    :param out_dir: folder to save the pdfs into
    :return: out_dir
    """
    import os
    import urllib.request
    from datasets import load_dataset  # kept local so dataset.py imports without it installed

    ds = load_dataset("PatronusAI/financebench", split="train")
    doc_names = sorted(set(ds["doc_name"]))  # each filing once, even if several questions use it

    print(f"\n=== Download FinanceBench PDFs: {len(doc_names)} filings ===")

    os.makedirs(out_dir, exist_ok=True)
    base_url = "https://github.com/patronus-ai/financebench/raw/main/pdfs/"

    for i, name in enumerate(doc_names, 1):
        path = os.path.join(out_dir, name + ".pdf")
        if os.path.exists(path):
            continue  # skip files already downloaded, so a rerun resumes instead of restarting

        urllib.request.urlretrieve(base_url + name + ".pdf", path)
        print(f"[{i}/{len(doc_names)}] {name}")

    return out_dir