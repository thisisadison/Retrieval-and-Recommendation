# HNSW — End-to-End Theory

## 1. Overview

Hierarchical Navigable Small World (HNSW) is a graph-based approximate nearest-neighbour (ANN) method. Instead of compressing vectors (like PQ) or hashing them (like LSH), it keeps the original vectors and connects each one to a small number of nearby vectors, forming a graph.

The main idea is:

```text
Original vectors
      ↓
Link each vector to its nearby vectors
      ↓
Multi-layer graph
      ↓
Walk the graph towards the query
      ↓
Approximate nearest neighbours
```

The key property of HNSW is that a search only visits a small part of the database, because every step of the walk moves closer to the query.

---

# 2. Input Vector

Suppose each input vector has `d` dimensions:

```text
x = [x₁, x₂, ..., x_d]
```

For example, with:

```text
d = 384
```

each vector is represented as:

```text
x ∈ R³⁸⁴
```

Unlike LSH and PQ, HNSW stores these vectors **exactly**. It does not compress them.

---

# 3. The Graph Idea

Each vector becomes a **node** in a graph. Each node is connected by **edges** to some of its nearest neighbours.

Conceptually:

```text
      A ─── B ─── C
      │     │     │
      D ─── E ─── F
            │
            G
```

If you start at any node and repeatedly move to the neighbour closest to the query, you gradually approach the query's true nearest neighbours.

This is called **greedy search**.

---

# 4. The Problem with a Single Graph

With one flat graph, a search that starts far from the query must take many small steps to reach it.

For example:

```text
Start ── step ── step ── step ── step ── step ── Query
```

Each step only moves to a nearby neighbour, so crossing the whole space is slow.

---

# 5. The Hierarchy (Layers)

HNSW solves this by building **several layers** of graphs, similar to a skip list or a road network:

```text
Layer 2:   A ──────────────────── F                (few nodes, long links)
           │                      │
Layer 1:   A ──────── C ───────── F ──── H         (more nodes)
           │          │           │      │
Layer 0:   A ── B ── C ── D ── E ── F ── G ── H    (all nodes, short links)
```

- **Top layers** contain few nodes, so each link covers a long distance, like motorways.
- **Layer 0** contains every node with short links, like local streets.

A search uses the top layers to move quickly to the right region, then the bottom layer to find the exact neighbours.

---

# 6. Assigning Each Vector a Layer

When a vector is inserted, it is randomly assigned a **top layer** `l`:

```text
l = ⌊ −ln(U) × m_L ⌋,   U ~ Uniform(0, 1)
```

where:

```text
m_L = 1 / ln(M)
```

This makes higher layers exponentially rarer:

```text
Layer 0 → every vector
Layer 1 → about 1 in M vectors
Layer 2 → about 1 in M² vectors
...
```

A vector assigned to layer `l` appears in **every layer from `l` down to 0**.

For example, with `M = 32` and 100,000 vectors:

```text
Layer 0 → 100,000 vectors
Layer 1 → about 3,000 vectors
Layer 2 → about 100 vectors
Layer 3 → about 3 vectors
```

---

# 7. Building the Graph (Insertion)

Vectors are inserted one at a time. To insert a new vector `x`:

1. Start at the **entry point** (a node on the top layer).
2. On each layer above `x`'s assigned layer, greedily walk to the node closest to `x`.
3. On each layer from `x`'s layer down to 0:
   - run a wider search that keeps a candidate list of size `efConstruction`;
   - connect `x` to up to `M` of the closest candidates (up to `2M` on layer 0);
   - add the reverse links too, trimming any neighbour that now has too many links.

Conceptually:

```text
New vector x
      ↓
Greedy walk down the upper layers
      ↓
Search for x's nearest nodes (candidate list = efConstruction)
      ↓
Link x to its M closest nodes on each of its layers
```

Building the graph is itself a series of searches, which is why HNSW builds faster on data that is easy to search.

---

# 8. Choosing the Neighbours (Heuristic)

HNSW does not simply link each node to its `M` closest nodes. It prefers neighbours that point in **different directions**.

A candidate is skipped if it is closer to an already-chosen neighbour than it is to `x`:

```text
      x ──── a
       ╲
        ╲ b   (b is right next to a → skipped)
         ╲
          c   (c points a different way → kept)
```

This keeps links spread out, so the graph stays connected across clusters and the search does not get trapped in one region.

---

# 9. Searching the Graph

To find the `k` nearest neighbours of a query `q`:

**Step 1: upper layers (greedy)**

Start at the entry point on the top layer. Move to whichever neighbour is closest to `q`, and repeat until no neighbour is closer. Then drop down one layer and continue from the same node.

```text
Layer 2:   entry ──→ closer node
                         ↓
Layer 1:            closer ──→ closer
                                  ↓
Layer 0:                     start the detailed search here
```

**Step 2: layer 0 (beam search)**

On layer 0, keep a **candidate list** of the `efSearch` best nodes found so far:

```text
repeat:
    take the closest unexplored candidate
    compute the distance from q to each of its neighbours
    add any neighbour closer than the worst candidate in the list
until no candidate can improve the list
```

**Step 3: return**

Return the `k` closest nodes in the candidate list.

---

# 10. Comparing Distances

Because HNSW stores the original vectors, every distance it computes is **exact**:

```text
distance(q, x) = ‖q − x‖²     (squared L2)
```

The approximation does not come from the distances, which are exact, but from **which** vectors get compared: the search only visits part of the graph.

---

# 11. Why the Search Is Approximate

The greedy walk only follows links. If the true nearest neighbour is not reachable by moving "closer" at every step, the search can stop at a **local minimum**:

```text
Query ★

    ●  ← search stops here (no neighbour is closer)

              ● ← true nearest neighbour, reachable only by first moving away
```

Therefore:

```text
Exact nearest neighbours
        ≠
HNSW approximate neighbours
```

in general. A larger candidate list (`efSearch`) makes this less likely, because more paths are explored at once.

This is also why HNSW struggles on data with **no structure**, where every point is almost equally far from the query: no step clearly moves "closer", so the walk has nothing to follow.

---

# 12. Role of `M`

`M` is the number of links per node on each upper layer (`2M` on layer 0).

```text
Larger M
    ↓
More links per node
    ↓
More paths to the true neighbours
    ↓
Higher recall, but more memory and slower build
```

`M` is fixed when the index is built. Changing it requires a rebuild.

---

# 13. Role of `efSearch`

`efSearch` is the size of the candidate list during a search.

```text
Larger efSearch
    ↓
More candidates explored
    ↓
Higher recall, but slower queries
```

`efSearch` is a **search-time** setting: one built index can be searched with any value, without rebuilding. Neither LSH nor PQ can change its accuracy without a rebuild.

`efSearch` must be at least `k` (the number of neighbours returned).

---

# 14. Role of `efConstruction`

`efConstruction` is the size of the candidate list while **building** the graph.

```text
Larger efConstruction
    ↓
Better neighbours chosen at insertion
    ↓
Better graph quality, but slower build
```

It only affects build time and graph quality, not the size of the index.

---

# 15. HNSW Storage

HNSW stores the full original vectors **plus** the graph links.

For each vector:

```text
Original vector:   d × 4 bytes           (float32)
Layer-0 links:     2M × 4 bytes          (one 4-byte ID per link)
Upper-layer links: small extra amount    (few nodes live above layer 0)
```

For:

```text
d = 384, M = 32
```

this is:

```text
Vector:  384 × 4  = 1536 bytes
Links:   2 × 32 × 4 = 256 bytes  (+ upper layers ≈ 272 bytes measured)
```

Therefore:

```text
HNSW index  =  Flat index  +  graph overhead
```

HNSW is **always larger than exact search**. It trades memory for speed. The overhead depends on `M`, not on `d`.

---

# 16. FAISS Implementation

In FAISS, an HNSW index can be created using:

```python
index = faiss.IndexHNSWFlat(d, M)
```

For example:

```python
index = faiss.IndexHNSWFlat(384, 32)
index.hnsw.efConstruction = 40
```

where:

```text
d = 384
M = 32
efConstruction = 40
```

The index is populated (this also builds the graph; no separate training step is needed):

```python
index.add(xb)
```

The search-time setting is changed directly on the index:

```python
index.hnsw.efSearch = 64
```

and searched:

```python
D, I = index.search(xq, k)
```

where:

```text
D → returned distances
I → indices of the returned neighbours
```

"Flat" in `IndexHNSWFlat` means the vectors are stored uncompressed.

---

# 17. Complete End-to-End Process

The entire process can be summarized as:

```text
                 DATABASE
                    │
                    ▼
             Original vectors
                    │
                    ▼
      Assign each vector a random top layer
                    │
                    ▼
     Insert: greedy walk down, then search
     for nearest nodes (efConstruction)
                    │
                    ▼
    Link to M diverse neighbours per layer
                    │
                    ▼
      Multi-layer graph + original vectors


                  QUERY
                    │
                    ▼
             Query vector
                    │
                    ▼
     Enter at the top layer's entry point
                    │
                    ▼
     Greedy walk down the upper layers
                    │
                    ▼
     Beam search on layer 0 (efSearch)
                    │
                    ▼
       Return the k closest candidates
                    │
                    ▼
       Approximate neighbours
```

---

# 18. The Core Mathematical Idea

**Layer assignment:**

```text
l = ⌊ −ln(U) × m_L ⌋,   m_L = 1 / ln(M)
```

so the number of nodes shrinks by a factor of about `M` per layer, and the number of layers grows like:

```text
number of layers ≈ log_M(n)
```

**Search cost:**

Each layer needs only a few greedy steps, and there are about `log(n)` layers, so a search costs roughly:

```text
O(log n)   distance computations
```

compared with exact search:

```text
O(n)       distance computations
```

This is why HNSW can be many times faster than a Flat index while still returning most of the true neighbours.

---

# 19. HNSW in One Sentence

> **HNSW links every vector to a few diverse nearby vectors across a hierarchy of increasingly sparse graph layers, then finds approximate nearest neighbours by greedily walking from the sparse top layer down to the dense bottom layer, keeping a candidate list of size efSearch.**
