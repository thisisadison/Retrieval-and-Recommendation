## PQ

Product Quantization (PQ) compresses high-dimensional vectors by dividing each vector into smaller subspaces and independently quantizing each subspace.

### 1. Training

Given a vector of dimension `d`, PQ divides it into `m` subspaces:

```text
d / m = dimensions per subspace
```

For `d = 384` and `m = 32`, each subspace contains 12 dimensions.

For each subspace, k-means clustering is performed to create a codebook of `2^nbits` centroids. For example, with `nbits = 10`:

```text
2^10 = 1024 centroids per subspace
```

The centroids represent the possible values that a subspace can be approximated by.

### 2. Encoding

Each database vector is split into the same `m` subspaces. For each subspace, the vector is assigned to its nearest centroid.

For example:

```text
Original vector:
[x₁, x₂, x₃, ..., x₃₈₄]

PQ code:
[731, 12, 891, ..., 204]
```

Instead of storing the original floating-point values, PQ stores only the centroid index for each subspace. This significantly reduces the amount of memory required.

### 3. Search

When a query arrives, it is also divided into `m` subspaces. The distance between each query subspace and all centroids in its corresponding codebook is calculated and stored in a lookup table.

For example, using **L2 (Euclidean) distance**, if a query subspace is:

```text
q = [1, 2]
```

and a centroid is:

```text
c = [4, 6]
```

the squared L2 distance is:

```text
(1 - 4)² + (2 - 6)²
= 9 + 16
= 25
```

This is repeated between the query subspace and every centroid in its codebook to create the distance lookup table.

For each database vector, its PQ codes identify which centroid was selected for each subspace. The corresponding distances are retrieved from the lookup tables and summed across all subspaces:

```text
Approximate distance
= distance₁ + distance₂ + ... + distanceₘ
```

The vectors with the smallest approximate distances are then returned as the top-k nearest neighbours.

### Why PQ is approximate

PQ replaces each subvector with its nearest centroid, so the original vectors are no longer represented exactly. This introduces **quantization error**, which can change the nearest-neighbour ranking and reduce recall compared to an exact Flat index.

### Key trade-off

```text
Higher compression
      ↓
Smaller index size
      ↓
More quantization
      ↓
Potentially lower recall
```

Increasing `m` generally reduces the dimensionality of each subspace, allowing the vector to be represented more precisely and often improving recall, at the cost of a larger PQ code.
