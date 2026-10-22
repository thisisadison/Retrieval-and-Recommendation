# LSH — End-to-End Theory

## 1. Overview

Locality-Sensitive Hashing (LSH) is an approximate nearest-neighbour (ANN) method that maps high-dimensional vectors into compact binary hash codes.

The main idea is:

```text
Original vector
      ↓
Random projections
      ↓
Binary hash code
      ↓
Compare hash codes
      ↓
Approximate nearest neighbours
```

The key property of LSH is that vectors that are similar in the original vector space are more likely to produce similar hash codes.

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

The vector contains continuous floating-point values.

---

# 3. Generate Random Projection Vectors

LSH generates random vectors that define projection directions.

For each hash bit, one random projection vector is generated:

```text
r₁, r₂, ..., r_K
```

where each projection vector has the same dimensionality as the input:

```text
rᵢ ∈ Rᵈ
```

If:

```text
nbits = K = 256
```

then 256 random projection vectors are used.

Each projection vector defines a hyperplane through the origin.

---

# 4. Random Hyperplanes

A random projection vector defines a hyperplane that divides the vector space into two regions.

Conceptually:

```text
             Hyperplane
                 │
       Region 1  │  Region 2
                 │
```

A vector can therefore be classified as being on one side or the other of the hyperplane.

This gives us a natural way to convert a continuous-valued vector into a binary value.

---

# 5. Calculate the Dot Product

For a vector `x` and a random projection vector `rᵢ`, calculate:

```text
x · rᵢ
```

The dot product measures the alignment between the input vector and the projection direction.

The sign of the dot product determines which side of the hyperplane the vector lies on.

Conceptually:

```text
x · rᵢ >= 0  →  1

x · rᵢ < 0   →  0
```

Therefore, **one random projection produces one binary bit**.

---

# 6. Construct the Hash Code

Repeat the projection process for all `K` random projection vectors.

For example, suppose:

```text
K = 8
```

The vector might produce:

```text
Projection 1 → 1
Projection 2 → 0
Projection 3 → 1
Projection 4 → 1
Projection 5 → 0
Projection 6 → 0
Projection 7 → 1
Projection 8 → 0
```

These bits are concatenated:

```text
10110010
```

This is the vector's **binary hash code**.

Therefore:

```text
1 projection → 1 bit
K projections → K-bit hash code
```

For example:

```text
d = 384
nbits = 256
```

means:

```text
Original vector: 384 floating-point values

Hash code:       256 binary bits
```

`d` and `nbits` describe two different things.

---

# 7. Hash the Entire Database

Every database vector is passed through the **same set of random projections**.

For example:

```text
Vector 1 → 10110010...
Vector 2 → 10110110...
Vector 3 → 01100101...
Vector 4 → 10110000...
...
```

The original database:

```text
xb
```

is therefore transformed conceptually into:

```text
Binary hash codes
```

The hash codes are stored in the LSH index.

---

# 8. Hash the Query

The query vector must be processed using the **same random projection vectors**.

Suppose:

```text
Query = q
```

The same projections are applied:

```text
q · r₁
q · r₂
...
q · r_K
```

Each result is converted into one bit.

For example:

```text
Query
  ↓
10110100...
```

The query now has a binary hash code that can be compared against the database hash codes.

---

# 9. Why Similar Vectors Produce Similar Hash Codes

The important property of random hyperplane hashing is that nearby vectors are more likely to fall on the same side of a random hyperplane.

Consider two similar vectors:

```text
x
y
```

If they are close to each other, a randomly selected hyperplane is less likely to separate them.

Therefore, they are likely to produce the same bit for that projection.

For example:

```text
Vector A → 10110110
Vector B → 10110100
```

Only a small number of bits differ.

For vectors that are very different, more projection decisions may differ:

```text
Vector A → 10110110
Vector C → 01001001
```

This is the locality-sensitive property:

```text
Similar vectors
      ↓
Similar hash codes

Dissimilar vectors
      ↓
More different hash codes
```

This relationship is probabilistic rather than guaranteed.

---

# 10. Compare Binary Hash Codes

Once the query and database vectors have been converted into binary codes, their codes can be compared.

A natural distance between binary codes is **Hamming distance**.

Hamming distance counts how many bit positions differ.

For example:

```text
Query:   10110010
Vector:  10111000
         ^   ^^
```

The number of differing positions is the Hamming distance.

Therefore:

```text
Smaller Hamming distance
        ↓
More similar hash codes
```

The LSH index uses the binary representation to identify approximate nearest neighbours.

---

# 11. Why the Search Is Approximate

The hashing process compresses the information contained in the original vector.

For example:

```text
Original:

[0.82, -1.23, 0.47, ..., 2.11]
```

becomes:

```text
Hash:

101100101101...
```

The hash code does not preserve the exact values of the original vector.

Consequently, the nearest neighbours according to the original vector-space distance may not always be the nearest neighbours according to the hash representation.

Therefore:

```text
Exact nearest neighbours
        ≠
LSH approximate neighbours
```

in general.

This is why LSH is an **approximate nearest-neighbour** method.

---

# 12. Role of `nbits`

`nbits` determines how many binary decisions are made for each vector.

For example:

```text
nbits = 256
```

means:

```text
256 random projections
        ↓
256 binary decisions
        ↓
256-bit hash code
```

Increasing `nbits` gives the representation more bits with which to distinguish vectors.

Conceptually:

```text
More bits
    ↓
More information in hash representation
    ↓
Potentially better neighbour discrimination
```

However, increasing `nbits` also increases the size of each hash code.

---

# 13. LSH Storage

An original `float32` vector with `d` dimensions requires:

```text
d × 32 bits
```

For:

```text
d = 384
```

this is:

```text
384 × 32 = 12,288 bits
```

A 256-bit LSH code requires:

```text
256 bits
```

Therefore, the hash representation can be substantially smaller than the original floating-point vector.

Conceptually:

```text
Original vector
384 × float32
      ↓
12,288 bits

LSH representation
256 bits
```

This compression is one of the reasons LSH can be useful for large-scale vector search.

---

# 14. FAISS Implementation

In FAISS, an LSH index can be created using:

```python
index = faiss.IndexLSH(d, nbits)
```

For example:

```python
index = faiss.IndexLSH(384, 256)
```

where:

```text
d = 384
nbits = 256
```

The index can then be populated:

```python
index.add(xb)
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

FAISS handles the hashing and binary-code search internally.

---

# 15. Complete End-to-End Process

The entire process can be summarized as:

```text
                 DATABASE
                    │
                    ▼
             Original vectors
                    │
                    ▼
          Random projection vectors
                    │
                    ▼
             Dot products
                    │
                    ▼
             Sign of each result
                    │
                    ▼
             Binary hash codes
                    │
                    ▼
              Store in index


                  QUERY
                    │
                    ▼
             Query vector
                    │
                    ▼
       Same random projections
                    │
                    ▼
             Dot products
                    │
                    ▼
             Sign of each result
                    │
                    ▼
           Query binary code
                    │
                    ▼
        Compare binary codes
                    │
                    ▼
       Approximate neighbours
```

---

# 16. The Core Mathematical Idea

For each random projection vector `rᵢ`, define a hash function:

```text
hᵢ(x) =
    1, if x · rᵢ ≥ 0
    0, if x · rᵢ < 0
```

Using `K` hash functions:

```text
H(x) = [h₁(x), h₂(x), ..., h_K(x)]
```

`H(x)` is the final binary hash code.

For example:

```text
H(x) = [1, 0, 1, 1, 0, 0, 1, 0]
```

The nearest-neighbour search is then performed using the similarity of these binary codes rather than directly comparing all original floating-point values.

---

# 17. LSH in One Sentence

> **LSH projects each high-dimensional vector onto multiple random hyperplanes, converts the resulting signs into a binary hash code, and uses the similarity between these codes to perform approximate nearest-neighbour search.**