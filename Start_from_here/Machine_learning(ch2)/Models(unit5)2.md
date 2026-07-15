# Unsupervised Learning Models:
1.K-means Clustering:
 - <img width="607" height="135" alt="image" src="https://github.com/user-attachments/assets/b2398305-5545-46b7-a630-deb2779e010d" />
```py
this type take numbers of k(clusters) that you want + n objects from dataset
 =>  output k clusters  with similarity between objects in the same cluster
 => and low similarity between objects in different clusters.
```
2.Hierarchical Clustering;
  - divides dataset at different layers
  - form tree like clustering structure.
  - types of constructing that type:
    - bottom up aggregation policy.
    - top down splitting policy.
   
  <img width="736" height="276" alt="image" src="https://github.com/user-attachments/assets/ad40ff27-9dd5-4c9c-b825-000ae2edbab0" />
  <img width="1024" height="768" alt="image" src="https://github.com/user-attachments/assets/87a7dee1-0651-4418-8318-eeef82467455" />

```
How this  model works?

## Simple example

Suppose we have 5 students and their study hours:

```text
Ahmed = 1 hour
Ali   = 2 hours
Sara  = 8 hours
Omar  = 9 hours
Mona  = 15 hours
```

Put them on a number line:

```text
Ahmed   Ali                 Sara   Omar                    Mona
  1      2                    8      9                       15
  ●------●--------------------●------●-----------------------●
```

Our goal is to **group similar students**.

---

## 1. Bottom-up: Agglomerative clustering

This is the most common hierarchical clustering method.

Initially, **every point is its own cluster**:

```text
{Ahmed} {Ali} {Sara} {Omar} {Mona}
```

### Step 1: Find the closest points

Ahmed = 1
Ali = 2

Distance:

```text
|1 - 2| = 1
```

They are very close, so merge them:

```text
{Ahmed, Ali}   {Sara}   {Omar}   {Mona}
```

Tree:

```text
Ahmed ──┐
        ├── Cluster 1
Ali   ──┘
```

### Step 2: Find the next closest points

Sara = 8
Omar = 9

```text
|8 - 9| = 1
```

Merge them:

```text
{Ahmed, Ali}   {Sara, Omar}   {Mona}
```

Tree:

```text
Ahmed ──┐
        ├─────────┐
Ali   ──┘         │
                  │

Sara  ──┐         │
        ├─────────┤
Omar  ──┘         │
                  │
Mona  ────────────┘
```

### Step 3: Continue merging

The algorithm calculates the distance between the remaining **clusters**.

Eventually:

```text
{Ahmed, Ali}
      ↓
{Sara, Omar}
      ↓
{Mona}
```

Everything eventually becomes **one giant cluster**.

That is why it creates a **tree structure**.

---

## The dendrogram

The tree is called a **dendrogram**.

```text
Distance
   ↑

15 |                         ┌──────── Mona
   |              ┌──────────┤
 8 |              │          │
   |     ┌────────┤          │
   |     │        │          │
 1 |  ┌──┴──┐  ┌──┴──┐      │
   | Ahmed Ali Sara Omar     Mona
   +--------------------------------→
```

The **height of the connection means distance**.

Ahmed and Ali connect very low:

```text
distance = 1
```

Sara and Omar also connect low:

```text
distance = 1
```

Mona connects much higher because she is far from the others.

---

## How do we choose the number of clusters?

This is the interesting part.

Imagine cutting the tree horizontally:

```text
             ┌──────────────
             │
      ┌──────┤
      │      │
 ┌────┴───┐  │
 │        │  │
Ahmed Ali Sara Omar Mona
------------------------- ← CUT HERE
```

If we cut at a certain height, we might get:

```text
Cluster 1 → Ahmed, Ali
Cluster 2 → Sara, Omar
Cluster 3 → Mona
```

So:

```text
K = 3 clusters
```

Cut higher:

```text
Cluster 1 → Ahmed, Ali, Sara, Omar
Cluster 2 → Mona
```

Now:

```text
K = 2
```

**The dendrogram helps us decide where to cut the tree.**

---

# Top-down: Divisive clustering

This is the opposite.

Start with **everyone in one cluster**:

```text
{Ahmed, Ali, Sara, Omar, Mona}
```

Then split:

```text
             Everyone
                |
        ┌───────┴───────┐
        │               │
 Ahmed, Ali       Sara, Omar, Mona
                         |
                   ┌─────┴─────┐
                   │           │
               Sara, Omar     Mona
```

So:

```text
Top Down
   ↓
Split
   ↓
Split
   ↓
Split
```

While agglomerative is:

```text
Bottom Up
   ↑
Merge
   ↑
Merge
   ↑
Merge
```

## The core idea

Think about your **Linux file system**:

```text
/
├── home
│   ├── Tom
│   │   ├── projects
│   │   └── documents
│
├── etc
│   ├── nginx
│   └── ssh
│
└── var
    ├── log
    └── cache
```

This is **hierarchical structure**.

You can think:

```text
files
  ↓
similar files grouped
  ↓
similar groups grouped
  ↓
larger groups
```


```
 
