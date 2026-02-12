# orca-globalizer

Convert ORCA per-node orbit vectors into global graphlet count vectors.

## Install (editable)
pip install -e .

## Usage
import numpy as np
from orca_globalizer import globalize_orca

node_vectors = np.loadtxt("orca_output.txt")  # N x 15 (orca_node=4) or N x 73 (orca_node=5)
g = globalize_orca(node_vectors, orca_node=5, method="sum", log_transform=False)
print(g.shape)  # (30,)


## Output modes: `method="sum"` vs `method="frequency"`

ORCA produces **per-node orbit counts**: a matrix of shape **(N nodes × O orbits)** where each entry tells you how many times that node participates in that orbit.

This package converts those node-level orbit counts into a **global graphlet vector**.

### `method="sum"` (global graphlet counts)
Returns **global counts** of each graphlet type (e.g., how many triangles, how many 4-cycles, etc.).

How it works:
1. Sum orbit counts across all nodes (total “node-incidences” per orbit).
2. Group orbit columns that belong to the same graphlet and sum them.
3. Divide by the **number of nodes in that graphlet** (2 / 3 / 4 / 5) to undo multi-counting.

Why the division?
Each graphlet instance touches `k` nodes, so when you sum over nodes it contributes `k` times.  
Example: 10 triangles → 30 node-incidences (3 per triangle) → 30 / 3 = 10 triangles.

### `method="frequency"` (normalized distribution)
Returns a **normalized** version of the global counts so the vector sums to 1:

`freq[i] = count[i] / sum(count)`

Interpretation:
- `freq[i]` is the fraction of all counted graphlets (in the output vector) that are of type `Gi`.
- Useful for comparing graphs of different sizes/densities.

### Log transform (optional)
If `log_transform=True`, the package applies a `log1p` transform (i.e., `log(1 + x)`) to reduce heavy-tailed dominance.

For `method="frequency"`:
- If `log_before_normalize=True` (default): **log then normalize**
- If `log_before_normalize=False`: **normalize then log**