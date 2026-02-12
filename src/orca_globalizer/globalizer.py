from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


# Orbit->graphlet grouping follows ORCA / Pržulj enumeration used by ORCA tools.
# This exact grouping is used in the alan-turing-institute/network-comparison ORCA interface.
ORBIT_TO_GRAPHLET_4: List[List[int]] = [
    [0],            # G0
    [1, 2],         # G1
    [3],            # G2
    [4, 5],         # G3
    [6, 7],         # G4
    [8],            # G5
    [9, 10, 11],    # G6
    [12, 13],       # G7
    [14],           # G8
]

ORBIT_TO_GRAPHLET_5: List[List[int]] = [
    [0],            # G0
    [1, 2],         # G1
    [3],            # G2
    [4, 5],         # G3
    [6, 7],         # G4
    [8],            # G5
    [9, 10, 11],    # G6
    [12, 13],       # G7
    [14],           # G8
    [15, 16, 17],   # G9
    [18, 19, 20, 21],  # G10
    [22, 23],       # G11
    [24, 25, 26],   # G12
    [27, 28, 29, 30],  # G13
    [31, 32, 33],   # G14
    [34],           # G15
    [35, 36, 37, 38],  # G16
    [39, 40, 41, 42],  # G17
    [43, 44],       # G18
    [45, 46, 47, 48],  # G19
    [49, 50],       # G20
    [51, 52, 53],   # G21
    [54, 55],       # G22
    [56, 57, 58],   # G23
    [59, 60, 61],   # G24
    [62, 63, 64],   # G25
    [65, 66, 67],   # G26
    [68, 69],       # G27
    [70, 71],       # G28
    [72],           # G29
]

# Nodes-per-graphlet for max size 4: [2, 3,3, 4,4,4,4,4,4]
# for max size 5: [2, 3,3, 4x6, 5x21]. 
NODES_PER_GRAPHLET_4 = np.array([2, 3, 3, 4, 4, 4, 4, 4, 4], dtype=float)
NODES_PER_GRAPHLET_5 = np.array([2, 3, 3] + [4] * 6 + [5] * 21, dtype=float)


def infer_max_graphlet_size(num_orbits: int) -> int:
    # ORCA node-orbit output is 15 (<=4 nodes) or 73 (<=5 nodes). 
    if num_orbits == 15:
        return 4
    if num_orbits == 73:
        return 5
    raise ValueError(f"Unsupported orbit vector length: {num_orbits}. Expected 15 or 73.")


@dataclass(frozen=True)
class GlobalizerConfig:
    max_graphlet_size: Optional[int] = None  # 4 or 5; if None, inferred from input
    log_transform: bool = False
    log_base: Optional[float] = None  # None -> natural log
    log_before_normalize: bool = True  # if method='frequency'
    validate_integer_counts: bool = True
    integer_tol: float = 1e-6


class GraphletGlobalizer:
    """
    Convert ORCA per-node orbit counts (N x 15 or N x 73) to global graphlet counts.

    Globalization logic:
      1) Sum node orbit vectors across nodes
      2) Sum orbit-columns into graphlet groups
      3) Divide by nodes-per-graphlet to avoid counting each graphlet once per node 
    """

    def __init__(self, config: GlobalizerConfig = GlobalizerConfig()):
        self.config = config

    def _get_maps(self, num_orbits: int) -> Tuple[List[List[int]], np.ndarray]:
        max_k = self.config.max_graphlet_size or infer_max_graphlet_size(num_orbits)

        if max_k == 4:
            if num_orbits != 15:
                raise ValueError("max_graphlet_size=4 expects 15 orbit columns.")
            return ORBIT_TO_GRAPHLET_4, NODES_PER_GRAPHLET_4

        if max_k == 5:
            if num_orbits != 73:
                raise ValueError("max_graphlet_size=5 expects 73 orbit columns.")
            return ORBIT_TO_GRAPHLET_5, NODES_PER_GRAPHLET_5

        raise ValueError("max_graphlet_size must be 4 or 5.")

    @staticmethod
    def _as_2d_array(node_vectors: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        x = np.asarray(node_vectors, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"node_vectors must be 2D (N x O). Got shape {x.shape}.")
        return x

    def transform(
        self,
        node_vectors: Union[np.ndarray, Sequence[Sequence[float]]],
        method: str = "sum",
        return_labels: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        method:
          - 'sum'       : global graphlet counts (default)
          - 'frequency' : normalized to sum to 1
        """
        X = self._as_2d_array(node_vectors)
        num_orbits = X.shape[1]
        orbit_to_graphlet, nodes_per_graphlet = self._get_maps(num_orbits)

        orbit_sums = X.sum(axis=0)  # (O,)

        # sum orbit columns into graphlets
        graphlet_node_sums = np.array(
            [orbit_sums[idxs].sum() for idxs in orbit_to_graphlet],
            dtype=float,
        )
        

        # divide to avoid multi-counting each graphlet once per node 
        global_counts = graphlet_node_sums / nodes_per_graphlet

        raw = global_counts.copy()
        if self.config.validate_integer_counts:
            rounded = np.rint(raw)
            if np.max(np.abs(raw - rounded)) <= self.config.integer_tol:
                raw = rounded

        method = method.lower().strip()

        if method in ("frequency","freq","normalized","norm"):
            if self.config.log_transform and self.config.log_before_normalize:
                x = np.log1p(raw) if self.config.log_base is None else (np.log1p(raw) / np.log(self.config.log_base))
                total = x.sum()
                out = x / total if total > 0 else x
            else:
                total = raw.sum()
                out = raw / total if total > 0 else raw
                if self.config.log_transform and not self.config.log_before_normalize:
                    out = np.log1p(out) if self.config.log_base is None else (np.log1p(out) / np.log(self.config.log_base))
        elif method in ("sum","count","counts","raw"):  # method == "sum"
            out = raw
            if self.config.log_transform:
                out = np.log1p(out) if self.config.log_base is None else (np.log1p(out) / np.log(self.config.log_base))
        else:
            raise ValueError("method must be 'sum' or 'frequency'.")
        
        if return_labels:
            labels = [f"G{i}" for i in range(len(out))]
            return out, labels
        return out


def globalize_orca(
    node_vectors: Union[np.ndarray, Sequence[Sequence[float]]],
    orca_node: Optional[int] = None,  # 4 or 5
    method: str = "sum",
    log_transform: bool = False,
    log_base: Optional[float] = None,
    log_before_normalize: bool = True,
    validate_integer_counts: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper.
    - orca_node=4 -> expects 15 orbits -> returns (9,) for G0..G8
    - orca_node=5 -> expects 73 orbits -> returns (30,) for G0..G29
    """
    cfg = GlobalizerConfig(
        max_graphlet_size=orca_node,
        log_transform=log_transform,
        log_base=log_base,
        log_before_normalize=log_before_normalize,
        validate_integer_counts=validate_integer_counts,
    )
    return GraphletGlobalizer(cfg).transform(node_vectors, method=method)