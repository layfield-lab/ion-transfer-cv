"""
Water Finger Coordinate (w) Calculation

Based on Kikkawa et al. (2015) - uses graph theory to determine
connectivity between ion and bulk water via hydrogen-bonded chain.

Algorithm:
1. Build graph with Cl⁻ and water molecules as vertices
2. Calculate all pairwise distances, apply ion-water offset
3. Sort edges by weight and use union-find (Kruskal approach) to find
   minimum threshold connecting ion to bulk water
4. The last edge added is the bottleneck; its weight is w

Reference:
  Kikkawa, N., Wang, L., & Morita, A. (2015).
  Microscopic Barrier Mechanism of Ion Transport through Liquid-Liquid Interface.
  J. Am. Chem. Soc., 137, 8022-8025.
"""

import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class WaterFingerCoordinate:
    """
    Compute water finger coordinate from molecular positions.

    Parameters:
        ion_water_offset (float): Distance offset for ion-water vs water-water
                                  connectivity (typically -0.4 Å)
        rOO_max (float): Maximum O-O distance for hydrogen bonds (typically 3.4 Å)
        bulk_z_cutoff (float): Z coordinate threshold for bulk water identification
    """

    def __init__(
        self,
        ion_water_offset: float = -0.4,
        rOO_max: float = 3.4,
        bulk_z_cutoff: float = 0.0
    ):
        """Initialize water finger coordinate calculator."""
        self.ion_water_offset = ion_water_offset
        self.rOO_max = rOO_max
        self.bulk_z_cutoff = bulk_z_cutoff

        # Cache for last calculation (for derivative computation)
        self._bottleneck_pair = None
        self._distances_array = None

    def calculate(
        self,
        ion_pos: np.ndarray,
        water_positions: np.ndarray,
        box: Optional[np.ndarray] = None,
        return_bottleneck: bool = False
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Calculate water finger coordinate w.

        Uses a Kruskal-like approach: sort all edges by weight, add them
        one by one until the ion (node 0) is connected to a bulk water.
        The last edge added is the bottleneck, and its weight is w.

        Parameters:
            ion_pos (np.ndarray): Position of Cl⁻ ion [x, y, z]
            water_positions (np.ndarray): Positions of water oxygen atoms (N, 3)
            box (np.ndarray): Periodic box vectors (3, 3) for PBC
            return_bottleneck (bool): If True, return indices of bottleneck pair

        Returns:
            w (float): Water finger coordinate in Ångströms
            bottleneck (Tuple[int, int]): Indices of bottleneck pair (if return_bottleneck=True)
        """

        n_waters = water_positions.shape[0]

        # Calculate all pairwise distances with PBC
        distances = self._calculate_distances_with_pbc(
            ion_pos, water_positions, box
        )  # (n_waters + 1, n_waters + 1)

        # Build edge list with weights
        # Node 0 = ion, Nodes 1..n_waters = water molecules
        edges = []

        # Ion-water edges (with offset applied to weight)
        for i in range(n_waters):
            weight = distances[0, i + 1] + self.ion_water_offset
            edges.append((weight, 0, i + 1))

        # Water-water edges
        for i in range(n_waters):
            for j in range(i + 1, n_waters):
                weight = distances[i + 1, j + 1]
                edges.append((weight, i + 1, j + 1))

        # Sort edges by weight
        edges.sort()

        # Find minimum threshold using union-find (Kruskal approach)
        w, bottleneck_pair = self._find_minimum_threshold(
            edges, water_positions
        )

        # Cache for derivative calculation
        self._bottleneck_pair = bottleneck_pair
        self._distances_array = distances
        self._water_positions = water_positions
        self._ion_pos = ion_pos
        self._box = box

        if return_bottleneck:
            return w, bottleneck_pair
        return w

    def _calculate_distances_with_pbc(
        self,
        ion_pos: np.ndarray,
        water_positions: np.ndarray,
        box: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate pairwise distances with periodic boundary conditions.

        Returns matrix of shape (n_waters + 1, n_waters + 1) where:
          - Row/Col 0: ion
          - Rows/Cols 1:n_waters: water molecules
        """

        n_waters = water_positions.shape[0]
        all_positions = np.vstack([ion_pos[np.newaxis, :], water_positions])
        n_atoms = all_positions.shape[0]

        distances = np.zeros((n_atoms, n_atoms))

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                delta = all_positions[j] - all_positions[i]

                # Apply PBC
                if box is not None:
                    for dim in range(3):
                        box_length = np.linalg.norm(box[dim])
                        delta[dim] = delta[dim] - box_length * np.round(delta[dim] / box_length)

                dist = np.linalg.norm(delta)
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _find_minimum_threshold(
        self,
        edges: List[Tuple[float, int, int]],
        water_positions: np.ndarray
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Find minimum distance threshold connecting ion to bulk water.

        Uses union-find (Kruskal-like): adds edges in order of weight
        until ion's component contains a bulk water molecule.
        The last edge added is the bottleneck.

        Parameters:
            edges: Sorted list of (weight, node_i, node_j) tuples
            water_positions: Water oxygen positions for bulk identification

        Returns:
            w (float): The minimum threshold distance (bottleneck weight)
            bottleneck_pair (Tuple[int, int]): Node indices of bottleneck edge
        """

        n_waters = water_positions.shape[0]
        n_nodes = n_waters + 1

        # Identify bulk water nodes (z < cutoff)
        bulk_water_nodes = set()
        for i in range(n_waters):
            if water_positions[i, 2] < self.bulk_z_cutoff:
                bulk_water_nodes.add(i + 1)

        if not bulk_water_nodes:
            logger.warning("No bulk water found; returning max edge weight")
            w, a, b = edges[-1]
            return w, (min(a, b), max(a, b))

        # Union-Find data structure
        parent = list(range(n_nodes))
        rank = [0] * n_nodes

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # Path compression
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Add edges in order of weight until ion connects to bulk water
        for weight, a, b in edges:
            union(a, b)

            # Check if ion (0) is connected to any bulk water
            ion_root = find(0)
            for bw in bulk_water_nodes:
                if find(bw) == ion_root:
                    return weight, (min(a, b), max(a, b))

        # Fallback: no connection found even with all edges
        logger.warning("Could not connect ion to bulk water")
        w, a, b = edges[-1]
        return w, (min(a, b), max(a, b))

    def get_bottleneck_pair(self) -> Optional[Tuple[int, int]]:
        """Get the bottleneck pair from the last calculation."""
        return self._bottleneck_pair
