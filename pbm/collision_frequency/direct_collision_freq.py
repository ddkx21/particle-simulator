from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pbm.volume_grid import VolumeGrid


class DirectCollisionFrequency:
    """O(N^2) brute-force подсчёт коллизий между объёмными классами."""

    def __init__(self, grid: VolumeGrid) -> None:
        self.grid = grid

    def compute(
        self,
        positions: NDArray[np.float64],
        radii: NDArray[np.float64],
        L: float = 0.0,
        periodic: bool = False,
    ) -> NDArray[np.float64]:
        n = len(radii)
        n_bins = self.grid.n_bins

        volumes = (4.0 / 3.0) * np.pi * radii**3
        bins = self.grid.bin_indices(volumes)
        collision_matrix = np.zeros((n_bins, n_bins), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[j] - positions[i]
                if periodic and L > 0:
                    dx -= L * np.round(dx / L)
                dist = np.linalg.norm(dx)
                contact = radii[i] + radii[j]
                if dist <= contact:
                    b_i, b_j = bins[i], bins[j]
                    collision_matrix[b_i, b_j] += 1.0
                    collision_matrix[b_j, b_i] += 1.0

        return collision_matrix
