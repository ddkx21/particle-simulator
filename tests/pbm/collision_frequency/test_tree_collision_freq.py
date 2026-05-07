"""Тесты pbm/collision_frequency/tree_collision_freq.py.

Использует FlatOctree (Taichi). Инициализация Taichi — в conftest.py.
"""

from __future__ import annotations

import numpy as np

from dem.octree.flat_tree import FlatOctree
from pbm import VolumeGrid
from pbm.collision_frequency import TreeCollisionFrequency


class TestTreeCollisionFrequency:
    def test_compute_returns_symmetric_matrix(self) -> None:
        n = 6
        L = 1.0
        positions = np.array(
            [
                [0.10, 0.10, 0.10],
                [0.12, 0.10, 0.10],  # контакт с #0
                [0.50, 0.50, 0.50],
                [0.80, 0.10, 0.10],
                [0.10, 0.80, 0.10],
                [0.10, 0.10, 0.80],
            ],
            dtype=np.float64,
        )
        radii = np.full(n, 0.025, dtype=np.float64)

        tree = FlatOctree(theta=0.5, mpl=1, num_particles=n)
        tree.build(positions, radii, L=L, periodic=False)

        g = VolumeGrid.from_radii_range(0.01, 0.05, 4)
        cf = TreeCollisionFrequency(tree, g, max_particles=n)
        M = cf.compute(positions, radii)

        assert M.shape == (g.n_bins, g.n_bins)
        np.testing.assert_array_equal(M, M.T)
        assert M.sum() > 0

    def test_no_overlap_returns_zero_matrix(self) -> None:
        n = 4
        L = 1.0
        positions = np.array(
            [
                [0.10, 0.10, 0.10],
                [0.90, 0.10, 0.10],
                [0.10, 0.90, 0.10],
                [0.10, 0.10, 0.90],
            ],
            dtype=np.float64,
        )
        radii = np.full(n, 0.01, dtype=np.float64)

        tree = FlatOctree(theta=0.5, mpl=1, num_particles=n)
        tree.build(positions, radii, L=L, periodic=False)

        g = VolumeGrid.from_radii_range(0.005, 0.05, 4)
        cf = TreeCollisionFrequency(tree, g, max_particles=n)
        M = cf.compute(positions, radii)
        assert M.sum() == 0

    def test_periodic_boundary_records_contact(self) -> None:
        n = 2
        L = 1.0
        positions = np.array(
            [
                [0.01, 0.5, 0.5],
                [0.99, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        radii = np.full(n, 0.05, dtype=np.float64)
        g = VolumeGrid.from_radii_range(0.02, 0.1, 3)

        tree_no = FlatOctree(theta=0.5, mpl=1, num_particles=n)
        tree_no.build(positions, radii, L=L, periodic=False)
        cf_no = TreeCollisionFrequency(tree_no, g, max_particles=n)
        assert cf_no.compute(positions, radii).sum() == 0

        tree_yes = FlatOctree(theta=0.5, mpl=1, num_particles=n)
        tree_yes.build(positions, radii, L=L, periodic=True)
        cf_yes = TreeCollisionFrequency(tree_yes, g, max_particles=n)
        assert cf_yes.compute(positions, radii).sum() > 0
