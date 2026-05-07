"""Тесты octree/flat_tree.py — FlatOctree (плоское октодерево)."""

from __future__ import annotations

import numpy as np

from dem.octree import FlatOctree


# --------------------------------------------------------------------------
# Построение дерева
# --------------------------------------------------------------------------
class TestBuild:
    def test_root_contains_all_particles(self) -> None:
        rng = np.random.default_rng(42)
        n = 50
        L = 1.0
        positions = rng.random((n, 3)) * L
        radii = rng.random(n) * 0.005 + 0.005

        tree = FlatOctree(theta=0.5, mpl=1, num_particles=n)
        tree.build(positions, radii, L=L, periodic=False)

        # Корень — узел 0; count[0] = N
        assert tree.nodes.count.to_numpy()[0] == n
        assert tree.num_particles[None] == n

    def test_node_count_growth_with_smaller_mpl(self) -> None:
        """Меньший mpl → больше внутренних узлов."""
        rng = np.random.default_rng(0)
        n = 80
        L = 1.0
        positions = rng.random((n, 3)) * L
        radii = np.full(n, 0.005)

        nodes_at_mpl = []
        for mpl in (8, 1):
            tree = FlatOctree(theta=0.5, mpl=mpl, num_particles=n)
            tree.build(positions, radii, L=L, periodic=False)
            nodes_at_mpl.append(int(tree.node_count[None]))
        assert nodes_at_mpl[1] > nodes_at_mpl[0]


# --------------------------------------------------------------------------
# Параметры дерева
# --------------------------------------------------------------------------
class TestUpdateParams:
    def test_update_params_changes_state(self) -> None:
        tree = FlatOctree(theta=0.5, mpl=1, num_particles=16)
        tree.update_params(theta=0.3, mpl=4)
        assert tree.theta == 0.3
        assert tree.mpl == 4
        assert tree.theta_sq[None] == 0.3 * 0.3
        assert tree.mpl_field[None] == 4


# --------------------------------------------------------------------------
# Periodic flag
# --------------------------------------------------------------------------
class TestPeriodicFlag:
    def test_periodic_flag_set(self) -> None:
        rng = np.random.default_rng(1)
        n = 10
        L = 1.0
        positions = rng.random((n, 3)) * L
        radii = np.full(n, 0.01)

        tree = FlatOctree(theta=0.5, mpl=1, num_particles=n)
        tree.build(positions, radii, L=L, periodic=False)
        assert tree.periodic[None] == 0

        tree.build(positions, radii, L=L, periodic=True)
        assert tree.periodic[None] == 1
