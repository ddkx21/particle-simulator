"""Тесты octree/tree_stats.py — извлечение статистики построенного октодерева."""

from __future__ import annotations

import numpy as np

from dem.octree import FlatOctree
from dem.octree.tree_stats import TreeStatistics, compute_tree_stats, print_tree_stats


def _build_demo_tree(n: int = 30, mpl: int = 4, seed: int = 0) -> FlatOctree:
    rng = np.random.default_rng(seed)
    L = 1.0
    positions = rng.random((n, 3)) * L
    radii = np.full(n, 0.01)
    tree = FlatOctree(theta=0.5, mpl=mpl, num_particles=n)
    tree.build(positions, radii, L=L, periodic=False)
    return tree


# --------------------------------------------------------------------------
# compute_tree_stats
# --------------------------------------------------------------------------
class TestComputeTreeStats:
    def test_returns_treestatistics_instance(self) -> None:
        tree = _build_demo_tree()
        stats = compute_tree_stats(tree)
        assert isinstance(stats, TreeStatistics)

    def test_total_particles_matches_input(self) -> None:
        n = 30
        tree = _build_demo_tree(n=n)
        stats = compute_tree_stats(tree)
        assert stats.total_particles == n

    def test_internal_plus_leaves_eq_total_nodes(self) -> None:
        tree = _build_demo_tree()
        stats = compute_tree_stats(tree)
        assert stats.internal_nodes + stats.total_leaves == stats.total_nodes

    def test_non_empty_plus_empty_eq_total_leaves(self) -> None:
        tree = _build_demo_tree()
        stats = compute_tree_stats(tree)
        assert stats.non_empty_leaves + stats.empty_leaves == stats.total_leaves

    def test_timing_fields_passed_through(self) -> None:
        tree = _build_demo_tree()
        stats = compute_tree_stats(tree, build_time_ms=12.5, compute_time_ms=4.5)
        assert stats.build_time_ms == 12.5
        assert stats.compute_time_ms == 4.5

    def test_tree_depth_at_least_one(self) -> None:
        tree = _build_demo_tree()
        stats = compute_tree_stats(tree)
        assert stats.tree_depth >= 1

    def test_nodes_per_level_sum_eq_total_nodes(self) -> None:
        tree = _build_demo_tree()
        stats = compute_tree_stats(tree)
        total = sum(cnt for _, cnt in stats.nodes_per_level)
        assert total == stats.total_nodes


# --------------------------------------------------------------------------
# print_tree_stats — smoke-тест без проверки формата
# --------------------------------------------------------------------------
class TestPrintTreeStats:
    def test_does_not_raise(self, capsys) -> None:
        tree = _build_demo_tree()
        stats = compute_tree_stats(tree, build_time_ms=10.0, compute_time_ms=2.0)
        print_tree_stats(stats)
        out = capsys.readouterr().out
        assert "Octree Statistics" in out
        assert "Tree depth" in out
