"""
Расширенная статистика октодерева.

Предоставляет:
- TreeStatistics: frozen dataclass со всеми метриками дерева
- compute_tree_stats(): извлечение статистики из построенного FlatOctree
- print_tree_stats(): форматированный вывод в stdout
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TreeStatistics:
    """Полная статистика построенного октодерева."""

    tree_depth: int
    total_nodes: int
    max_nodes: int
    internal_nodes: int
    total_leaves: int
    non_empty_leaves: int
    empty_leaves: int
    total_particles: int
    mpl: int
    # Заполнение листьев
    min_particles_per_leaf: int
    max_particles_per_leaf: int
    avg_particles_per_leaf: float
    median_particles_per_leaf: float
    # Гистограмма заполнения: {label: count}
    leaf_fill_histogram: dict[str, int]
    # Узлы на уровень: [(level, node_count), ...]
    nodes_per_level: list[tuple[int, int]]
    # Тайминг (заполняется вызывающим кодом)
    build_time_ms: float | None
    compute_time_ms: float | None


def _compute_node_depths(octree) -> np.ndarray:
    """Реконструкция глубины каждого узла из _parent_level_offsets.

    Returns:
        np.ndarray shape (n_nodes,) dtype int32 — глубина каждого узла.
    """
    n_nodes = int(octree.node_count[None])
    depths = np.zeros(n_nodes, dtype=np.int32)
    # root = level 0, уже 0

    offsets = octree._parent_level_offsets
    parent_count = int(octree.parent_count[None])
    if not offsets or parent_count == 0:
        return depths

    parent_fc = octree.parent_first_child.to_numpy()[:parent_count]
    child_offsets = np.arange(8, dtype=np.int64)

    for lev_idx, start in enumerate(offsets):
        end = offsets[lev_idx + 1] if lev_idx + 1 < len(offsets) else parent_count
        child_depth = lev_idx + 1
        fcs = parent_fc[start:end]
        valid = fcs >= 0
        valid_fcs = fcs[valid].astype(np.int64)
        if len(valid_fcs) == 0:
            continue
        # Broadcast: (n_parents, 1) + (8,) -> (n_parents, 8) -> ravel
        children = (valid_fcs[:, np.newaxis] + child_offsets).ravel()
        children = children[children < n_nodes]
        depths[children] = child_depth

    return depths


def compute_tree_stats(
    octree,
    build_time_ms: float | None = None,
    compute_time_ms: float | None = None,
) -> TreeStatistics:
    """Извлечь полную статистику из построенного FlatOctree."""
    n_nodes = int(octree.node_count[None])
    n_particles = int(octree.num_particles[None])
    mpl = int(octree.mpl_field[None])
    max_nodes = octree.max_nodes

    first_child = octree.nodes.first_child.to_numpy()[:n_nodes]
    count = octree.nodes.count.to_numpy()[:n_nodes]

    is_leaf = first_child < 0
    is_internal = ~is_leaf
    leaf_counts = count[is_leaf]

    total_leaves = int(is_leaf.sum())
    internal_nodes = int(is_internal.sum())
    non_empty_mask = leaf_counts > 0
    non_empty_leaves = int(non_empty_mask.sum())
    empty_leaves = total_leaves - non_empty_leaves

    # Статистика заполнения (только непустые листья)
    if non_empty_leaves > 0:
        ne_counts = leaf_counts[non_empty_mask]
        min_ppl = int(ne_counts.min())
        max_ppl = int(ne_counts.max())
        avg_ppl = float(ne_counts.mean())
        median_ppl = float(np.median(ne_counts))
    else:
        min_ppl = max_ppl = 0
        avg_ppl = median_ppl = 0.0

    # Гистограмма заполнения
    histogram: dict[str, int] = {}
    if total_leaves > 0:
        max_count = int(leaf_counts.max())
        # Deduplicate and sort bin edges to handle small mpl safely
        raw_edges = [0, max(mpl // 4, 1), max(mpl // 2, 1), mpl, max_count + 1]
        bin_edges = sorted(set(raw_edges))

        hist_counts, _ = np.histogram(leaf_counts, bins=bin_edges)

        # Build labels dynamically from actual bin edges
        histogram["[0]"] = int((leaf_counts == 0).sum())
        for i, cnt_val in enumerate(hist_counts):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if lo == 0 and hi == 1:
                continue  # already counted as [0]
            elif lo == 0:
                label = f"[1, {hi})"
                histogram[label] = int(cnt_val) - histogram["[0]"]
            elif hi == max_count + 1 and lo > mpl:
                histogram[f"({mpl}, ...]"] = int(cnt_val)
            else:
                bracket_r = "]" if hi == mpl or hi == max_count + 1 else ")"
                histogram[f"[{lo}, {hi}{bracket_r}"] = int(cnt_val)

    # Глубины узлов
    depths = _compute_node_depths(octree)
    tree_depth = int(depths.max()) + 1 if n_nodes > 0 else 0

    # Узлы на уровень
    level_counts = np.bincount(depths)
    nodes_per_level = [(int(lev), int(cnt)) for lev, cnt in enumerate(level_counts) if cnt > 0]

    return TreeStatistics(
        tree_depth=tree_depth,
        total_nodes=n_nodes,
        max_nodes=max_nodes,
        internal_nodes=internal_nodes,
        total_leaves=total_leaves,
        non_empty_leaves=non_empty_leaves,
        empty_leaves=empty_leaves,
        total_particles=n_particles,
        mpl=mpl,
        min_particles_per_leaf=min_ppl,
        max_particles_per_leaf=max_ppl,
        avg_particles_per_leaf=avg_ppl,
        median_particles_per_leaf=median_ppl,
        leaf_fill_histogram=histogram,
        nodes_per_level=nodes_per_level,
        build_time_ms=build_time_ms,
        compute_time_ms=compute_time_ms,
    )


def print_tree_stats(stats: TreeStatistics) -> None:
    """Форматированный вывод статистики дерева в stdout."""
    pct = stats.total_nodes / stats.max_nodes * 100 if stats.max_nodes > 0 else 0
    print("=== Octree Statistics ===")
    print(f"  Tree depth:          {stats.tree_depth}")
    print(f"  Total nodes:         {stats.total_nodes} / {stats.max_nodes} ({pct:.1f}%)")
    print(f"    Internal:          {stats.internal_nodes}")
    print(
        f"    Leaves:            {stats.total_leaves} "
        f"(non-empty: {stats.non_empty_leaves}, empty: {stats.empty_leaves})"
    )
    print(f"  Total particles:     {stats.total_particles}")
    print(f"  mpl:                 {stats.mpl}")
    print(
        f"  Particles/leaf:      min={stats.min_particles_per_leaf}, "
        f"max={stats.max_particles_per_leaf}, "
        f"avg={stats.avg_particles_per_leaf:.1f}, "
        f"median={stats.median_particles_per_leaf:.1f}"
    )

    if stats.build_time_ms is not None:
        print(f"  Build time:          {stats.build_time_ms:.1f} ms")
    if stats.compute_time_ms is not None:
        print(f"  Compute time:        {stats.compute_time_ms:.1f} ms")

    if stats.leaf_fill_histogram:
        print("\n  Leaf fill histogram:")
        for label, cnt in stats.leaf_fill_histogram.items():
            print(f"    {label:16s} {cnt}")

    if stats.nodes_per_level:
        print("\n  Nodes per level:")
        for level, cnt in stats.nodes_per_level:
            print(f"    Level {level}:  {cnt:>8}")
    print()
