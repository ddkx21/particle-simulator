"""
Интерактивная 3D-визуализация октодерева через PyVista.

Предоставляет:
- visualize_tree(): wireframe-боксы узлов + частицы, раскраска по глубине или заполненности,
  интерактивный slider для фильтрации уровней.

PyVista — необязательная зависимость; при отсутствии функция выбрасывает ImportError.
"""

from __future__ import annotations

import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from .tree_stats import _compute_node_depths


def _build_wireframe_boxes(
    min_x: np.ndarray,
    min_y: np.ndarray,
    min_z: np.ndarray,
    max_x: np.ndarray,
    max_y: np.ndarray,
    max_z: np.ndarray,
) -> "pv.PolyData":
    """Vectorized построение wireframe-боксов для N узлов.

    Returns:
        pv.PolyData с линиями (12 рёбер на бокс).
    """
    n = len(min_x)
    if n == 0:
        return pv.PolyData()

    # 8 вершин на бокс: (x_bit, y_bit, z_bit) из {0,1}^3
    # Порядок: 000, 001, 010, 011, 100, 101, 110, 111
    corner_bits = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=np.float64)  # (8, 3)

    lo = np.column_stack([min_x, min_y, min_z])  # (n, 3)
    hi = np.column_stack([max_x, max_y, max_z])  # (n, 3)
    extent = hi - lo  # (n, 3)

    # Для каждого бокса: pts[box*8+k] = lo[box] + corner_bits[k] * extent[box]
    lo_rep = np.repeat(lo, 8, axis=0)       # (n*8, 3)
    ext_rep = np.repeat(extent, 8, axis=0)  # (n*8, 3)
    bits_tiled = np.tile(corner_bits, (n, 1))  # (n*8, 3)
    pts = lo_rep + bits_tiled * ext_rep

    # 12 рёбер на бокс
    edges_local = np.array([
        [0, 1], [2, 3], [4, 5], [6, 7],  # X-рёбра
        [0, 2], [1, 3], [4, 6], [5, 7],  # Y-рёбра
        [0, 4], [1, 5], [2, 6], [3, 7],  # Z-рёбра
    ], dtype=np.int64)  # (12, 2)

    # Offsets для каждого бокса
    box_offsets = np.arange(n, dtype=np.int64) * 8  # (n,)
    # Expand: (n, 12, 2) — глобальные индексы вершин
    global_edges = edges_local[np.newaxis, :, :] + box_offsets[:, np.newaxis, np.newaxis]
    # (n*12, 2)
    global_edges = global_edges.reshape(-1, 2)

    # VTK lines format: [2, p1, p2, 2, p1, p2, ...]
    n_edges = global_edges.shape[0]
    lines = np.empty((n_edges, 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = global_edges[:, 0]
    lines[:, 2] = global_edges[:, 1]
    lines = lines.ravel()

    mesh = pv.PolyData(pts, lines=lines)
    return mesh


def visualize_tree(
    octree,
    positions: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    color_by: str = "depth",
    max_level: int | None = None,
    show_particles: bool = True,
    opacity: float = 0.3,
) -> None:
    """Интерактивная 3D-визуализация октодерева.

    Args:
        octree: построенный FlatOctree
        positions: (N, 3) позиции частиц; если None — читаем из octree
        radii: (N,) радиусы; если None — читаем из octree
        color_by: 'depth' (по глубине) или 'fill' (count/mpl)
        max_level: показать уровни 0..max_level (None = все)
        show_particles: рисовать частицы
        opacity: прозрачность wireframe-линий
    """
    if not HAS_PYVISTA:
        raise ImportError(
            "PyVista не установлен. Установите: pip install pyvista"
        )

    if color_by not in ("depth", "fill"):
        raise ValueError(f"color_by must be 'depth' or 'fill', got {color_by!r}")

    n_nodes = int(octree.node_count[None])
    n_particles = int(octree.num_particles[None])
    mpl = int(octree.mpl_field[None])

    # Извлекаем данные из Taichi fields
    cnt = octree.nodes.count.to_numpy()[:n_nodes]
    nx0 = octree.nodes.min_x.to_numpy()[:n_nodes]
    ny0 = octree.nodes.min_y.to_numpy()[:n_nodes]
    nz0 = octree.nodes.min_z.to_numpy()[:n_nodes]
    nx1 = octree.nodes.max_x.to_numpy()[:n_nodes]
    ny1 = octree.nodes.max_y.to_numpy()[:n_nodes]
    nz1 = octree.nodes.max_z.to_numpy()[:n_nodes]

    depths = _compute_node_depths(octree)

    # Фильтр: только непустые узлы
    mask = cnt > 0
    if max_level is not None:
        mask &= depths <= max_level

    tree_depth = int(depths.max()) + 1 if n_nodes > 0 else 0

    # Позиции и радиусы частиц
    if positions is None:
        pos_field = octree.particle_positions.to_numpy()[:n_particles]
    else:
        pos_field = positions[:n_particles]
    if radii is None:
        rad_field = octree.particle_radii.to_numpy()[:n_particles]
    else:
        rad_field = radii[:n_particles]

    # --- Создаём plotter ---
    plotter = pv.Plotter()
    plotter.set_background("white")

    # --- Wireframe по уровням (для slider) ---
    level_actors = {}
    for lev in range(tree_depth):
        lev_mask = mask & (depths == lev)
        if not np.any(lev_mask):
            continue

        mesh = _build_wireframe_boxes(
            nx0[lev_mask], ny0[lev_mask], nz0[lev_mask],
            nx1[lev_mask], ny1[lev_mask], nz1[lev_mask],
        )

        # Скалярные данные для раскраски (12 рёбер на бокс → repeat 12)
        n_boxes = int(lev_mask.sum())
        if color_by == "fill":
            scalar_per_box = cnt[lev_mask].astype(np.float64) / max(mpl, 1)
        else:
            scalar_per_box = depths[lev_mask].astype(np.float64)

        scalar_per_edge = np.repeat(scalar_per_box, 12)
        mesh.cell_data["scalar"] = scalar_per_edge

        clim = [0.0, float(tree_depth - 1)] if color_by == "depth" else [0.0, 1.0]
        cmap = "viridis" if color_by == "depth" else "plasma"

        actor = plotter.add_mesh(
            mesh,
            scalars="scalar",
            cmap=cmap,
            clim=clim,
            line_width=1,
            opacity=opacity,
            show_scalar_bar=(lev == 0),
            label=f"Level {lev}",
        )
        level_actors[lev] = actor

    # --- Частицы ---
    if show_particles and n_particles > 0:
        cloud = pv.PolyData(pos_field)
        if n_particles > 10000:
            plotter.add_mesh(
                cloud,
                color="red",
                point_size=2,
                render_points_as_spheres=True,
                opacity=0.6,
                label="Particles",
            )
        else:
            cloud["radius"] = rad_field
            spheres = cloud.glyph(scale="radius", geom=pv.Sphere(theta_resolution=8, phi_resolution=8))
            plotter.add_mesh(
                spheres,
                color="red",
                opacity=0.6,
                label="Particles",
            )

    # --- Slider для уровней ---
    if level_actors:
        max_lev = max(level_actors.keys())

        def slider_callback(value):
            cutoff = int(round(value))
            for lev, actor in level_actors.items():
                actor.visibility = lev <= cutoff

        plotter.add_slider_widget(
            slider_callback,
            rng=[0, max_lev],
            value=max_lev,
            title="Max Level",
            pointa=(0.025, 0.1),
            pointb=(0.31, 0.1),
            style="modern",
        )

    plotter.add_legend()
    plotter.show()
