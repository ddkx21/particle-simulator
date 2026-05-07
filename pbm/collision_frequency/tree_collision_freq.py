"""Tree-ускоренный подсчёт коллизий между объёмными классами.

Использует октодерево для отсечения далёких поддеревьев,
где столкновения невозможны (min_dist > r_i + max_radius[node]).
"""

import numpy as np
import taichi as ti
from numpy.typing import NDArray

from pbm.volume_grid import VolumeGrid


@ti.data_oriented
class TreeCollisionFrequency:
    """Подсчёт коллизий через обход октодерева.

    Привязывается к конкретному FlatOctree при создании.
    """

    def __init__(self, flat_tree, grid: VolumeGrid, max_particles: int) -> None:
        self.grid = grid
        self.n_bins = grid.n_bins

        # Ссылки на поля дерева (Taichi fields, не копии)
        self.tree_nodes = flat_tree.nodes
        self.tree_leaf_indices = flat_tree.leaf_indices
        self.tree_particle_positions = flat_tree.particle_positions
        self.tree_particle_radii = flat_tree.particle_radii
        self.tree_periodic = flat_tree.periodic
        self.tree_L = flat_tree.L

        # Собственные поля
        self.particle_bin = ti.field(dtype=ti.i32, shape=max_particles)
        self.collision_matrix = ti.field(
            dtype=ti.i32,
            shape=(grid.n_bins, grid.n_bins),
        )

    def compute(
        self,
        positions: NDArray[np.float64],
        radii: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n = len(radii)
        self._assign_bins(radii, n)
        self._clear_matrix()
        self._count_collisions(n)
        return self.collision_matrix.to_numpy().astype(np.float64)

    def _assign_bins(self, radii: NDArray[np.float64], n: int) -> None:
        volumes = (4.0 / 3.0) * np.pi * radii**3
        bins = self.grid.bin_indices(volumes)
        for idx in range(n):
            self.particle_bin[idx] = int(bins[idx])

    @ti.kernel
    def _clear_matrix(self):
        for i, j in ti.ndrange(self.n_bins, self.n_bins):
            self.collision_matrix[i, j] = 0

    @ti.kernel
    def _count_collisions(self, n: ti.i32):
        for i in range(n):
            pos_i = self.tree_particle_positions[i]
            r_i = self.tree_particle_radii[i]
            bin_i = self.particle_bin[i]

            node_idx = 0
            while node_idx >= 0:
                cnt = self.tree_nodes.count[node_idx]
                if cnt == 0:
                    node_idx = self.tree_nodes.next[node_idx]
                    continue

                is_leaf = self.tree_nodes.first_child[node_idx] < 0

                if is_leaf:
                    ls = self.tree_nodes.leaf_start[node_idx]
                    for k in range(cnt):
                        j = self.tree_leaf_indices[ls + k]
                        if j >= 0 and j > i:
                            pos_j = self.tree_particle_positions[j]
                            dx = pos_j[0] - pos_i[0]
                            dy = pos_j[1] - pos_i[1]
                            dz = pos_j[2] - pos_i[2]

                            if self.tree_periodic[None] == 1:
                                L_val = self.tree_L[None]
                                if ti.abs(dx) > L_val * 0.5:
                                    dx -= ti.math.sign(dx) * L_val
                                if ti.abs(dy) > L_val * 0.5:
                                    dy -= ti.math.sign(dy) * L_val
                                if ti.abs(dz) > L_val * 0.5:
                                    dz -= ti.math.sign(dz) * L_val

                            dist_sq = dx * dx + dy * dy + dz * dz
                            contact = r_i + self.tree_particle_radii[j]

                            if dist_sq <= contact * contact:
                                bin_j = self.particle_bin[j]
                                ti.atomic_add(
                                    self.collision_matrix[bin_i, bin_j],
                                    1,
                                )
                                ti.atomic_add(
                                    self.collision_matrix[bin_j, bin_i],
                                    1,
                                )

                    node_idx = self.tree_nodes.next[node_idx]
                    continue

                # Внутренний узел: min dist to bbox vs max contact distance
                max_r_node = self.tree_nodes.max_radius[node_idx]
                max_contact = r_i + max_r_node

                min_dist_sq = self._min_dist_sq_to_bbox(
                    pos_i[0],
                    pos_i[1],
                    pos_i[2],
                    self.tree_nodes.min_x[node_idx],
                    self.tree_nodes.min_y[node_idx],
                    self.tree_nodes.min_z[node_idx],
                    self.tree_nodes.max_x[node_idx],
                    self.tree_nodes.max_y[node_idx],
                    self.tree_nodes.max_z[node_idx],
                )

                if min_dist_sq > max_contact * max_contact:
                    node_idx = self.tree_nodes.next[node_idx]
                else:
                    node_idx = self.tree_nodes.first_child[node_idx]

    @ti.func
    def _min_dist_sq_to_bbox(
        self,
        px: ti.f64,
        py: ti.f64,
        pz: ti.f64,
        bmin_x: ti.f64,
        bmin_y: ti.f64,
        bmin_z: ti.f64,
        bmax_x: ti.f64,
        bmax_y: ti.f64,
        bmax_z: ti.f64,
    ) -> ti.f64:
        dx = ti.f64(0.0)
        if px < bmin_x:
            dx = bmin_x - px
        elif px > bmax_x:
            dx = px - bmax_x

        dy = ti.f64(0.0)
        if py < bmin_y:
            dy = bmin_y - py
        elif py > bmax_y:
            dy = py - bmax_y

        dz = ti.f64(0.0)
        if pz < bmin_z:
            dz = bmin_z - pz
        elif pz > bmax_z:
            dz = pz - bmax_z

        L_val = self.tree_L[None]
        if self.tree_periodic[None] == 1 and L_val > 0.0:
            if dx > L_val * 0.5:
                dx = L_val - dx
            if dy > L_val * 0.5:
                dy = L_val - dy
            if dz > L_val * 0.5:
                dz = L_val - dz

        return dx * dx + dy * dy + dz * dz
