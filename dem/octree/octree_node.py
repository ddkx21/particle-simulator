"""
OctreeNode структура для плоского октодерева (flat-array octree).

Все узлы хранятся в непрерывных Taichi-полях, навигация через int-индексы.
Поддержка листьев с несколькими частицами через leaf_start + count в побочном массиве.
"""

import taichi as ti


@ti.data_oriented
class OctreeNode:
    """
    Структура узла плоского октодерева.

    Поля:
    - min_x/y/z, max_x/y/z: ограничивающий параллелепипед
    - R3_sum: Σ Rj³ в поддереве
    - R3_cx/cy/cz: R³-взвешенный центр (хранятся как суммы x*r³)
    - force_sum_x/y/z: Σ F в поддереве (для Стокслет-аппроксимации)
    - fmom_XX: сырые силовые моменты P_αβ = Σ_j r_j,α * F_j,β (9 компонент)
      Дипольный тензор D = P - r_cm ⊗ F_eff вычисляется на лету.
    - first_child: индекс первого ребёнка; -1 = лист
    - next: следующий узел для безстекового обхода; -1 = конец
    - leaf_start: начало слота в leaf_indices; -1 = внутренний/пустой
    - count: число частиц в поддереве
    """

    def __init__(self, max_nodes: int = 100000):
        self.max_nodes = max_nodes

        # Ограничивающий параллелепипед
        self.min_x = ti.field(dtype=ti.f64, shape=max_nodes)
        self.min_y = ti.field(dtype=ti.f64, shape=max_nodes)
        self.min_z = ti.field(dtype=ti.f64, shape=max_nodes)
        self.max_x = ti.field(dtype=ti.f64, shape=max_nodes)
        self.max_y = ti.field(dtype=ti.f64, shape=max_nodes)
        self.max_z = ti.field(dtype=ti.f64, shape=max_nodes)

        # R³-взвешенные агрегаты
        self.R3_sum = ti.field(dtype=ti.f64, shape=max_nodes)
        self.R3_cx = ti.field(dtype=ti.f64, shape=max_nodes)
        self.R3_cy = ti.field(dtype=ti.f64, shape=max_nodes)
        self.R3_cz = ti.field(dtype=ti.f64, shape=max_nodes)

        # Силовые агрегаты для стокслета (монополь)
        self.force_sum_x = ti.field(dtype=ti.f64, shape=max_nodes)
        self.force_sum_y = ti.field(dtype=ti.f64, shape=max_nodes)
        self.force_sum_z = ti.field(dtype=ti.f64, shape=max_nodes)

        # Сырые силовые моменты P_αβ = Σ_j r_j,α * F_j,β (дипольная коррекция)
        # D_αβ = P_αβ - r_cm,α * F_eff,β вычисляется на лету
        self.fmom_xx = ti.field(dtype=ti.f64, shape=max_nodes)
        self.fmom_xy = ti.field(dtype=ti.f64, shape=max_nodes)
        self.fmom_xz = ti.field(dtype=ti.f64, shape=max_nodes)
        self.fmom_yx = ti.field(dtype=ti.f64, shape=max_nodes)
        self.fmom_yy = ti.field(dtype=ti.f64, shape=max_nodes)
        self.fmom_yz = ti.field(dtype=ti.f64, shape=max_nodes)
        self.fmom_zx = ti.field(dtype=ti.f64, shape=max_nodes)
        self.fmom_zy = ti.field(dtype=ti.f64, shape=max_nodes)
        self.fmom_zz = ti.field(dtype=ti.f64, shape=max_nodes)

        # Максимальный радиус частицы в поддереве (для отсечения в collision frequency)
        self.max_radius = ti.field(dtype=ti.f64, shape=max_nodes)

        # Структура дерева
        self.first_child = ti.field(dtype=ti.i32, shape=max_nodes)
        self.next = ti.field(dtype=ti.i32, shape=max_nodes)
        self.leaf_start = ti.field(dtype=ti.i32, shape=max_nodes)
        self.count = ti.field(dtype=ti.i32, shape=max_nodes)

    @ti.func
    def is_leaf(self, idx: ti.i32) -> ti.i32:
        return self.first_child[idx] < 0

    @ti.func
    def size(self, idx: ti.i32) -> ti.f64:
        dx = self.max_x[idx] - self.min_x[idx]
        dy = self.max_y[idx] - self.min_y[idx]
        dz = self.max_z[idx] - self.min_z[idx]
        return ti.max(ti.max(dx, dy), dz)

    @ti.func
    def get_octant(self, idx: ti.i32, x: ti.f64, y: ti.f64, z: ti.f64) -> ti.i32:
        cx = (self.min_x[idx] + self.max_x[idx]) * 0.5
        cy = (self.min_y[idx] + self.max_y[idx]) * 0.5
        cz = (self.min_z[idx] + self.max_z[idx]) * 0.5
        octant = 0
        if x >= cx:
            octant |= 1
        if y >= cy:
            octant |= 2
        if z >= cz:
            octant |= 4
        return octant

    @ti.func
    def get_center(self, idx: ti.i32) -> ti.types.vector(3, ti.f64):
        """R³-взвешенный центр, или геометрический центр если R3_sum == 0."""
        result = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        if self.R3_sum[idx] > 0.0:
            inv = 1.0 / self.R3_sum[idx]
            result = ti.Vector(
                [self.R3_cx[idx] * inv, self.R3_cy[idx] * inv, self.R3_cz[idx] * inv]
            )
        else:
            result = ti.Vector(
                [
                    (self.min_x[idx] + self.max_x[idx]) * 0.5,
                    (self.min_y[idx] + self.max_y[idx]) * 0.5,
                    (self.min_z[idx] + self.max_z[idx]) * 0.5,
                ]
            )
        return result

    @ti.func
    def init_node(
        self,
        idx: ti.i32,
        min_x: ti.f64,
        min_y: ti.f64,
        min_z: ti.f64,
        max_x: ti.f64,
        max_y: ti.f64,
        max_z: ti.f64,
        next_idx: ti.i32,
    ):
        self.min_x[idx] = min_x
        self.min_y[idx] = min_y
        self.min_z[idx] = min_z
        self.max_x[idx] = max_x
        self.max_y[idx] = max_y
        self.max_z[idx] = max_z
        self.first_child[idx] = -1
        self.next[idx] = next_idx
        self.leaf_start[idx] = -1
        self.count[idx] = 0
        self.R3_sum[idx] = 0.0
        self.R3_cx[idx] = 0.0
        self.R3_cy[idx] = 0.0
        self.R3_cz[idx] = 0.0
        self.max_radius[idx] = 0.0
        self.force_sum_x[idx] = 0.0
        self.force_sum_y[idx] = 0.0
        self.force_sum_z[idx] = 0.0
        self.fmom_xx[idx] = 0.0
        self.fmom_xy[idx] = 0.0
        self.fmom_xz[idx] = 0.0
        self.fmom_yx[idx] = 0.0
        self.fmom_yy[idx] = 0.0
        self.fmom_yz[idx] = 0.0
        self.fmom_zx[idx] = 0.0
        self.fmom_zy[idx] = 0.0
        self.fmom_zz[idx] = 0.0
