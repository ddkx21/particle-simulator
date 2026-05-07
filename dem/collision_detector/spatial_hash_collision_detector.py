"""Spatial Hash Collision Detector — O(N) обнаружение столкновений.

Алгоритм:
1. Assign cells — каждая частица получает hash bucket по своей позиции
2. Prefix sum — вычисляем стартовый индекс каждого bucket'а
3. Scatter — раскладываем индексы частиц по отсортированному массиву
4. Detect collisions — для каждой частицы проверяем 27 соседних cells
"""

import numpy as np
import taichi as ti

from .collision_detector_base import CollisionDetector

_EMPTY_PAIRS = np.empty((0, 2), dtype=np.int32)


def _next_power_of_2(n: int) -> int:
    """Ближайшая степень двойки >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


@ti.data_oriented
class SpatialHashCollisionDetector(CollisionDetector):
    """Детектор столкновений на основе spatial hashing.

    Параметры:
        num_particles: максимальное число частиц
        L: размер домена (кубический)
        boundary_mode: "periodic" или "open"
    """

    def __init__(self, num_particles: int, L: float = 1.0, boundary_mode: str = "periodic"):
        self._max_particles = num_particles
        self._L = L
        self._boundary_mode = boundary_mode

        # Hash table — load factor ~50%
        self._hash_table_size = _next_power_of_2(2 * num_particles)
        self._hash_mask = self._hash_table_size - 1

        # --- Taichi fields ---
        # Позиции и радиусы
        self.ti_positions = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
        self.ti_radii = ti.field(dtype=ti.f64, shape=num_particles)

        # Spatial hash structures
        self.cell_count = ti.field(dtype=ti.i32, shape=self._hash_table_size)
        self.cell_offset = ti.field(dtype=ti.i32, shape=self._hash_table_size)
        self.cell_offset_copy = ti.field(dtype=ti.i32, shape=self._hash_table_size)
        self.sorted_indices = ti.field(dtype=ti.i32, shape=num_particles)
        self.particle_cell = ti.field(dtype=ti.i32, shape=num_particles)

        # Collision results
        self.collision_flag = ti.field(dtype=ti.i32, shape=())
        self.collision_count = ti.field(dtype=ti.i32, shape=())
        self.collided_pairs = ti.Vector.field(2, dtype=ti.i32, shape=num_particles)

        # Runtime params (scalar fields для использования в kernels)
        self.ti_num_particles = ti.field(dtype=ti.i32, shape=())
        self.ti_cell_size = ti.field(dtype=ti.f64, shape=())
        self.ti_grid_dim = ti.field(dtype=ti.i32, shape=())
        self.ti_L = ti.field(dtype=ti.f64, shape=())
        self.ti_boundary_mode = ti.field(dtype=ti.i32, shape=())  # 1 = periodic, 0 = open
        self.ti_hash_mask = ti.field(dtype=ti.i32, shape=())
        self.ti_hash_table_size = ti.field(dtype=ti.i32, shape=())

    # --- Taichi kernels ---

    @ti.func
    def _hash_cell(self, ix: ti.i32, iy: ti.i32, iz: ti.i32) -> ti.i32:
        """Spatial hash: отображение (ix, iy, iz) -> bucket index."""
        return (ix * 73856093 ^ iy * 19349663 ^ iz * 83492791) & self.ti_hash_mask[None]

    @ti.kernel
    def _assign_cells(self):
        """Для каждой частицы вычисляем cell hash."""
        n = self.ti_num_particles[None]
        cell_size = self.ti_cell_size[None]
        grid_dim = self.ti_grid_dim[None]
        is_periodic = self.ti_boundary_mode[None] == 1

        for i in range(n):
            pos = self.ti_positions[i]
            ix = ti.cast(ti.floor(pos[0] / cell_size), ti.i32)
            iy = ti.cast(ti.floor(pos[1] / cell_size), ti.i32)
            iz = ti.cast(ti.floor(pos[2] / cell_size), ti.i32)

            if is_periodic:
                ix = ix % grid_dim
                iy = iy % grid_dim
                iz = iz % grid_dim
            else:
                if ix < 0:
                    ix = 0
                if iy < 0:
                    iy = 0
                if iz < 0:
                    iz = 0
                if ix >= grid_dim:
                    ix = grid_dim - 1
                if iy >= grid_dim:
                    iy = grid_dim - 1
                if iz >= grid_dim:
                    iz = grid_dim - 1

            h = self._hash_cell(ix, iy, iz)
            self.particle_cell[i] = h
            ti.atomic_add(self.cell_count[h], 1)

    @ti.kernel
    def _prefix_sum(self):
        """Serial prefix sum -> cell_offset[]. Также копируем в cell_offset_copy."""
        size = self.ti_hash_table_size[None]
        running = 0
        ti.loop_config(serialize=True)
        for i in range(size):
            self.cell_offset[i] = running
            self.cell_offset_copy[i] = running
            running += self.cell_count[i]

    @ti.kernel
    def _scatter_particles(self):
        """Раскладываем индексы частиц по отсортированному массиву."""
        n = self.ti_num_particles[None]
        for i in range(n):
            h = self.particle_cell[i]
            slot = ti.atomic_add(self.cell_offset_copy[h], 1)
            self.sorted_indices[slot] = i

    @ti.kernel
    def _detect_collisions(self):
        """Для каждой частицы проверяем 27 соседних cells на столкновения."""
        n = self.ti_num_particles[None]
        cell_size = self.ti_cell_size[None]
        grid_dim = self.ti_grid_dim[None]
        is_periodic = self.ti_boundary_mode[None] == 1
        L_val = self.ti_L[None]
        self.ti_hash_table_size[None]

        for i in range(n):
            pos_i = self.ti_positions[i]
            r_i = self.ti_radii[i]

            ix = ti.cast(ti.floor(pos_i[0] / cell_size), ti.i32)
            iy = ti.cast(ti.floor(pos_i[1] / cell_size), ti.i32)
            iz = ti.cast(ti.floor(pos_i[2] / cell_size), ti.i32)

            if is_periodic:
                ix = ix % grid_dim
                iy = iy % grid_dim
                iz = iz % grid_dim
            else:
                if ix < 0:
                    ix = 0
                if iy < 0:
                    iy = 0
                if iz < 0:
                    iz = 0
                if ix >= grid_dim:
                    ix = grid_dim - 1
                if iy >= grid_dim:
                    iy = grid_dim - 1
                if iz >= grid_dim:
                    iz = grid_dim - 1

            # Проверяем 27 соседних cells (включая текущую)
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    for dk in ti.static(range(-1, 2)):
                        nx = ix + di
                        ny = iy + dj
                        nz = iz + dk

                        valid = True
                        if is_periodic:
                            nx = nx % grid_dim
                            ny = ny % grid_dim
                            nz = nz % grid_dim
                        else:
                            if (
                                nx < 0
                                or ny < 0
                                or nz < 0
                                or nx >= grid_dim
                                or ny >= grid_dim
                                or nz >= grid_dim
                            ):
                                valid = False

                        if valid:
                            nh = self._hash_cell(nx, ny, nz)
                            start = self.cell_offset[nh]
                            end = start + self.cell_count[nh]

                            # Clamp end
                            if end > n:
                                end = n

                            for slot in range(start, end):
                                j = self.sorted_indices[slot]
                                if i < j:
                                    pos_j = self.ti_positions[j]
                                    r_j = self.ti_radii[j]

                                    dx = pos_j[0] - pos_i[0]
                                    dy = pos_j[1] - pos_i[1]
                                    dz = pos_j[2] - pos_i[2]

                                    # Minimum image convention для periodic
                                    if is_periodic:
                                        if ti.abs(dx) > L_val * 0.5:
                                            dx -= ti.math.sign(dx) * L_val
                                        if ti.abs(dy) > L_val * 0.5:
                                            dy -= ti.math.sign(dy) * L_val
                                        if ti.abs(dz) > L_val * 0.5:
                                            dz -= ti.math.sign(dz) * L_val

                                    dist = ti.sqrt(dx * dx + dy * dy + dz * dz)
                                    r_sum = r_i + r_j

                                    if dist <= r_sum:
                                        self.collision_flag[None] = 1
                                        idx = ti.atomic_add(self.collision_count[None], 1)
                                        if idx < self.collided_pairs.shape[0]:
                                            self.collided_pairs[idx] = ti.Vector([i, j])

    # --- Public API ---

    def detect(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        *,
        L: float | None = None,
        boundary_mode: str | None = None,
    ) -> tuple[bool, np.ndarray]:
        """Обнаружить столкновения между частицами.

        Args:
            positions: (N, 3) массив позиций
            radii: (N,) массив радиусов
            L: размер домена (опционально, иначе используется значение из __init__)
            boundary_mode: "periodic" или "open" (опционально)

        Returns:
            (is_collision, collided_pairs) где collided_pairs имеет форму (K, 2), dtype=int32
        """
        n = positions.shape[0]
        use_L = L if L is not None else self._L
        use_boundary = boundary_mode if boundary_mode is not None else self._boundary_mode

        # cell_size = 2 * max(radii) — обеспечивает что столкновение возможно только в соседних cells
        max_radius = float(np.max(radii))
        cell_size = 2.0 * max_radius
        if cell_size <= 0:
            return False, _EMPTY_PAIRS.copy()

        grid_dim = max(1, int(np.ceil(use_L / cell_size)))

        # Копируем данные в Taichi fields (паддим до _max_particles для from_numpy)
        positions_c = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
        radii_c = np.ascontiguousarray(radii, dtype=np.float64)

        max_n = self._max_particles
        if n < max_n:
            pos_padded = np.zeros((max_n, 3), dtype=np.float64)
            pos_padded[:n] = positions_c[:n]
            rad_padded = np.zeros(max_n, dtype=np.float64)
            rad_padded[:n] = radii_c[:n]
        else:
            pos_padded = positions_c[:max_n]
            rad_padded = radii_c[:max_n]

        self.ti_positions.from_numpy(pos_padded)
        self.ti_radii.from_numpy(rad_padded)

        # Устанавливаем параметры
        self.ti_num_particles[None] = n
        self.ti_cell_size[None] = cell_size
        self.ti_grid_dim[None] = grid_dim
        self.ti_L[None] = use_L
        self.ti_boundary_mode[None] = 1 if use_boundary == "periodic" else 0
        self.ti_hash_mask[None] = self._hash_mask
        self.ti_hash_table_size[None] = self._hash_table_size

        # Очищаем state
        self.cell_count.fill(0)
        self.cell_offset.fill(0)
        self.cell_offset_copy.fill(0)
        self.sorted_indices.fill(0)
        self.collision_flag[None] = 0
        self.collision_count[None] = 0

        # 4 прохода
        self._assign_cells()
        self._prefix_sum()
        self._scatter_particles()
        self._detect_collisions()

        # Считываем результат
        is_collision = self.collision_flag[None] == 1
        if is_collision:
            count = min(self.collision_count[None], self.collided_pairs.shape[0])
            pairs = self.collided_pairs.to_numpy()[:count]
            # Дедупликация: hash-коллизии могут привести к обнаружению одной пары
            # из нескольких bucket'ов с одинаковым hash
            if len(pairs) > 0:
                pairs = np.unique(pairs, axis=0)
        else:
            pairs = _EMPTY_PAIRS.copy()

        return is_collision, pairs
