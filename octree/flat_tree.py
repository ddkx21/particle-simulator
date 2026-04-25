"""
Плоское октодерево (Flat Octree) для Taichi.

Стратегия: послойное построение (level-by-level) — максимальный параллелизм,
без локов и гонок данных.

Каждый уровень обрабатывается тремя фазами:
  1. assign_octants  — параллельно определяем октант каждой частицы
  2. subdivide       — serial, но только по активным узлам текущего уровня
  3. scatter         — параллельно раскладываем частицы по дочерним листам

Обход дерева (force/stokeslet) — параллельный Taichi kernel (stackless walk).

Семантика nodes.count:
  - Для листа: число частиц, хранящихся в leaf_indices[leaf_start .. leaf_start+count).
    Всегда <= mpl (кроме случая достижения max_depth).
  - Для внутреннего узла: число частиц во всём поддереве.
    Используется для early-skip (cnt==0) в force/stokeslet kernels.
"""

import taichi as ti
import numpy as np
from .octree_node import OctreeNode

# Минимальный порог R³ суммы для пропуска пустых/ничтожных узлов в force kernel.
# cnt==0 уже ловит пустые узлы; этот порог — дополнительная защита от
# числовых артефактов при экстремально малых радиусах.
_R3_SKIP_THRESHOLD = 1e-30

# Аналогичный порог для суммарной силы в stokeslet kernel.
_FORCE_SQ_SKIP_THRESHOLD = 1e-60


@ti.data_oriented
class FlatOctree:
    """
    Плоское октодерево с послойным построением.

    Параметры:
    - theta: параметр Барнса-Хатта
    - mpl: max particles per leaf — контролирует глубину дерева
    - num_particles: начальная ёмкость
    """

    def __init__(self, theta: float = 0.5, mpl: int = 1,
                 num_particles: int = 10000,
                 correction_grid_resolution: int = 0):
        self.theta = theta
        self.mpl = mpl

        self.theta_sq = ti.field(dtype=ti.f64, shape=())
        self.theta_sq[None] = theta * theta

        # mpl как поле Taichi — можно менять во время выполнения через update_params()
        self.mpl_field = ti.field(dtype=ti.i32, shape=())
        self.mpl_field[None] = mpl

        n = max(num_particles, 64)
        # Умное выделение: реальные деревья используют ~N/mpl узлов, а не худший случай 8*N.
        # Множитель 4 даёт комфортный запас; минимум 1024 для малых N.
        max_nodes = max(4 * n // max(mpl, 1), 1024)
        self.max_nodes = max_nodes

        # Исправление #1: слоты листьев = N (каждая частица ровно в одном листе)
        max_leaf_slots = n
        self.max_leaf_slots = max_leaf_slots

        # Узлы дерева
        self.nodes = OctreeNode(max_nodes)

        # Побочный массив индексов листьев — ровно N слотов
        self.leaf_indices = ti.field(dtype=ti.i32, shape=max_leaf_slots)

        # Отслеживание родителей для распространения снизу вверх
        # parents[] заполняется послойно,
        # поэтому обратный обход в _sweep_bottom_up корректен.
        self.parents = ti.field(dtype=ti.i32, shape=max_nodes)
        self.parent_first_child = ti.field(dtype=ti.i32, shape=max_nodes)
        self.parent_count = ti.field(dtype=ti.i32, shape=())
        self.node_count = ti.field(dtype=ti.i32, shape=())
        self.leaf_slot_count = ti.field(dtype=ti.i32, shape=())

        # Данные частиц
        self.particle_positions = ti.Vector.field(3, dtype=ti.f64, shape=n)
        self.particle_radii = ti.field(dtype=ti.f64, shape=n)
        self.num_particles = ti.field(dtype=ti.i32, shape=())

        # === Поля для послойного построения ===
        self.part_node = ti.field(dtype=ti.i32, shape=n)
        self.part_oct = ti.field(dtype=ti.i32, shape=n)
        self.node_child_cnt = ti.field(dtype=ti.i32, shape=(max_nodes, 8))
        self.overflow_flag = ti.field(dtype=ti.i32, shape=())

        # Исправление #4: диапазон узлов текущего уровня — обнуляем только их
        self.level_start = ti.field(dtype=ti.i32, shape=())
        self.level_end = ti.field(dtype=ti.i32, shape=())

        # Смещения уровней родителей для параллельного распространения снизу вверх.
        # parent_level_offsets[k] = начальный индекс в parents[] для уровня k.
        # Хранится как Python-список (макс. 40 уровней, пренебрежимо малые накладные расходы).
        self._parent_level_offsets: list = []

        # Область моделирования
        self.L = ti.field(dtype=ti.f64, shape=())
        self.periodic = ti.field(dtype=ti.i32, shape=())


        # Предвыделенные выходные массивы (переиспользуются каждый шаг для снижения нагрузки на аллокатор)
        self._out_fx = np.zeros(n, dtype=np.float64)
        self._out_fy = np.zeros(n, dtype=np.float64)
        self._out_fz = np.zeros(n, dtype=np.float64)
        self._out_vx = np.zeros(n, dtype=np.float64)
        self._out_vy = np.zeros(n, dtype=np.float64)
        self._out_vz = np.zeros(n, dtype=np.float64)

        # Поле сил частиц — ядро расчёта сил записывает сюда,
        # _init_leaf_forces_from_fields читает отсюда напрямую (без лишнего копирования)
        self.particle_forces = ti.Vector.field(3, dtype=ti.f64, shape=n)

        # Периодическая поправка из COMSOL (Taichi JIT компилирует все ветки — поля нужны всегда)
        self.correction_grid_resolution = correction_grid_resolution
        N_corr = max(correction_grid_resolution, 2)
        self.corr_grid_u = ti.field(dtype=ti.f64, shape=(N_corr, N_corr, N_corr))
        self.corr_grid_v = ti.field(dtype=ti.f64, shape=(N_corr, N_corr, N_corr))
        self.corr_grid_w = ti.field(dtype=ti.f64, shape=(N_corr, N_corr, N_corr))
        self.corr_grid_min = ti.field(dtype=ti.f64, shape=())
        self.corr_grid_inv_dx = ti.field(dtype=ti.f64, shape=())
        self.corr_grid_n = ti.field(dtype=ti.i32, shape=())
        self.corr_Fz_inv = ti.field(dtype=ti.f64, shape=())
        self.corr_L_ratio = ti.field(dtype=ti.f64, shape=())
        self.corr_eta_ratio = ti.field(dtype=ti.f64, shape=())
        self.corr_enabled = ti.field(dtype=ti.i32, shape=())
        self.corr_enabled[None] = 0

    def load_periodic_correction(self, correction, L_sim: float, eta_sim: float = None):
        """Загрузка данных периодической поправки COMSOL в Taichi-поля."""
        if self.correction_grid_resolution == 0:
            raise RuntimeError(
                "correction_grid_resolution=0: Taichi-поля для поправки не выделены."
            )
        grid_data = correction.get_grid_data()
        self.corr_grid_u.from_numpy(grid_data['grid_u'])
        self.corr_grid_v.from_numpy(grid_data['grid_v'])
        self.corr_grid_w.from_numpy(grid_data['grid_w'])
        self.corr_grid_min[None] = grid_data['grid_min']
        self.corr_grid_inv_dx[None] = 1.0 / grid_data['grid_dx']
        self.corr_grid_n[None] = grid_data['grid_resolution']
        self.corr_Fz_inv[None] = 1.0 / grid_data['Fz_comsol']
        self.corr_L_ratio[None] = grid_data['L_comsol'] / L_sim
        eta_comsol = grid_data['eta_comsol']
        self.corr_eta_ratio[None] = eta_comsol / eta_sim if eta_sim is not None else 1.0
        self.corr_enabled[None] = 1

    def update_params(self, theta: float, mpl: int) -> None:
        """Обновление theta и mpl без перевыделения памяти."""
        self.theta = theta
        self.mpl = mpl
        self.theta_sq[None] = theta * theta
        self.mpl_field[None] = mpl

    # =================================================================
    # Построение — послойное
    # =================================================================

    def build(self, positions: np.ndarray, radii: np.ndarray,
              L: float, periodic: bool = False):
        """Построить октодерево послойно."""
        n = positions.shape[0]

        if n > self.particle_positions.shape[0]:
            raise ValueError(
                f"Number of particles ({n}) exceeds allocated capacity "
                f"({self.particle_positions.shape[0]}). "
                f"Recreate FlatOctree with larger num_particles."
            )

        # Копируем данные частиц
        self._update_particles(positions, radii, n)

        # Сброс счётчиков
        self.node_count[None] = 1
        self.parent_count[None] = 0
        self.leaf_slot_count[None] = 0
        self.L[None] = L
        self.periodic[None] = 1 if periodic else 0

        # Исправление #4: инициализация диапазона уровней — только корень
        self.level_start[None] = 0
        self.level_end[None] = 1

        # Инициализация корня
        self._init_root_kernel(L, n)

        # Послойное построение
        MAX_LEVELS = 40
        self._parent_level_offsets = []
        for level in range(MAX_LEVELS):
            self.overflow_flag[None] = 0
            parent_count_before = self.parent_count[None]

            # Фаза 1: назначение октантов (параллельно, обнуляет только текущий уровень)
            self._level_assign_octants()

            # Фаза 2: подразделение переполненных листьев, выделение дочерних узлов (последовательно)
            self._level_subdivide_and_alloc()

            # Фаза 3: распределение частиц по дочерним узлам (параллельно)
            self._level_scatter()

            # Исправление #4: сдвигаем окно уровня на новые дочерние узлы
            new_start = self.level_end[None]
            new_end = self.node_count[None]
            self.level_start[None] = new_start
            self.level_end[None] = new_end

            # Записываем смещение уровня родителей, только если добавлены новые родители
            parent_count_after = self.parent_count[None]
            if parent_count_after > parent_count_before:
                self._parent_level_offsets.append(parent_count_before)

            if self.overflow_flag[None] == 0:
                break

        # Исправление #2: разделение финализации на два отдельных ядра
        self._finalize_leaves_alloc()
        self._finalize_leaves_scatter()

        # Построение указателей next для безстекового обхода (удаление пустых)
        self._build_next_pointers()

        # Распространение R³-агрегатов снизу вверх
        self._propagate_r3()

    @ti.kernel
    def _init_root_kernel(self, L: ti.f64, n: ti.i32):
        self.nodes.min_x[0] = 0.0
        self.nodes.min_y[0] = 0.0
        self.nodes.min_z[0] = 0.0
        self.nodes.max_x[0] = L
        self.nodes.max_y[0] = L
        self.nodes.max_z[0] = L
        self.nodes.first_child[0] = -1
        self.nodes.next[0] = -1
        self.nodes.leaf_start[0] = -1
        self.nodes.count[0] = n
        self.nodes.R3_sum[0] = 0.0
        self.nodes.R3_cx[0] = 0.0
        self.nodes.R3_cy[0] = 0.0
        self.nodes.R3_cz[0] = 0.0
        self.nodes.force_sum_x[0] = 0.0
        self.nodes.force_sum_y[0] = 0.0
        self.nodes.force_sum_z[0] = 0.0

        # Все частицы начинают в корне
        for i in range(n):
            self.part_node[i] = 0

    @ti.kernel
    def _update_particles(self, positions: ti.types.ndarray(),
                          radii: ti.types.ndarray(), n: ti.i32):
        self.num_particles[None] = n
        for i in range(n):
            self.particle_positions[i] = ti.Vector([
                positions[i, 0], positions[i, 1], positions[i, 2]
            ])
            self.particle_radii[i] = radii[i]

    @ti.kernel
    def _level_assign_octants(self):
        """Фаза 1: Для каждой частицы определяем октант в текущем узле.

        Исправление #4: обнуляем node_child_cnt только для узлов в [level_start, level_end),
        а не для всего дерева.
        """
        lstart = self.level_start[None]
        lend = self.level_end[None]

        # Обнуляем счётчики дочерних узлов ТОЛЬКО для узлов текущего уровня
        for node in range(lstart, lend):
            for oct in range(8):
                self.node_child_cnt[node, oct] = 0

        # Назначаем октанты (параллельно по частицам)
        n = self.num_particles[None]
        for i in range(n):
            node = self.part_node[i]
            if node >= lstart and node < lend:
                p = self.particle_positions[i]
                oct = self.nodes.get_octant(node, p[0], p[1], p[2])
                self.part_oct[i] = oct
                ti.atomic_add(self.node_child_cnt[node, oct], 1)

    @ti.kernel
    def _level_subdivide_and_alloc(self):
        """Фаза 2: Подразделение листьев, превысивших mpl, выделение дочерних узлов.

        Последовательное ядро — работает только с узлами текущего уровня.
        Примечание: self.node_count[None] растёт внутри цикла, но Taichi вычисляет
        верхнюю границу диапазона ОДИН раз перед входом в цикл, поэтому новые узлы не обходятся.
        """
        mpl = self.mpl_field[None]
        lstart = self.level_start[None]
        lend = self.level_end[None]

        ti.loop_config(serialize=True)
        for node in range(lstart, lend):
            # Обрабатываем только листья с избыточным числом частиц
            if (self.nodes.first_child[node] < 0 and
                    self.nodes.count[node] > mpl):
                # Защита от переполнения: проверяем наличие места для 8 новых узлов
                base = self.node_count[None]
                if base + 8 > self.nodes.max_nodes:
                    continue  # пропускаем подразделение — дерево заполнено
                # Подразделение: выделяем 8 дочерних узлов
                self.node_count[None] = base + 8
                self.nodes.first_child[node] = base

                # Записываем родителя (гарантирован послойный порядок)
                p_idx = self.parent_count[None]
                self.parents[p_idx] = node
                self.parent_first_child[p_idx] = base
                self.parent_count[None] = p_idx + 1

                cx = (self.nodes.min_x[node] + self.nodes.max_x[node]) * 0.5
                cy = (self.nodes.min_y[node] + self.nodes.max_y[node]) * 0.5
                cz = (self.nodes.min_z[node] + self.nodes.max_z[node]) * 0.5

                for oct in range(8):
                    child = base + oct
                    mn_x = cx if (oct & 1) else self.nodes.min_x[node]
                    mn_y = cy if (oct & 2) else self.nodes.min_y[node]
                    mn_z = cz if (oct & 4) else self.nodes.min_z[node]
                    mx_x = self.nodes.max_x[node] if (oct & 1) else cx
                    mx_y = self.nodes.max_y[node] if (oct & 2) else cy
                    mx_z = self.nodes.max_z[node] if (oct & 4) else cz
                    # указатель next будет установлен в _build_next_pointers
                    self.nodes.init_node(child, mn_x, mn_y, mn_z,
                                         mx_x, mx_y, mx_z, -1)
                    self.nodes.count[child] = self.node_child_cnt[node, oct]

                self.overflow_flag[None] = 1

    @ti.kernel
    def _level_scatter(self):
        """Фаза 3: Перемещение частиц из подразделённых родителей в дочерние узлы."""
        n = self.num_particles[None]
        for i in range(n):
            node = self.part_node[i]
            if node >= 0 and self.nodes.first_child[node] >= 0:
                # Родитель был подразделён — перемещаем в дочерний узел
                oct = self.part_oct[i]
                child = self.nodes.first_child[node] + oct
                self.part_node[i] = child

    # --- Исправление #2: _finalize_leaves разделён на два ядра ---

    @ti.kernel
    def _finalize_leaves_alloc(self):
        """Выделение слотов листьев с ТОЧНЫМИ счётчиками (не leaf_cap).

        Исправление #1: каждый лист получает ровно count слотов, а не фиксированный leaf_cap.
        Суммарный leaf_slot_count = N (каждая частица ровно в одном листе).
        """
        ti.loop_config(serialize=True)
        for node in range(self.node_count[None]):
            if self.nodes.first_child[node] < 0 and self.nodes.count[node] > 0:
                cnt = self.nodes.count[node]
                start = self.leaf_slot_count[None]
                self.leaf_slot_count[None] = start + cnt  # exact, not leaf_cap
                self.nodes.leaf_start[node] = start
                # Сбрасываем count для атомарного заполнения на фазе раскладки
                self.nodes.count[node] = 0

    @ti.kernel
    def _finalize_leaves_scatter(self):
        """Распределение частиц по слотам листьев (параллельно)."""
        n = self.num_particles[None]
        for i in range(n):
            node = self.part_node[i]
            if node >= 0 and self.nodes.first_child[node] < 0:
                ls = self.nodes.leaf_start[node]
                if ls >= 0:
                    slot = ti.atomic_add(self.nodes.count[node], 1)
                    self.leaf_indices[ls + slot] = i

    @ti.kernel
    def _build_next_pointers_pass1(self, level_start: ti.i32, level_end: ti.i32):
        """Проход 1: установка указателей next для цепочки сиблингов (параллельно по уровню).

        В рамках одного уровня каждый родитель записывает в свои непересекающиеся дочерние узлы,
        поэтому параллелизация безопасна. Межуровневая зависимость (parent_next читается из
        next родителя, установленного дедом) разрешается послойным вызовом сверху вниз.
        """
        for i in range(level_start, level_end):
            parent_idx = self.parents[i]
            base = self.parent_first_child[i]
            parent_next = self.nodes.next[parent_idx]

            for oct in range(8):
                child = base + oct
                nxt = parent_next if oct == 7 else base + oct + 1
                self.nodes.next[child] = nxt

    @ti.kernel
    def _build_next_pointers_pass2(self, level_start: ti.i32, level_end: ti.i32):
        """Проход 2: удаление пустых узлов (параллельно по уровню).

        Каждый родитель модифицирует только указатели next своих дочерних узлов и свой
        first_child — непересекающиеся в рамках одного уровня.
        """
        for i in range(level_start, level_end):
            parent_idx = self.parents[i]
            base = self.parent_first_child[i]

            if self.nodes.count[parent_idx] == 0:
                self.nodes.first_child[parent_idx] = self.nodes.next[parent_idx]
                continue

            first_occupied = -1
            last_occupied = -1
            for j in range(8):
                child_idx = base + j
                if self.nodes.count[child_idx] > 0:
                    if first_occupied < 0:
                        first_occupied = child_idx
                        last_occupied = child_idx
                    else:
                        self.nodes.next[last_occupied] = child_idx
                        last_occupied = child_idx

            if last_occupied >= 0:
                self.nodes.next[last_occupied] = self.nodes.next[parent_idx]
                self.nodes.first_child[parent_idx] = first_occupied

    def _build_next_pointers(self):
        """Построение указателей next послойно (параллельно внутри каждого уровня)."""
        total = self.parent_count[None]
        offsets = self._parent_level_offsets
        # Проход 1 сверху вниз: установка цепочек сиблингов
        for lev in range(len(offsets)):
            lstart = offsets[lev]
            lend = offsets[lev + 1] if lev + 1 < len(offsets) else total
            if lend > lstart:
                self._build_next_pointers_pass1(lstart, lend)
        # Проход 2 сверху вниз: удаление пустых узлов
        for lev in range(len(offsets)):
            lstart = offsets[lev]
            lend = offsets[lev + 1] if lev + 1 < len(offsets) else total
            if lend > lstart:
                self._build_next_pointers_pass2(lstart, lend)

    # =================================================================
    # Распространение снизу вверх — общая логика (Исправление #3: DRY)
    # =================================================================

    @ti.kernel
    def _init_leaf_r3(self):
        """Проход 1 для R³: обнуление всех агрегатов, вычисление R³ листьев из частиц."""
        n_nodes = self.node_count[None]
        for idx in range(n_nodes):
            self.nodes.R3_sum[idx] = 0.0
            self.nodes.R3_cx[idx] = 0.0
            self.nodes.R3_cy[idx] = 0.0
            self.nodes.R3_cz[idx] = 0.0
            self.nodes.force_sum_x[idx] = 0.0
            self.nodes.force_sum_y[idx] = 0.0
            self.nodes.force_sum_z[idx] = 0.0

            if (self.nodes.is_leaf(idx) and self.nodes.count[idx] > 0
                    and self.nodes.leaf_start[idx] >= 0):
                ls = self.nodes.leaf_start[idx]
                cnt = self.nodes.count[idx]
                r3_sum = ti.f64(0.0)
                wcx = ti.f64(0.0)
                wcy = ti.f64(0.0)
                wcz = ti.f64(0.0)
                for k in range(cnt):
                    pi = self.leaf_indices[ls + k]
                    if pi >= 0:
                        r3 = self.particle_radii[pi] ** 3
                        pos = self.particle_positions[pi]
                        r3_sum += r3
                        wcx += pos[0] * r3
                        wcy += pos[1] * r3
                        wcz += pos[2] * r3
                self.nodes.R3_sum[idx] = r3_sum
                self.nodes.R3_cx[idx] = wcx
                self.nodes.R3_cy[idx] = wcy
                self.nodes.R3_cz[idx] = wcz

    @ti.kernel
    def _sweep_r3_level(self, level_start: ti.i32, level_end: ti.i32):
        """Параллельный проход R³ для одного уровня родителей.

        Все родители в [level_start, level_end) находятся на одном уровне дерева,
        поэтому их дочерние узлы уже заполнены — параллелизация безопасна.
        """
        for i in range(level_start, level_end):
            parent_idx = self.parents[i]
            base = self.parent_first_child[i]
            r3s = ti.f64(0.0)
            wcx = ti.f64(0.0)
            wcy = ti.f64(0.0)
            wcz = ti.f64(0.0)
            for j in range(8):
                child_idx = base + j
                if self.nodes.count[child_idx] > 0:
                    r3s += self.nodes.R3_sum[child_idx]
                    wcx += self.nodes.R3_cx[child_idx]
                    wcy += self.nodes.R3_cy[child_idx]
                    wcz += self.nodes.R3_cz[child_idx]
            self.nodes.R3_sum[parent_idx] = r3s
            self.nodes.R3_cx[parent_idx] = wcx
            self.nodes.R3_cy[parent_idx] = wcy
            self.nodes.R3_cz[parent_idx] = wcz

    def _propagate_r3(self):
        """Распространение R³-агрегатов снизу вверх (двухпроходное, параллельно по уровням)."""
        self._init_leaf_r3()
        total = self.parent_count[None]
        offsets = self._parent_level_offsets
        # Проход от самого глубокого уровня к корню
        for lev in range(len(offsets) - 1, -1, -1):
            lstart = offsets[lev]
            lend = offsets[lev + 1] if lev + 1 < len(offsets) else total
            if lend > lstart:
                self._sweep_r3_level(lstart, lend)

    @ti.kernel
    def _init_leaf_forces(self, forces_np: ti.types.ndarray()):
        """Проход 1 для сил: обнуление всего, вычисление сумм сил и сырых моментов для листьев."""
        n_nodes = self.node_count[None]
        for idx in range(n_nodes):
            self.nodes.force_sum_x[idx] = 0.0
            self.nodes.force_sum_y[idx] = 0.0
            self.nodes.force_sum_z[idx] = 0.0
            self.nodes.fmom_xx[idx] = 0.0
            self.nodes.fmom_xy[idx] = 0.0
            self.nodes.fmom_xz[idx] = 0.0
            self.nodes.fmom_yx[idx] = 0.0
            self.nodes.fmom_yy[idx] = 0.0
            self.nodes.fmom_yz[idx] = 0.0
            self.nodes.fmom_zx[idx] = 0.0
            self.nodes.fmom_zy[idx] = 0.0
            self.nodes.fmom_zz[idx] = 0.0

            if (self.nodes.is_leaf(idx) and self.nodes.count[idx] > 0
                    and self.nodes.leaf_start[idx] >= 0):
                ls = self.nodes.leaf_start[idx]
                cnt = self.nodes.count[idx]
                fsx = ti.f64(0.0)
                fsy = ti.f64(0.0)
                fsz = ti.f64(0.0)
                pxx = ti.f64(0.0); pxy = ti.f64(0.0); pxz = ti.f64(0.0)
                pyx = ti.f64(0.0); pyy = ti.f64(0.0); pyz = ti.f64(0.0)
                pzx = ti.f64(0.0); pzy = ti.f64(0.0); pzz = ti.f64(0.0)
                for k in range(cnt):
                    pi = self.leaf_indices[ls + k]
                    if pi >= 0:
                        fx = forces_np[pi, 0]
                        fy = forces_np[pi, 1]
                        fz = forces_np[pi, 2]
                        fsx += fx
                        fsy += fy
                        fsz += fz
                        # Сырые силовые моменты P_αβ = Σ r_j,α * F_j,β
                        rx = self.particle_positions[pi][0]
                        ry = self.particle_positions[pi][1]
                        rz = self.particle_positions[pi][2]
                        pxx += rx * fx; pxy += rx * fy; pxz += rx * fz
                        pyx += ry * fx; pyy += ry * fy; pyz += ry * fz
                        pzx += rz * fx; pzy += rz * fy; pzz += rz * fz
                self.nodes.force_sum_x[idx] = fsx
                self.nodes.force_sum_y[idx] = fsy
                self.nodes.force_sum_z[idx] = fsz
                self.nodes.fmom_xx[idx] = pxx; self.nodes.fmom_xy[idx] = pxy
                self.nodes.fmom_xz[idx] = pxz; self.nodes.fmom_yx[idx] = pyx
                self.nodes.fmom_yy[idx] = pyy; self.nodes.fmom_yz[idx] = pyz
                self.nodes.fmom_zx[idx] = pzx; self.nodes.fmom_zy[idx] = pzy
                self.nodes.fmom_zz[idx] = pzz

    @ti.kernel
    def _sweep_forces_level(self, level_start: ti.i32, level_end: ti.i32):
        """Параллельный проход по силам для одного уровня родителей."""
        for i in range(level_start, level_end):
            parent_idx = self.parents[i]
            base = self.parent_first_child[i]
            fsx = ti.f64(0.0)
            fsy = ti.f64(0.0)
            fsz = ti.f64(0.0)
            pxx = ti.f64(0.0); pxy = ti.f64(0.0); pxz = ti.f64(0.0)
            pyx = ti.f64(0.0); pyy = ti.f64(0.0); pyz = ti.f64(0.0)
            pzx = ti.f64(0.0); pzy = ti.f64(0.0); pzz = ti.f64(0.0)
            for j in range(8):
                child_idx = base + j
                if self.nodes.count[child_idx] > 0:
                    fsx += self.nodes.force_sum_x[child_idx]
                    fsy += self.nodes.force_sum_y[child_idx]
                    fsz += self.nodes.force_sum_z[child_idx]
                    pxx += self.nodes.fmom_xx[child_idx]
                    pxy += self.nodes.fmom_xy[child_idx]
                    pxz += self.nodes.fmom_xz[child_idx]
                    pyx += self.nodes.fmom_yx[child_idx]
                    pyy += self.nodes.fmom_yy[child_idx]
                    pyz += self.nodes.fmom_yz[child_idx]
                    pzx += self.nodes.fmom_zx[child_idx]
                    pzy += self.nodes.fmom_zy[child_idx]
                    pzz += self.nodes.fmom_zz[child_idx]
            self.nodes.force_sum_x[parent_idx] = fsx
            self.nodes.force_sum_y[parent_idx] = fsy
            self.nodes.force_sum_z[parent_idx] = fsz
            self.nodes.fmom_xx[parent_idx] = pxx; self.nodes.fmom_xy[parent_idx] = pxy
            self.nodes.fmom_xz[parent_idx] = pxz; self.nodes.fmom_yx[parent_idx] = pyx
            self.nodes.fmom_yy[parent_idx] = pyy; self.nodes.fmom_yz[parent_idx] = pyz
            self.nodes.fmom_zx[parent_idx] = pzx; self.nodes.fmom_zy[parent_idx] = pzy
            self.nodes.fmom_zz[parent_idx] = pzz

    def _sweep_bottom_up(self):
        """Параллельный по уровням обход снизу вверх (общий для R³ и сил)."""
        total = self.parent_count[None]
        offsets = self._parent_level_offsets
        for lev in range(len(offsets) - 1, -1, -1):
            lstart = offsets[lev]
            lend = offsets[lev + 1] if lev + 1 < len(offsets) else total
            if lend > lstart:
                self._sweep_r3_level(lstart, lend)
                self._sweep_forces_level(lstart, lend)

    def _propagate_forces(self, forces_np: np.ndarray):
        """Распространение агрегатов сил + сырых моментов снизу вверх (двухпроходное, параллельно по уровням)."""
        self._init_leaf_forces(forces_np)
        total = self.parent_count[None]
        offsets = self._parent_level_offsets
        for lev in range(len(offsets) - 1, -1, -1):
            lstart = offsets[lev]
            lend = offsets[lev + 1] if lev + 1 < len(offsets) else total
            if lend > lstart:
                self._sweep_forces_level(lstart, lend)

    @ti.kernel
    def _init_leaf_forces_from_fields(self):
        """Проход 1 для сил (из полей Taichi): читает из поля particle_forces."""
        n_nodes = self.node_count[None]
        for idx in range(n_nodes):
            self.nodes.force_sum_x[idx] = 0.0
            self.nodes.force_sum_y[idx] = 0.0
            self.nodes.force_sum_z[idx] = 0.0
            self.nodes.fmom_xx[idx] = 0.0
            self.nodes.fmom_xy[idx] = 0.0
            self.nodes.fmom_xz[idx] = 0.0
            self.nodes.fmom_yx[idx] = 0.0
            self.nodes.fmom_yy[idx] = 0.0
            self.nodes.fmom_yz[idx] = 0.0
            self.nodes.fmom_zx[idx] = 0.0
            self.nodes.fmom_zy[idx] = 0.0
            self.nodes.fmom_zz[idx] = 0.0

            if (self.nodes.is_leaf(idx) and self.nodes.count[idx] > 0
                    and self.nodes.leaf_start[idx] >= 0):
                ls = self.nodes.leaf_start[idx]
                cnt = self.nodes.count[idx]
                fsx = ti.f64(0.0)
                fsy = ti.f64(0.0)
                fsz = ti.f64(0.0)
                pxx = ti.f64(0.0); pxy = ti.f64(0.0); pxz = ti.f64(0.0)
                pyx = ti.f64(0.0); pyy = ti.f64(0.0); pyz = ti.f64(0.0)
                pzx = ti.f64(0.0); pzy = ti.f64(0.0); pzz = ti.f64(0.0)
                for k in range(cnt):
                    pi = self.leaf_indices[ls + k]
                    if pi >= 0:
                        f = self.particle_forces[pi]
                        fx = f[0]
                        fy = f[1]
                        fz = f[2]
                        fsx += fx
                        fsy += fy
                        fsz += fz
                        rx = self.particle_positions[pi][0]
                        ry = self.particle_positions[pi][1]
                        rz = self.particle_positions[pi][2]
                        pxx += rx * fx; pxy += rx * fy; pxz += rx * fz
                        pyx += ry * fx; pyy += ry * fy; pyz += ry * fz
                        pzx += rz * fx; pzy += rz * fy; pzz += rz * fz
                self.nodes.force_sum_x[idx] = fsx
                self.nodes.force_sum_y[idx] = fsy
                self.nodes.force_sum_z[idx] = fsz
                self.nodes.fmom_xx[idx] = pxx; self.nodes.fmom_xy[idx] = pxy
                self.nodes.fmom_xz[idx] = pxz; self.nodes.fmom_yx[idx] = pyx
                self.nodes.fmom_yy[idx] = pyy; self.nodes.fmom_yz[idx] = pyz
                self.nodes.fmom_zx[idx] = pzx; self.nodes.fmom_zy[idx] = pzy
                self.nodes.fmom_zz[idx] = pzz

    def _propagate_forces_from_fields(self):
        """Распространение агрегатов сил снизу вверх, используя силы из полей Taichi."""
        self._init_leaf_forces_from_fields()
        total = self.parent_count[None]
        offsets = self._parent_level_offsets
        for lev in range(len(offsets) - 1, -1, -1):
            lstart = offsets[lev]
            lend = offsets[lev + 1] if lev + 1 < len(offsets) else total
            if lend > lstart:
                self._sweep_forces_level(lstart, lend)

    # =================================================================
    # Вычисление сил (дипольных) — параллельный обход
    # =================================================================

    @ti.kernel
    def _compute_forces_kernel(self,
                               fx: ti.types.ndarray(),
                               fy: ti.types.ndarray(),
                               fz: ti.types.ndarray(),
                               m_const: ti.f64):
        n = self.num_particles[None]
        theta_sq = self.theta_sq[None]
        r3_threshold = _R3_SKIP_THRESHOLD  # Исправление #6: именованная константа

        for i in range(n):
            pos_i = self.particle_positions[i]
            r_i3 = self.particle_radii[i] ** 3
            f_x = ti.f64(0.0)
            f_y = ti.f64(0.0)
            f_z = ti.f64(0.0)

            node_idx = 0
            while node_idx >= 0:
                cnt = self.nodes.count[node_idx]
                if cnt == 0 or self.nodes.R3_sum[node_idx] < r3_threshold:
                    node_idx = self.nodes.next[node_idx]
                    continue

                is_leaf = self.nodes.first_child[node_idx] < 0

                if is_leaf:
                    # Лист: точное попарное взаимодействие со всеми частицами
                    # cnt = число частиц в этом листе (≤ mpl)
                    ls = self.nodes.leaf_start[node_idx]
                    for k in range(cnt):
                        j = self.leaf_indices[ls + k]
                        if j >= 0 and j != i:
                            pos_j = self.particle_positions[j]
                            dx = pos_j[0] - pos_i[0]
                            dy = pos_j[1] - pos_i[1]
                            dz = pos_j[2] - pos_i[2]

                            if self.periodic[None] == 1:
                                L_val = self.L[None]
                                if ti.abs(dx) > L_val * 0.5:
                                    dx -= ti.math.sign(dx) * L_val
                                if ti.abs(dy) > L_val * 0.5:
                                    dy -= ti.math.sign(dy) * L_val
                                if ti.abs(dz) > L_val * 0.5:
                                    dz -= ti.math.sign(dz) * L_val

                            r_sq = dx * dx + dy * dy + dz * dz
                            r_sum = self.particle_radii[i] + self.particle_radii[j]

                            if r_sq >= r_sum * r_sum:
                                r = ti.sqrt(r_sq)
                                r7 = r_sq * r_sq * r_sq * r
                                r_j3 = self.particle_radii[j] ** 3
                                factor = m_const * r_i3 * r_j3 / r7

                                dx2 = dx * dx
                                dy2 = dy * dy
                                dz2 = dz * dz
                                ang_xy = 4.0 * dz2 - dx2 - dy2
                                f_x += factor * dx * ang_xy
                                f_y += factor * dy * ang_xy
                                f_z += factor * dz * (2.0 * dz2 - 3.0 * dx2 - 3.0 * dy2)

                    node_idx = self.nodes.next[node_idx]
                    continue

                # Внутренний узел: критерий открытия Барнса-Хатта
                # cnt = суммарное число частиц в поддереве (используется для раннего пропуска выше)
                s = self.nodes.size(node_idx)
                center = self.nodes.get_center(node_idx)
                dx = center[0] - pos_i[0]
                dy = center[1] - pos_i[1]
                dz = center[2] - pos_i[2]

                if self.periodic[None] == 1:
                    L_val = self.L[None]
                    if ti.abs(dx) > L_val * 0.5:
                        dx -= ti.math.sign(dx) * L_val
                    if ti.abs(dy) > L_val * 0.5:
                        dy -= ti.math.sign(dy) * L_val
                    if ti.abs(dz) > L_val * 0.5:
                        dz -= ti.math.sign(dz) * L_val

                D_sq = dx * dx + dy * dy + dz * dz

                if s * s < theta_sq * D_sq and D_sq > 0.0:
                    # Аппроксимация: используем агрегированный R³ и центр
                    r = ti.sqrt(D_sq)
                    r7 = D_sq * D_sq * D_sq * r
                    R3_eff = self.nodes.R3_sum[node_idx]
                    factor = m_const * r_i3 * R3_eff / r7

                    dx2 = dx * dx
                    dy2 = dy * dy
                    dz2 = dz * dz
                    ang_xy = 4.0 * dz2 - dx2 - dy2
                    f_x += factor * dx * ang_xy
                    f_y += factor * dy * ang_xy
                    f_z += factor * dz * (2.0 * dz2 - 3.0 * dx2 - 3.0 * dy2)

                    node_idx = self.nodes.next[node_idx]
                else:
                    node_idx = self.nodes.first_child[node_idx]

            fx[i] = f_x
            fy[i] = f_y
            fz[i] = f_z
            # Сохраняем силы в поле Taichi для _init_leaf_forces_from_fields
            self.particle_forces[i] = ti.Vector([f_x, f_y, f_z])

    def compute_forces(self, m_const: float) -> np.ndarray:
        """Вычисление дипольных сил. m_const = 12*pi*eps0*eps_oil*E^2."""
        n = self.num_particles[None]
        fx = self._out_fx[:n]
        fy = self._out_fy[:n]
        fz = self._out_fz[:n]
        fx[:] = 0.0
        fy[:] = 0.0
        fz[:] = 0.0
        self._compute_forces_kernel(fx, fy, fz, m_const)
        return np.stack([fx, fy, fz], axis=1)

    # =================================================================
    # Вычисление стокслета (конвекции) — параллельный обход
    # =================================================================

    @staticmethod
    @ti.func
    def _trilinear_interp(grid: ti.template(), x: ti.f64, y: ti.f64, z: ti.f64,
                          grid_min: ti.f64, inv_dx: ti.f64, n: ti.i32) -> ti.f64:
        """Трилинейная интерполяция на регулярной 3D сетке с clamping."""
        fx = (x - grid_min) * inv_dx
        fy = (y - grid_min) * inv_dx
        fz = (z - grid_min) * inv_dx
        n_max = ti.cast(n - 1, ti.f64)
        fx = ti.max(0.0, ti.min(fx, n_max))
        fy = ti.max(0.0, ti.min(fy, n_max))
        fz = ti.max(0.0, ti.min(fz, n_max))
        ix = ti.min(ti.cast(ti.floor(fx), ti.i32), n - 2)
        iy = ti.min(ti.cast(ti.floor(fy), ti.i32), n - 2)
        iz = ti.min(ti.cast(ti.floor(fz), ti.i32), n - 2)
        tx = fx - ti.cast(ix, ti.f64)
        ty = fy - ti.cast(iy, ti.f64)
        tz = fz - ti.cast(iz, ti.f64)
        c000 = grid[ix, iy, iz]
        c001 = grid[ix, iy, iz + 1]
        c010 = grid[ix, iy + 1, iz]
        c011 = grid[ix, iy + 1, iz + 1]
        c100 = grid[ix + 1, iy, iz]
        c101 = grid[ix + 1, iy, iz + 1]
        c110 = grid[ix + 1, iy + 1, iz]
        c111 = grid[ix + 1, iy + 1, iz + 1]
        return (c000 * (1 - tx) * (1 - ty) * (1 - tz) +
                c001 * (1 - tx) * (1 - ty) * tz +
                c010 * (1 - tx) * ty * (1 - tz) +
                c011 * (1 - tx) * ty * tz +
                c100 * tx * (1 - ty) * (1 - tz) +
                c101 * tx * (1 - ty) * tz +
                c110 * tx * ty * (1 - tz) +
                c111 * tx * ty * tz)

    @staticmethod
    @ti.func
    def _precompute_grid_idx(x: ti.f64, y: ti.f64, z: ti.f64,
                             grid_min: ti.f64, inv_dx: ti.f64, n: ti.i32):
        fx = (x - grid_min) * inv_dx
        fy = (y - grid_min) * inv_dx
        fz = (z - grid_min) * inv_dx
        n_max = ti.cast(n - 1, ti.f64)
        fx = ti.max(0.0, ti.min(fx, n_max))
        fy = ti.max(0.0, ti.min(fy, n_max))
        fz = ti.max(0.0, ti.min(fz, n_max))
        ix = ti.min(ti.cast(ti.floor(fx), ti.i32), n - 2)
        iy = ti.min(ti.cast(ti.floor(fy), ti.i32), n - 2)
        iz = ti.min(ti.cast(ti.floor(fz), ti.i32), n - 2)
        tx = fx - ti.cast(ix, ti.f64)
        ty = fy - ti.cast(iy, ti.f64)
        tz = fz - ti.cast(iz, ti.f64)
        return ix, iy, iz, tx, ty, tz

    @staticmethod
    @ti.func
    def _interp_precomp(grid: ti.template(),
                        ix: ti.i32, iy: ti.i32, iz: ti.i32,
                        tx: ti.f64, ty: ti.f64, tz: ti.f64) -> ti.f64:
        c000 = grid[ix, iy, iz]
        c001 = grid[ix, iy, iz + 1]
        c010 = grid[ix, iy + 1, iz]
        c011 = grid[ix, iy + 1, iz + 1]
        c100 = grid[ix + 1, iy, iz]
        c101 = grid[ix + 1, iy, iz + 1]
        c110 = grid[ix + 1, iy + 1, iz]
        c111 = grid[ix + 1, iy + 1, iz + 1]
        return (c000 * (1 - tx) * (1 - ty) * (1 - tz) +
                c001 * (1 - tx) * (1 - ty) * tz +
                c010 * (1 - tx) * ty * (1 - tz) +
                c011 * (1 - tx) * ty * tz +
                c100 * tx * (1 - ty) * (1 - tz) +
                c101 * tx * (1 - ty) * tz +
                c110 * tx * ty * (1 - tz) +
                c111 * tx * ty * tz)

    @ti.kernel
    def _compute_stokeslet_kernel(self,
                                  forces_np: ti.types.ndarray(),
                                  vx: ti.types.ndarray(),
                                  vy: ti.types.ndarray(),
                                  vz: ti.types.ndarray(),
                                  eta_const: ti.f64):
        n = self.num_particles[None]
        theta_sq = self.theta_sq[None]
        fsq_threshold = _FORCE_SQ_SKIP_THRESHOLD  # Исправление #6: именованная константа

        for i in range(n):
            pos_i = self.particle_positions[i]
            v_x = ti.f64(0.0)
            v_y = ti.f64(0.0)
            v_z = ti.f64(0.0)

            node_idx = 0
            while node_idx >= 0:
                cnt = self.nodes.count[node_idx]
                fs2 = (self.nodes.force_sum_x[node_idx] ** 2 +
                       self.nodes.force_sum_y[node_idx] ** 2 +
                       self.nodes.force_sum_z[node_idx] ** 2)
                if cnt == 0 or fs2 < fsq_threshold:
                    node_idx = self.nodes.next[node_idx]
                    continue

                is_leaf = self.nodes.first_child[node_idx] < 0

                if is_leaf:
                    ls = self.nodes.leaf_start[node_idx]
                    for k in range(cnt):
                        j = self.leaf_indices[ls + k]
                        if j >= 0 and j != i:
                            pos_j = self.particle_positions[j]
                            dx = pos_i[0] - pos_j[0]
                            dy = pos_i[1] - pos_j[1]
                            dz = pos_i[2] - pos_j[2]

                            if self.periodic[None] == 1:
                                L_val = self.L[None]
                                if ti.abs(dx) > L_val * 0.5:
                                    dx -= ti.math.sign(dx) * L_val
                                if ti.abs(dy) > L_val * 0.5:
                                    dy -= ti.math.sign(dy) * L_val
                                if ti.abs(dz) > L_val * 0.5:
                                    dz -= ti.math.sign(dz) * L_val

                            r_sq = dx * dx + dy * dy + dz * dz
                            r_sum = self.particle_radii[i] + self.particle_radii[j]

                            if r_sq >= r_sum * r_sum:
                                r = ti.sqrt(r_sq)
                                inv_r = 1.0 / r
                                inv_r2 = inv_r * inv_r

                                Fx = forces_np[j, 0]
                                Fy = forces_np[j, 1]
                                Fz = forces_np[j, 2]
                                dot = dx * Fx + dy * Fy + dz * Fz

                                coeff = eta_const * inv_r
                                v_x += coeff * (Fx + dx * inv_r2 * dot)
                                v_y += coeff * (Fy + dy * inv_r2 * dot)
                                v_z += coeff * (Fz + dz * inv_r2 * dot)

                                # Поправка от периодических образов (COMSOL) — полный тензор
                                if self.corr_enabled[None] == 1:
                                    cL = self.corr_L_ratio[None]
                                    xc = dx * cL
                                    yc = dy * cL
                                    zc = dz * cL
                                    gmin = self.corr_grid_min[None]
                                    ginv = self.corr_grid_inv_dx[None]
                                    gn = self.corr_grid_n[None]
                                    Fz_inv = self.corr_Fz_inv[None]
                                    eta_r = self.corr_eta_ratio[None]

                                    # G_z: (xc, yc, zc) — предвычисление индексов один раз
                                    i1x, i1y, i1z, t1x, t1y, t1z = self._precompute_grid_idx(xc, yc, zc, gmin, ginv, gn)
                                    gz_x = self._interp_precomp(self.corr_grid_u, i1x, i1y, i1z, t1x, t1y, t1z)
                                    gz_y = self._interp_precomp(self.corr_grid_v, i1x, i1y, i1z, t1x, t1y, t1z)
                                    gz_z = self._interp_precomp(self.corr_grid_w, i1x, i1y, i1z, t1x, t1y, t1z)

                                    # G_x: перестановка x↔z → (zc, yc, xc)
                                    i2x, i2y, i2z, t2x, t2y, t2z = self._precompute_grid_idx(zc, yc, xc, gmin, ginv, gn)
                                    gx_x = self._interp_precomp(self.corr_grid_w, i2x, i2y, i2z, t2x, t2y, t2z)
                                    gx_y = self._interp_precomp(self.corr_grid_v, i2x, i2y, i2z, t2x, t2y, t2z)
                                    gx_z = self._interp_precomp(self.corr_grid_u, i2x, i2y, i2z, t2x, t2y, t2z)

                                    # G_y: перестановка y↔z → (xc, zc, yc)
                                    i3x, i3y, i3z, t3x, t3y, t3z = self._precompute_grid_idx(xc, zc, yc, gmin, ginv, gn)
                                    gy_x = self._interp_precomp(self.corr_grid_u, i3x, i3y, i3z, t3x, t3y, t3z)
                                    gy_y = self._interp_precomp(self.corr_grid_w, i3x, i3y, i3z, t3x, t3y, t3z)
                                    gy_z = self._interp_precomp(self.corr_grid_v, i3x, i3y, i3z, t3x, t3y, t3z)

                                    common = Fz_inv * eta_r * cL
                                    sx = Fx * common
                                    sy = Fy * common
                                    sz = Fz * common

                                    v_x += gx_x * sx + gy_x * sy + gz_x * sz
                                    v_y += gx_y * sx + gy_y * sy + gz_y * sz
                                    v_z += gx_z * sx + gy_z * sy + gz_z * sz

                    node_idx = self.nodes.next[node_idx]
                    continue

                # Внутренний узел: критерий открытия
                s = self.nodes.size(node_idx)
                center = self.nodes.get_center(node_idx)
                dx = pos_i[0] - center[0]
                dy = pos_i[1] - center[1]
                dz = pos_i[2] - center[2]

                if self.periodic[None] == 1:
                    L_val = self.L[None]
                    if ti.abs(dx) > L_val * 0.5:
                        dx -= ti.math.sign(dx) * L_val
                    if ti.abs(dy) > L_val * 0.5:
                        dy -= ti.math.sign(dy) * L_val
                    if ti.abs(dz) > L_val * 0.5:
                        dz -= ti.math.sign(dz) * L_val

                D_sq = dx * dx + dy * dy + dz * dz

                if s * s < theta_sq * D_sq and D_sq > 0.0:
                    r = ti.sqrt(D_sq)
                    inv_r = 1.0 / r
                    inv_r2 = inv_r * inv_r

                    Fx = self.nodes.force_sum_x[node_idx]
                    Fy = self.nodes.force_sum_y[node_idx]
                    Fz = self.nodes.force_sum_z[node_idx]
                    dot = dx * Fx + dy * Fy + dz * Fz

                    # Монопольный вклад
                    coeff = eta_const * inv_r
                    v_x += coeff * (Fx + dx * inv_r2 * dot)
                    v_y += coeff * (Fy + dy * inv_r2 * dot)
                    v_z += coeff * (Fz + dz * inv_r2 * dot)

                    # Поправка от периодических образов (COMSOL) — полный тензор, монопольное приближение
                    if self.corr_enabled[None] == 1:
                        cL = self.corr_L_ratio[None]
                        xc = dx * cL
                        yc = dy * cL
                        zc = dz * cL
                        gmin = self.corr_grid_min[None]
                        ginv = self.corr_grid_inv_dx[None]
                        gn = self.corr_grid_n[None]
                        Fz_inv = self.corr_Fz_inv[None]
                        eta_r = self.corr_eta_ratio[None]

                        # G_z: (xc, yc, zc) — предвычисление индексов один раз
                        i1x, i1y, i1z, t1x, t1y, t1z = self._precompute_grid_idx(xc, yc, zc, gmin, ginv, gn)
                        gz_x = self._interp_precomp(self.corr_grid_u, i1x, i1y, i1z, t1x, t1y, t1z)
                        gz_y = self._interp_precomp(self.corr_grid_v, i1x, i1y, i1z, t1x, t1y, t1z)
                        gz_z = self._interp_precomp(self.corr_grid_w, i1x, i1y, i1z, t1x, t1y, t1z)

                        # G_x: перестановка x↔z → (zc, yc, xc)
                        i2x, i2y, i2z, t2x, t2y, t2z = self._precompute_grid_idx(zc, yc, xc, gmin, ginv, gn)
                        gx_x = self._interp_precomp(self.corr_grid_w, i2x, i2y, i2z, t2x, t2y, t2z)
                        gx_y = self._interp_precomp(self.corr_grid_v, i2x, i2y, i2z, t2x, t2y, t2z)
                        gx_z = self._interp_precomp(self.corr_grid_u, i2x, i2y, i2z, t2x, t2y, t2z)

                        # G_y: перестановка y↔z → (xc, zc, yc)
                        i3x, i3y, i3z, t3x, t3y, t3z = self._precompute_grid_idx(xc, zc, yc, gmin, ginv, gn)
                        gy_x = self._interp_precomp(self.corr_grid_u, i3x, i3y, i3z, t3x, t3y, t3z)
                        gy_y = self._interp_precomp(self.corr_grid_w, i3x, i3y, i3z, t3x, t3y, t3z)
                        gy_z = self._interp_precomp(self.corr_grid_v, i3x, i3y, i3z, t3x, t3y, t3z)

                        common = Fz_inv * eta_r * cL
                        sx = Fx * common
                        sy = Fy * common
                        sz = Fz * common

                        v_x += gx_x * sx + gy_x * sy + gz_x * sz
                        v_y += gx_y * sx + gy_y * sy + gz_y * sz
                        v_z += gx_z * sx + gy_z * sy + gz_z * sz

                    # --- Дипольная коррекция: δv = ∇T : D ---
                    # R³-взвешенный центр масс
                    r3s = self.nodes.R3_sum[node_idx]
                    cmx = ti.f64(0.0)
                    cmy = ti.f64(0.0)
                    cmz = ti.f64(0.0)
                    if r3s > 0.0:
                        inv_r3s = 1.0 / r3s
                        cmx = self.nodes.R3_cx[node_idx] * inv_r3s
                        cmy = self.nodes.R3_cy[node_idx] * inv_r3s
                        cmz = self.nodes.R3_cz[node_idx] * inv_r3s
                    else:
                        cmx = (self.nodes.min_x[node_idx] + self.nodes.max_x[node_idx]) * 0.5
                        cmy = (self.nodes.min_y[node_idx] + self.nodes.max_y[node_idx]) * 0.5
                        cmz = (self.nodes.min_z[node_idx] + self.nodes.max_z[node_idx]) * 0.5

                    # D_αβ = P_αβ - r_cm,α * F_eff,β (дипольный тензор)
                    Dxx = self.nodes.fmom_xx[node_idx] - cmx * Fx
                    Dxy = self.nodes.fmom_xy[node_idx] - cmx * Fy
                    Dxz = self.nodes.fmom_xz[node_idx] - cmx * Fz
                    Dyx = self.nodes.fmom_yx[node_idx] - cmy * Fx
                    Dyy = self.nodes.fmom_yy[node_idx] - cmy * Fy
                    Dyz = self.nodes.fmom_yz[node_idx] - cmy * Fz
                    Dzx = self.nodes.fmom_zx[node_idx] - cmz * Fx
                    Dzy = self.nodes.fmom_zy[node_idx] - cmz * Fy
                    Dzz = self.nodes.fmom_zz[node_idx] - cmz * Fz

                    rx = dx; ry = dy; rz = dz

                    # u = D^T · R (транспонированный дипольный тензор на вектор)
                    ux = Dxx * rx + Dyx * ry + Dzx * rz
                    uy = Dxy * rx + Dyy * ry + Dzy * rz
                    uz = Dxz * rx + Dyz * ry + Dzz * rz

                    # w = D · R (дипольный тензор на вектор)
                    wx = Dxx * rx + Dxy * ry + Dxz * rz
                    wy = Dyx * rx + Dyy * ry + Dyz * rz
                    wz = Dzx * rx + Dzy * ry + Dzz * rz

                    # s_tr = tr(D), q = R^T D R = R · w (след тензора и квадратичная форма)
                    s_tr = Dxx + Dyy + Dzz
                    q = rx * wx + ry * wy + rz * wz

                    inv_r3 = inv_r * inv_r2
                    inv_r5 = inv_r3 * inv_r2

                    v_x += eta_const * (inv_r3 * (ux - wx - s_tr * rx) + 3.0 * rx * q * inv_r5)
                    v_y += eta_const * (inv_r3 * (uy - wy - s_tr * ry) + 3.0 * ry * q * inv_r5)
                    v_z += eta_const * (inv_r3 * (uz - wz - s_tr * rz) + 3.0 * rz * q * inv_r5)

                    node_idx = self.nodes.next[node_idx]
                else:
                    node_idx = self.nodes.first_child[node_idx]

            vx[i] = v_x
            vy[i] = v_y
            vz[i] = v_z

    def compute_stokeslet(self, forces: np.ndarray, eta_const: float) -> np.ndarray:
        """Вычисление стокслет-скоростей. eta_const = 1/(8*pi*eta_oil)."""
        n = self.num_particles[None]
        forces_c = np.ascontiguousarray(forces[:n], dtype=np.float64)
        vx = self._out_vx[:n]
        vy = self._out_vy[:n]
        vz = self._out_vz[:n]
        vx[:] = 0.0
        vy[:] = 0.0
        vz[:] = 0.0
        self._propagate_forces(forces_c)
        self._compute_stokeslet_kernel(forces_c, vx, vy, vz, eta_const)
        return np.stack([vx, vy, vz], axis=1)

    def compute_stokeslet_from_fields(self, forces: np.ndarray, eta_const: float) -> np.ndarray:
        """Вычисление стокслета с использованием сил из полей Taichi (без лишнего копирования).

        Вызывать после compute_forces() — использует поле particle_forces,
        записанное ядром расчёта сил, вместо повторного копирования массива.
        """
        n = self.num_particles[None]
        forces_c = np.ascontiguousarray(forces[:n], dtype=np.float64)
        vx = self._out_vx[:n]
        vy = self._out_vy[:n]
        vz = self._out_vz[:n]
        vx[:] = 0.0
        vy[:] = 0.0
        vz[:] = 0.0
        self._propagate_forces_from_fields()
        self._compute_stokeslet_kernel(forces_c, vx, vy, vz, eta_const)
        return np.stack([vx, vy, vz], axis=1)

    def compute_forces_and_stokeslet(self, m_const: float, eta_const: float) -> tuple:
        """Совмещённый расчёт сил и стокслета.

        Оптимизация: ядро сил записывает в поле particle_forces,
        _init_leaf_forces_from_fields читает оттуда напрямую (без лишнего копирования).

        Returns: (forces (N,3), velocities (N,3))
        """
        forces = self.compute_forces(m_const)
        velocities = self.compute_stokeslet_from_fields(forces, eta_const)
        return forces, velocities

    @ti.kernel
    def _compute_total_velocity_kernel(self,
                                       forces_np: ti.types.ndarray(),
                                       vx: ti.types.ndarray(),
                                       vy: ti.types.ndarray(),
                                       vz: ti.types.ndarray(),
                                       eta_const: ti.f64,
                                       stokes_factor: ti.f64):
        """Объединённое ядро: v_total = v_migration + v_convection за один обход."""
        n = self.num_particles[None]
        theta_sq = self.theta_sq[None]
        fsq_threshold = _FORCE_SQ_SKIP_THRESHOLD

        for i in range(n):
            pos_i = self.particle_positions[i]
            fi = self.particle_forces[i]
            inv_stokes_r = 1.0 / (stokes_factor * self.particle_radii[i])
            v_x = fi[0] * inv_stokes_r
            v_y = fi[1] * inv_stokes_r
            v_z = fi[2] * inv_stokes_r

            node_idx = 0
            while node_idx >= 0:
                cnt = self.nodes.count[node_idx]
                fs2 = (self.nodes.force_sum_x[node_idx] ** 2 +
                       self.nodes.force_sum_y[node_idx] ** 2 +
                       self.nodes.force_sum_z[node_idx] ** 2)
                if cnt == 0 or fs2 < fsq_threshold:
                    node_idx = self.nodes.next[node_idx]
                    continue

                is_leaf = self.nodes.first_child[node_idx] < 0

                if is_leaf:
                    ls = self.nodes.leaf_start[node_idx]
                    for k in range(cnt):
                        j = self.leaf_indices[ls + k]
                        if j >= 0 and j != i:
                            pos_j = self.particle_positions[j]
                            dx = pos_i[0] - pos_j[0]
                            dy = pos_i[1] - pos_j[1]
                            dz = pos_i[2] - pos_j[2]

                            if self.periodic[None] == 1:
                                L_val = self.L[None]
                                if ti.abs(dx) > L_val * 0.5:
                                    dx -= ti.math.sign(dx) * L_val
                                if ti.abs(dy) > L_val * 0.5:
                                    dy -= ti.math.sign(dy) * L_val
                                if ti.abs(dz) > L_val * 0.5:
                                    dz -= ti.math.sign(dz) * L_val

                            r_sq = dx * dx + dy * dy + dz * dz
                            r_sum = self.particle_radii[i] + self.particle_radii[j]

                            if r_sq >= r_sum * r_sum:
                                r = ti.sqrt(r_sq)
                                inv_r = 1.0 / r
                                inv_r2 = inv_r * inv_r

                                Fx = forces_np[j, 0]
                                Fy = forces_np[j, 1]
                                Fz = forces_np[j, 2]
                                dot = dx * Fx + dy * Fy + dz * Fz

                                coeff = eta_const * inv_r
                                v_x += coeff * (Fx + dx * inv_r2 * dot)
                                v_y += coeff * (Fy + dy * inv_r2 * dot)
                                v_z += coeff * (Fz + dz * inv_r2 * dot)

                                if self.corr_enabled[None] == 1:
                                    cL = self.corr_L_ratio[None]
                                    xc = dx * cL
                                    yc = dy * cL
                                    zc = dz * cL
                                    gmin = self.corr_grid_min[None]
                                    ginv = self.corr_grid_inv_dx[None]
                                    gn = self.corr_grid_n[None]
                                    Fz_inv = self.corr_Fz_inv[None]
                                    eta_r = self.corr_eta_ratio[None]

                                    i1x, i1y, i1z, t1x, t1y, t1z = self._precompute_grid_idx(xc, yc, zc, gmin, ginv, gn)
                                    gz_x = self._interp_precomp(self.corr_grid_u, i1x, i1y, i1z, t1x, t1y, t1z)
                                    gz_y = self._interp_precomp(self.corr_grid_v, i1x, i1y, i1z, t1x, t1y, t1z)
                                    gz_z = self._interp_precomp(self.corr_grid_w, i1x, i1y, i1z, t1x, t1y, t1z)

                                    i2x, i2y, i2z, t2x, t2y, t2z = self._precompute_grid_idx(zc, yc, xc, gmin, ginv, gn)
                                    gx_x = self._interp_precomp(self.corr_grid_w, i2x, i2y, i2z, t2x, t2y, t2z)
                                    gx_y = self._interp_precomp(self.corr_grid_v, i2x, i2y, i2z, t2x, t2y, t2z)
                                    gx_z = self._interp_precomp(self.corr_grid_u, i2x, i2y, i2z, t2x, t2y, t2z)

                                    i3x, i3y, i3z, t3x, t3y, t3z = self._precompute_grid_idx(xc, zc, yc, gmin, ginv, gn)
                                    gy_x = self._interp_precomp(self.corr_grid_u, i3x, i3y, i3z, t3x, t3y, t3z)
                                    gy_y = self._interp_precomp(self.corr_grid_w, i3x, i3y, i3z, t3x, t3y, t3z)
                                    gy_z = self._interp_precomp(self.corr_grid_v, i3x, i3y, i3z, t3x, t3y, t3z)

                                    common = Fz_inv * eta_r * cL
                                    sx = Fx * common
                                    sy = Fy * common
                                    sz = Fz * common

                                    v_x += gx_x * sx + gy_x * sy + gz_x * sz
                                    v_y += gx_y * sx + gy_y * sy + gz_y * sz
                                    v_z += gx_z * sx + gy_z * sy + gz_z * sz

                    node_idx = self.nodes.next[node_idx]
                    continue

                s = self.nodes.size(node_idx)
                center = self.nodes.get_center(node_idx)
                dx = pos_i[0] - center[0]
                dy = pos_i[1] - center[1]
                dz = pos_i[2] - center[2]

                if self.periodic[None] == 1:
                    L_val = self.L[None]
                    if ti.abs(dx) > L_val * 0.5:
                        dx -= ti.math.sign(dx) * L_val
                    if ti.abs(dy) > L_val * 0.5:
                        dy -= ti.math.sign(dy) * L_val
                    if ti.abs(dz) > L_val * 0.5:
                        dz -= ti.math.sign(dz) * L_val

                D_sq = dx * dx + dy * dy + dz * dz

                if s * s < theta_sq * D_sq and D_sq > 0.0:
                    r = ti.sqrt(D_sq)
                    inv_r = 1.0 / r
                    inv_r2 = inv_r * inv_r

                    Fx = self.nodes.force_sum_x[node_idx]
                    Fy = self.nodes.force_sum_y[node_idx]
                    Fz = self.nodes.force_sum_z[node_idx]
                    dot = dx * Fx + dy * Fy + dz * Fz

                    coeff = eta_const * inv_r
                    v_x += coeff * (Fx + dx * inv_r2 * dot)
                    v_y += coeff * (Fy + dy * inv_r2 * dot)
                    v_z += coeff * (Fz + dz * inv_r2 * dot)

                    if self.corr_enabled[None] == 1:
                        cL = self.corr_L_ratio[None]
                        xc = dx * cL
                        yc = dy * cL
                        zc = dz * cL
                        gmin = self.corr_grid_min[None]
                        ginv = self.corr_grid_inv_dx[None]
                        gn = self.corr_grid_n[None]
                        Fz_inv = self.corr_Fz_inv[None]
                        eta_r = self.corr_eta_ratio[None]

                        i1x, i1y, i1z, t1x, t1y, t1z = self._precompute_grid_idx(xc, yc, zc, gmin, ginv, gn)
                        gz_x = self._interp_precomp(self.corr_grid_u, i1x, i1y, i1z, t1x, t1y, t1z)
                        gz_y = self._interp_precomp(self.corr_grid_v, i1x, i1y, i1z, t1x, t1y, t1z)
                        gz_z = self._interp_precomp(self.corr_grid_w, i1x, i1y, i1z, t1x, t1y, t1z)

                        i2x, i2y, i2z, t2x, t2y, t2z = self._precompute_grid_idx(zc, yc, xc, gmin, ginv, gn)
                        gx_x = self._interp_precomp(self.corr_grid_w, i2x, i2y, i2z, t2x, t2y, t2z)
                        gx_y = self._interp_precomp(self.corr_grid_v, i2x, i2y, i2z, t2x, t2y, t2z)
                        gx_z = self._interp_precomp(self.corr_grid_u, i2x, i2y, i2z, t2x, t2y, t2z)

                        i3x, i3y, i3z, t3x, t3y, t3z = self._precompute_grid_idx(xc, zc, yc, gmin, ginv, gn)
                        gy_x = self._interp_precomp(self.corr_grid_u, i3x, i3y, i3z, t3x, t3y, t3z)
                        gy_y = self._interp_precomp(self.corr_grid_w, i3x, i3y, i3z, t3x, t3y, t3z)
                        gy_z = self._interp_precomp(self.corr_grid_v, i3x, i3y, i3z, t3x, t3y, t3z)

                        common = Fz_inv * eta_r * cL
                        sx = Fx * common
                        sy = Fy * common
                        sz = Fz * common

                        v_x += gx_x * sx + gy_x * sy + gz_x * sz
                        v_y += gx_y * sx + gy_y * sy + gz_y * sz
                        v_z += gx_z * sx + gy_z * sy + gz_z * sz

                    # --- Дипольная коррекция: δv = ∇T : D ---
                    r3s = self.nodes.R3_sum[node_idx]
                    cmx = ti.f64(0.0)
                    cmy = ti.f64(0.0)
                    cmz = ti.f64(0.0)
                    if r3s > 0.0:
                        inv_r3s = 1.0 / r3s
                        cmx = self.nodes.R3_cx[node_idx] * inv_r3s
                        cmy = self.nodes.R3_cy[node_idx] * inv_r3s
                        cmz = self.nodes.R3_cz[node_idx] * inv_r3s
                    else:
                        cmx = (self.nodes.min_x[node_idx] + self.nodes.max_x[node_idx]) * 0.5
                        cmy = (self.nodes.min_y[node_idx] + self.nodes.max_y[node_idx]) * 0.5
                        cmz = (self.nodes.min_z[node_idx] + self.nodes.max_z[node_idx]) * 0.5

                    Dxx = self.nodes.fmom_xx[node_idx] - cmx * Fx
                    Dxy = self.nodes.fmom_xy[node_idx] - cmx * Fy
                    Dxz = self.nodes.fmom_xz[node_idx] - cmx * Fz
                    Dyx = self.nodes.fmom_yx[node_idx] - cmy * Fx
                    Dyy = self.nodes.fmom_yy[node_idx] - cmy * Fy
                    Dyz = self.nodes.fmom_yz[node_idx] - cmy * Fz
                    Dzx = self.nodes.fmom_zx[node_idx] - cmz * Fx
                    Dzy = self.nodes.fmom_zy[node_idx] - cmz * Fy
                    Dzz = self.nodes.fmom_zz[node_idx] - cmz * Fz

                    rx = dx; ry = dy; rz = dz

                    ux = Dxx * rx + Dyx * ry + Dzx * rz
                    uy = Dxy * rx + Dyy * ry + Dzy * rz
                    uz = Dxz * rx + Dyz * ry + Dzz * rz

                    wx = Dxx * rx + Dxy * ry + Dxz * rz
                    wy = Dyx * rx + Dyy * ry + Dyz * rz
                    wz = Dzx * rx + Dzy * ry + Dzz * rz

                    s_tr = Dxx + Dyy + Dzz
                    q = rx * wx + ry * wy + rz * wz

                    inv_r3 = inv_r * inv_r2
                    inv_r5 = inv_r3 * inv_r2

                    v_x += eta_const * (inv_r3 * (ux - wx - s_tr * rx) + 3.0 * rx * q * inv_r5)
                    v_y += eta_const * (inv_r3 * (uy - wy - s_tr * ry) + 3.0 * ry * q * inv_r5)
                    v_z += eta_const * (inv_r3 * (uz - wz - s_tr * rz) + 3.0 * rz * q * inv_r5)

                    node_idx = self.nodes.next[node_idx]
                else:
                    node_idx = self.nodes.first_child[node_idx]

            vx[i] = v_x
            vy[i] = v_y
            vz[i] = v_z

    def compute_total_velocity(self, forces: np.ndarray, eta_const: float,
                               stokes_factor: float) -> np.ndarray:
        """Вычисление полной скорости (миграция + конвекция) за один обход дерева."""
        n = self.num_particles[None]
        forces_c = np.ascontiguousarray(forces[:n], dtype=np.float64)
        vx = self._out_vx[:n]
        vy = self._out_vy[:n]
        vz = self._out_vz[:n]
        vx[:] = 0.0
        vy[:] = 0.0
        vz[:] = 0.0
        self._propagate_forces_from_fields()
        self._compute_total_velocity_kernel(forces_c, vx, vy, vz, eta_const, stokes_factor)
        return np.stack([vx, vy, vz], axis=1)

    def compute_forces_and_total_velocity(self, m_const: float, eta_const: float,
                                          stokes_factor: float) -> tuple:
        """Расчёт сил + полной скорости за минимальное число обходов."""
        forces = self.compute_forces(m_const)
        total_velocity = self.compute_total_velocity(forces, eta_const, stokes_factor)
        return forces, total_velocity

    # =================================================================
    # Диагностика
    # =================================================================

    def get_tree_stats(self) -> dict:
        """Возвращает статистику дерева."""
        nc = self.node_count[None]
        pc = self.parent_count[None]
        np_ = self.num_particles[None]
        return {
            'node_count': nc,
            'max_nodes': self.max_nodes,
            'node_utilization': nc / self.max_nodes if self.max_nodes > 0 else 0.0,
            'parent_count': pc,
            'num_particles': np_,
            'leaf_slot_count': self.leaf_slot_count[None],
        }
