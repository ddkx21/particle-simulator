"""
TreeDropletForceCalculator: калькулятор сил на основе октодерева.

Интерфейс совместим с DirectDropletForceCalculator.
Алгоритм Барнса-Хатта O(N log N).
"""

from __future__ import annotations

import numpy as np
import taichi as ti

from force_calculator.force_calculator_base import ForceCalculator
from .flat_tree import FlatOctree

@ti.data_oriented
class TreeDropletForceCalculator(ForceCalculator):
    """
    Калькулятор дипольных сил и Стокслет-скоростей через октодерево.

    Параметры:
    - theta: параметр Барнса-Хатта
    - mpl: max particles per leaf
    - eps_oil, eta_oil, eta_water, E: физические константы
    - L: размер домена
    - periodic: периодические граничные условия
    """

    def __init__(self,
                 num_particles: int = 10000,
                 theta: float = 0.5,
                 mpl: int = 1,
                 eps_oil: float = 2.85,
                 eta_oil: float = 0.065,
                 eta_water: float = 0.001,
                 E: float = 3e5,
                 L: float = 1.0,
                 periodic: bool = False,
                 correction_grid_resolution: int = 0):

        self.eps0 = 8.85418781762039e-12
        self.eps_oil = eps_oil
        self.eta_oil = eta_oil
        self.eta_water = eta_water
        self.E = E
        self.L = L
        self.periodic = periodic
        self.boundary_mode = "periodic" if periodic else "open"

        # Константа дипольной силы: 12*pi*eps0*eps_oil*E^2
        self.m_const = 12.0 * np.pi * self.eps0 * eps_oil * E**2

        # Префактор тензора Озеена: 1/(8*pi*eta_oil)
        self.eta_const = 1.0 / (8.0 * np.pi * eta_oil)

        self.num_particles = num_particles
        self.theta = theta
        self.mpl = mpl
        self.correction_grid_resolution = correction_grid_resolution

        self.octree = FlatOctree(theta=theta, mpl=mpl,
                                 num_particles=num_particles,
                                 correction_grid_resolution=correction_grid_resolution)

        # Флаг устаревшего дерева — дерево перестраивается перед каждым использованием
        self._tree_is_stale = True
        # Флаг: силы уже посчитаны и доступны в полях Taichi (без лишнего копирования)
        self._forces_in_fields = False

        # Сохраняем ссылку на поправку для перезагрузки при пересоздании дерева
        self._correction = None
        self._correction_L_sim = None

    def update_params(self, theta: float, mpl: int) -> None:
        """Обновление параметров дерева без перевыделения памяти."""
        self.theta = theta
        self.mpl = mpl
        self.octree.update_params(theta, mpl)
        self._tree_is_stale = True

    @staticmethod
    def _to_contiguous(arr: np.ndarray, dtype=np.float64) -> np.ndarray:
        """Приведение массива к C-непрерывному формату с заданным типом."""
        return np.ascontiguousarray(arr, dtype=dtype)

    def _ensure_octree_capacity(self, n: int) -> None:
        """Пересоздание октодерева, если число частиц превышает выделенную ёмкость."""
        if n > self.num_particles:
            self.octree = FlatOctree(theta=self.theta, mpl=self.mpl,
                                     num_particles=n,
                                     correction_grid_resolution=self.correction_grid_resolution)
            self.num_particles = n
            # Перезагрузка поправки в новое дерево
            if self._correction is not None:
                self.octree.load_periodic_correction(self._correction, self._correction_L_sim)

    def load_periodic_correction(self, correction, L_sim: float):
        """Загрузка данных периодической поправки COMSOL."""
        self._correction = correction
        self._correction_L_sim = L_sim
        self.octree.load_periodic_correction(correction, L_sim)

    def calculate(self,
                  positions: np.ndarray,   # форма (N, 3), float64
                  radii: np.ndarray,       # форма (N,), float64
                  *,
                  L: float | None = None,
                  periodic: bool | None = None,
                  ) -> np.ndarray:         # форма (N, 3), float64
        """
        Вычислить дипольные силы через октодерево.

        Возвращает: np.ndarray (N, 3) с силами.
        """
        positions = self._to_contiguous(positions)
        radii = self._to_contiguous(radii)
        n = positions.shape[0]

        use_L = L if L is not None else self.L
        use_periodic = periodic if periodic is not None else self.periodic

        self._ensure_octree_capacity(n)
        self.octree.build(positions, radii, use_L, use_periodic)
        self._tree_is_stale = False
        self._forces_in_fields = True  # Ядро записало силы в поля Taichi

        forces = self.octree.compute_forces(self.m_const)

        return forces

    def calculate_convection(self,
                             positions: np.ndarray,   # форма (N, 3), float64
                             radii: np.ndarray,       # форма (N,), float64
                             forces: np.ndarray,      # форма (N, 3), float64
                             *,
                             L: float | None = None,
                             periodic: bool | None = None,
                             ) -> np.ndarray:         # форма (N, 3), float64
        """
        Вычислить Стокслет-скорости (конвекцию) через октодерево.
        Дерево перестраивается если помечено как устаревшее.

        Возвращает: np.ndarray (N, 3) со скоростями.
        """
        forces = self._to_contiguous(forces)

        use_L = L if L is not None else self.L
        use_periodic = periodic if periodic is not None else self.periodic

        # Перестраиваем дерево, если оно устарело
        if self._tree_is_stale:
            positions = self._to_contiguous(positions)
            radii = self._to_contiguous(radii)
            n = positions.shape[0]
            self._ensure_octree_capacity(n)
            self.octree.build(positions, radii, use_L, use_periodic)
            self._tree_is_stale = False

        # Используем силы из полей Taichi, если дерево построено на этом же шаге
        if self._forces_in_fields:
            velocities = self.octree.compute_stokeslet_from_fields(forces, self.eta_const)
        else:
            velocities = self.octree.compute_stokeslet(forces, self.eta_const)

        # Помечаем дерево как устаревшее (на следующем вызове нужно свежее дерево)
        self._tree_is_stale = True
        self._forces_in_fields = False

        return velocities

    def calculate_forces_and_convection(self,
                                        positions: np.ndarray,
                                        radii: np.ndarray,
                                        *,
                                        L: float | None = None,
                                        periodic: bool | None = None,
                                        ) -> tuple:
        """
        Вычислить силы и конвекцию в одном вызове.

        Оптимизация: ядро сил записывает в поля Taichi, ядро стокслета
        читает оттуда напрямую (без лишнего копирования).

        Возвращает: (forces (N,3), velocities (N,3))
        """
        positions = self._to_contiguous(positions)
        radii = self._to_contiguous(radii)
        n = positions.shape[0]

        use_L = L if L is not None else self.L
        use_periodic = periodic if periodic is not None else self.periodic

        self._ensure_octree_capacity(n)
        self.octree.build(positions, radii, use_L, use_periodic)
        self._tree_is_stale = True

        forces, velocities = self.octree.compute_forces_and_stokeslet(
            self.m_const, self.eta_const
        )

        return forces, velocities
