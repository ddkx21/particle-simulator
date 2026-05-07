from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pbm.volume_grid import VolumeGrid


class DEMExtractedKernel:
    """Извлечение ядра столкновений Q(v_i, v_j) из DEM-симуляции.

    Размерность Q — м³/с (как у аналитического ядра).

    Вывод:
        n_coll[i,j] = Q_ij * <n_i> * <n_j> * V_d * T          (i ≠ j)
        n_coll[i,i] = 0.5 * Q_ii * <n_i>^2 * V_d * T          (i = j)
    откуда
        Q_ij = n_coll[i,j] / (T * <n_i> * <n_j> * V_d)        (i ≠ j)
        Q_ii = 2 * n_coll[i,i] / (T * <n_i>^2 * V_d)          (i = j)
    где n_avg хранится как T*<n> в self._conc_time_sum.
    """

    def __init__(self, grid: VolumeGrid, domain_volume: float = 1.0) -> None:
        self.grid = grid
        self.domain_volume = domain_volume
        n = grid.n_bins

        # Каждое физическое столкновение учитывается ровно один раз
        # в ячейке [min(b_i,b_j), max(b_i,b_j)] (верхний треугольник).
        self._collision_count = np.zeros((n, n), dtype=np.float64)
        self._conc_time_sum = np.zeros(n, dtype=np.float64)
        self._total_time = 0.0

    def record_collision(self, r_i: float, r_j: float) -> None:
        v_i = (4.0 / 3.0) * np.pi * r_i**3
        v_j = (4.0 / 3.0) * np.pi * r_j**3
        b_i = self.grid.bin_index(v_i)
        b_j = self.grid.bin_index(v_j)
        a, b = (b_i, b_j) if b_i <= b_j else (b_j, b_i)
        self._collision_count[a, b] += 1.0

    def update_concentrations(self, radii: NDArray[np.float64], dt: float) -> None:
        volumes = (4.0 / 3.0) * np.pi * radii**3
        bins = self.grid.bin_indices(volumes)
        # Интегрируем плотность n_i(t) = (число в бине i) / V_d по времени
        np.add.at(self._conc_time_sum, bins, dt / self.domain_volume)
        self._total_time += dt

    def finalize(self) -> NDArray[np.float64]:
        """Возвращает симметричную матрицу Q[i,j] в м³/с."""
        n = self.grid.n_bins
        if self._total_time <= 0:
            return np.zeros((n, n))

        n_avg = self._conc_time_sum / self._total_time  # 1/м³
        T = self._total_time
        V_d = self.domain_volume

        K = np.zeros((n, n), dtype=np.float64)
        # Внедиагональные: Q_ij = count / (T * n_i * n_j * V_d)
        # Диагональные: Q_ii = 2 * count / (T * n_i^2 * V_d)
        for i in range(n):
            ni = n_avg[i]
            if ni <= 0:
                continue
            for j in range(i, n):
                nj = n_avg[j]
                if nj <= 0:
                    continue
                if i == j:
                    denom = T * ni * ni * V_d
                    if denom > 0:
                        K[i, i] = 2.0 * self._collision_count[i, i] / denom
                else:
                    denom = T * ni * nj * V_d
                    if denom > 0:
                        val = self._collision_count[i, j] / denom
                        K[i, j] = val
                        K[j, i] = val
        return K

    def reset(self) -> None:
        self._collision_count[:] = 0.0
        self._conc_time_sum[:] = 0.0
        self._total_time = 0.0

    @staticmethod
    def load_from_file(path: str, grid: VolumeGrid) -> NDArray[np.float64]:
        """Загрузка предвычисленного ядра из .npz файла.

        Ожидает ключи 'volumes' и 'freq' в файле.
        Интерполирует на сетку grid при помощи RectBivariateSpline.
        """
        from scipy.interpolate import RectBivariateSpline

        data = np.load(path)
        volumes = data["volumes"]
        freq = data["freq"]

        spline = RectBivariateSpline(volumes, volumes, freq, kx=3, ky=3)
        K = spline(grid.centers, grid.centers)
        np.maximum(K, 0.0, out=K)
        return K
