from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pbm.volume_grid import VolumeGrid


class DEMExtractedKernel:
    """Извлечение ядра столкновений из DEM-симуляции.

    K[i,j] = collision_count[i,j] / (T * n_avg[i] * n_avg[j] * dV[i] * dV[j])
    """

    def __init__(self, grid: VolumeGrid, domain_volume: float = 1.0) -> None:
        self.grid = grid
        self.domain_volume = domain_volume
        n = grid.n_bins

        self._collision_count = np.zeros((n, n), dtype=np.float64)
        self._conc_time_sum = np.zeros(n, dtype=np.float64)
        self._total_time = 0.0

    def record_collision(self, r_i: float, r_j: float) -> None:
        v_i = (4.0 / 3.0) * np.pi * r_i**3
        v_j = (4.0 / 3.0) * np.pi * r_j**3
        b_i = self.grid.bin_index(v_i)
        b_j = self.grid.bin_index(v_j)
        self._collision_count[b_i, b_j] += 1.0
        self._collision_count[b_j, b_i] += 1.0

    def update_concentrations(self, radii: NDArray[np.float64], dt: float) -> None:
        volumes = (4.0 / 3.0) * np.pi * radii**3
        bins = self.grid.bin_indices(volumes)
        np.add.at(self._conc_time_sum, bins, dt)
        self._total_time += dt

    def finalize(self) -> NDArray[np.float64]:
        if self._total_time <= 0:
            return np.zeros((self.grid.n_bins, self.grid.n_bins))

        conc_avg = self._conc_time_sum / self._total_time
        dV = self.grid.widths
        n = self.grid.n_bins

        K = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                denom = self._total_time * conc_avg[i] * conc_avg[j] * dV[i] * dV[j]
                if denom > 0:
                    K[i, j] = self._collision_count[i, j] / denom
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
