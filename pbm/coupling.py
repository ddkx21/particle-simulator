from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pbm.kernels.dem_extracted_kernel import DEMExtractedKernel
from pbm.volume_grid import VolumeGrid

if TYPE_CHECKING:
    from pbm.pbm_solver import PBMSolver

logger = logging.getLogger(__name__)


class DEMPBMCoupling:
    """Односторонняя связка DEM → PBM.

    Собирает статистику столкновений из DEM,
    периодически продвигает PBM на coupling_interval.
    """

    def __init__(
        self,
        grid: VolumeGrid,
        pbm_solver: PBMSolver,
        domain_volume: float,
        coupling_interval: float = 1.0,
    ) -> None:
        self.grid = grid
        self.pbm_solver = pbm_solver
        self.domain_volume = domain_volume
        self.coupling_interval = coupling_interval

        self._dem_kernel = DEMExtractedKernel(grid, domain_volume)
        self._pbm_N: NDArray[np.float64] | None = None
        self._last_sync_time = 0.0
        self._pbm_time = 0.0

        self.history_t: list[float] = []
        self.history_dem_N: list[NDArray[np.float64]] = []
        self.history_pbm_N: list[NDArray[np.float64]] = []

    def initialize_from_dem(self, radii: NDArray[np.float64], t0: float = 0.0) -> None:
        self._pbm_N = self.grid.histogram(radii)
        self._last_sync_time = t0
        self._pbm_time = t0

    def on_collision(
        self,
        collided_pairs: NDArray[np.intp],
        radii: NDArray[np.float64],
    ) -> None:
        for idx1, idx2 in collided_pairs:
            self._dem_kernel.record_collision(radii[idx1], radii[idx2])

    def step(
        self,
        current_time: float,
        radii: NDArray[np.float64],
        dt: float,
    ) -> None:
        self._dem_kernel.update_concentrations(radii, dt)

        if current_time - self._last_sync_time >= self.coupling_interval:
            self._sync(current_time, radii)

    def _sync(self, current_time: float, radii: NDArray[np.float64]) -> None:
        if self._pbm_N is None:
            self.initialize_from_dem(radii, current_time)
            return

        dt_pbm = current_time - self._pbm_time
        if dt_pbm <= 0:
            return

        K_dem = self._dem_kernel.finalize()
        # Q_dem и аналитический Q теперь имеют одинаковую размерность (м³/с),
        # так что прямое сравнение корректно.
        Q_current_max = float(np.max(self.pbm_solver.Q)) if self.pbm_solver.Q is not None else 0.0
        K_dem_max = float(np.max(K_dem)) if K_dem.size else 0.0
        sanity_threshold = max(Q_current_max * 1e2, 1e-18)
        kernel_updated = False
        if np.any(~np.isfinite(K_dem)):
            logger.warning("K_dem содержит NaN/inf — пропускаем обновление ядра")
        elif K_dem_max > sanity_threshold:
            logger.warning(
                "K_dem max=%.3e существенно больше текущего Q max=%.3e — "
                "статистика DEM пока недостаточна, пропускаем обновление ядра",
                K_dem_max,
                Q_current_max,
            )
        elif np.any(K_dem > 0):
            self.pbm_solver.update_kernel(K_dem)
            kernel_updated = True
        # Всегда сбрасываем накопители DEM-кернела, иначе при sanity-fail
        # bias накапливается между неудачными попытками
        self._dem_kernel.reset()

        # atol адаптивно: ~1e-6 от суммарного числа частиц, но не меньше 1e-3
        total = float(np.sum(self._pbm_N))
        atol = max(1e-3, 1e-6 * total) if total > 0 else 1e-3

        try:
            result = self.pbm_solver.solve(
                self._pbm_N,
                (self._pbm_time, current_time),
                rtol=1e-4,
                atol=atol,
            )
        except RuntimeError as exc:
            logger.error("PBM solve_ivp упал: %s — состояние не обновляется", exc)
            self._last_sync_time = current_time
            return

        N_new = result["N"][-1]
        if np.any(~np.isfinite(N_new)):
            logger.error(
                "PBM solve вернул NaN/inf: t=%.3f→%.3f, N0_sum=%.3e, Q_max=%.3e",
                self._pbm_time,
                current_time,
                float(np.sum(self._pbm_N)),
                float(np.max(self.pbm_solver.Q)),
            )
            self._last_sync_time = current_time
            return
        # Небольшие отрицательные значения — численный шум BDF, клиппим к 0
        np.maximum(N_new, 0.0, out=N_new)

        self._pbm_N = N_new
        self._pbm_time = current_time
        self._last_sync_time = current_time

        dem_N = self.grid.histogram(radii)
        self.history_t.append(current_time)
        self.history_dem_N.append(dem_N)
        self.history_pbm_N.append(self._pbm_N.copy())

        total_pbm = np.sum(self._pbm_N)
        logger.info(
            "t=%.2f  DEM particles=%d  PBM total=%.1f  Q_max=%.3e  K_dem_max=%.3e  upd=%s",
            current_time,
            len(radii),
            total_pbm,
            Q_current_max,
            K_dem_max,
            kernel_updated,
        )

    def get_pbm_distribution(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self._pbm_N is None:
            return self.grid.centers, np.zeros(self.grid.n_bins)
        return self.grid.centers, self._pbm_N.copy()

    def get_dem_kernel(self) -> NDArray[np.float64]:
        return self._dem_kernel.finalize()
