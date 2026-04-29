from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pbm.volume_grid import VolumeGrid
from pbm.kernels.dem_extracted_kernel import DEMExtractedKernel

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
        if np.any(K_dem > 0):
            self.pbm_solver.update_kernel(K_dem)
            self._dem_kernel.reset()

        result = self.pbm_solver.solve(
            self._pbm_N,
            (self._pbm_time, current_time),
            rtol=1e-5,
            atol=1e-5,
        )

        self._pbm_N = result["N"][-1]
        self._pbm_time = current_time
        self._last_sync_time = current_time

        dem_N = self.grid.histogram(radii)
        self.history_t.append(current_time)
        self.history_dem_N.append(dem_N)
        self.history_pbm_N.append(self._pbm_N.copy())

        total_dem = np.sum(dem_N)
        total_pbm = np.sum(self._pbm_N)
        logger.info(
            "t=%.2f  DEM particles=%d  PBM total=%.0f",
            current_time, len(radii), total_pbm,
        )

    def get_pbm_distribution(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self._pbm_N is None:
            return self.grid.centers, np.zeros(self.grid.n_bins)
        return self.grid.centers, self._pbm_N.copy()

    def get_dem_kernel(self) -> NDArray[np.float64]:
        return self._dem_kernel.finalize()
