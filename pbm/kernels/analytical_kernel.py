from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pbm.volume_grid import VolumeGrid

class AnalyticalElectrostaticKernel:
    """Аналитическое ядро столкновений для электростатической коалесценции.

    Q(v1, v2) = [4/sqrt(3)] * (eps0 * eps_oil * E^2) / eta_oil
                * (v1^(2/3) * v2^(2/3)) / (v1^(1/3) + v2^(1/3))
    """

    def __init__(
        self,
        eps0: float = 8.85e-12,
        eps_oil: float = 2.85,
        E: float = 3e5,
        eta_oil: float = 0.065,
    ) -> None:
        self.prefactor = (4.0 / np.sqrt(3.0)) * (eps0 * eps_oil * E**2) / eta_oil

    def evaluate(self, v1: float | NDArray, v2: float | NDArray) -> float | NDArray:
        v1_23 = np.float_power(v1, 2.0 / 3.0)
        v2_23 = np.float_power(v2, 2.0 / 3.0)
        v1_13 = np.float_power(v1, 1.0 / 3.0)
        v2_13 = np.float_power(v2, 1.0 / 3.0)
        return self.prefactor * (v1_23 * v2_23) / (v1_13 + v2_13)

    def build_matrix(self, grid: VolumeGrid) -> NDArray[np.float64]:
        c = grid.centers
        v1, v2 = np.meshgrid(c, c, indexing="ij")
        return self.evaluate(v1, v2)
