from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from pbm.volume_grid import VolumeGrid
from pbm.redistribution import fixed_pivot, cell_average


class PBMSolver:
    """Солвер для уравнений популяционного баланса (PBM) методом агрегации.

    dN/dt = Birth - Death
    """

    def __init__(
        self,
        grid: VolumeGrid,
        kernel_matrix: NDArray[np.float64],
        method: str = "cell_average",
        integrator: str = "BDF",
        scale_factor: float = 1.0,
    ) -> None:
        self.grid = grid
        self.Q = kernel_matrix
        self.method = method
        self.integrator = integrator
        self.scale_factor = scale_factor

        self._birth_map: list[list[tuple[int, float]]] | None = None
        if method == "fixed_pivot":
            self._precompute_birth_map()

    def _precompute_birth_map(self) -> None:
        """Предвычисляет маппинг (j, k) → target bins для fixed_pivot."""
        n = self.grid.n_bins
        x = self.grid.centers
        self._birth_map = []
        for j in range(n):
            for k in range(j, n):
                v_new = x[j] + x[k]
                targets = fixed_pivot(v_new, self.grid)
                self._birth_map.append(targets)

    def rhs(self, t: float, N: NDArray[np.float64]) -> NDArray[np.float64]:
        n = self.grid.n_bins
        x = self.grid.centers
        Q = self.Q

        if self.method == "cell_average":
            return self._rhs_cell_average(N, n, x, Q)
        else:
            return self._rhs_fixed_pivot(N, n, x, Q)

    def _rhs_fixed_pivot(
        self,
        N: NDArray[np.float64],
        n: int,
        x: NDArray[np.float64],
        Q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        birth = np.zeros(n)
        idx = 0

        for j in range(n):
            for k in range(j, n):
                coeff = 0.5 if j == k else 1.0
                rate = coeff * Q[j, k] * N[j] * N[k]

                if rate > 0 and self._birth_map is not None:
                    for bin_idx, weight in self._birth_map[idx]:
                        birth[bin_idx] += weight * rate

                idx += 1

        death = N * (Q @ N)
        return (birth - death) * self.scale_factor

    def _rhs_cell_average(
        self,
        N: NDArray[np.float64],
        n: int,
        x: NDArray[np.float64],
        Q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        B = np.zeros(n)
        M = np.zeros(n)
        edges = self.grid.edges

        for j in range(n):
            for k in range(j, n):
                coeff = 0.5 if j == k else 1.0
                rate = coeff * Q[j, k] * N[j] * N[k]
                if rate <= 0:
                    continue

                v_new = x[j] + x[k]
                if v_new >= edges[-1]:
                    i_cell = n - 1
                elif v_new < edges[0]:
                    i_cell = 0
                else:
                    i_cell = int(np.searchsorted(edges, v_new, side="right")) - 1
                    i_cell = min(i_cell, n - 1)

                B[i_cell] += rate
                M[i_cell] += rate * v_new

        birth = cell_average(B, M, self.grid)
        death = N * (Q @ N)
        return (birth - death) * self.scale_factor

    def solve(
        self,
        N0: NDArray[np.float64],
        t_span: tuple[float, float],
        t_eval: NDArray[np.float64] | None = None,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> dict[str, Any]:
        sol = solve_ivp(
            self.rhs,
            t_span,
            N0,
            method=self.integrator,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"PBM интегрирование не выполнено: {sol.message}")

        N_sol = sol.y.T  # (n_times, n_bins)
        x = self.grid.centers

        return {
            "t": sol.t,
            "N": N_sol,
            "total_count": np.sum(N_sol, axis=1),
            "total_volume": N_sol @ x,
        }

    def update_kernel(self, kernel_matrix: NDArray[np.float64]) -> None:
        self.Q = kernel_matrix
        if self.method == "fixed_pivot":
            self._precompute_birth_map()
