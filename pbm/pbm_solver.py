from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from pbm.redistribution import cell_average
from pbm.volume_grid import VolumeGrid


class PBMSolver:
    """Солвер уравнений популяционного баланса (агрегация).

    Уравнение (абсолютные числа N в общем объёме V_d):
        dN_i/dt = (1/V_d) * [ 0.5 * Σ_{j,k: j+k→i} Q_jk N_j N_k
                              − N_i * Σ_j Q_ij N_j ]

    Поддерживаются два метода перераспределения новорождённых:
      - "fixed_pivot" (Kumar–Ramkrishna 1996): линейная интерполяция между
        двумя ближайшими центрами. Сохраняет число и объём.
      - "cell_average" (Kumar 2006): два прохода — собрать (B, M) по бинам,
        затем перераспределить с учётом среднего объёма.
    """

    def __init__(
        self,
        grid: VolumeGrid,
        kernel_matrix: NDArray[np.float64],
        method: str = "cell_average",
        integrator: str = "BDF",
        domain_volume: float = 1.0,
    ) -> None:
        if method not in ("cell_average", "fixed_pivot"):
            raise ValueError(f"Неизвестный method: {method}")
        if domain_volume <= 0:
            raise ValueError("domain_volume должен быть > 0")
        self.grid = grid
        self.Q = np.asarray(kernel_matrix, dtype=np.float64)
        self.method = method
        self.integrator = integrator
        self.domain_volume = float(domain_volume)

        # Предвычисляемые таблицы для векторизованного RHS
        self._pair_target_bin: NDArray[np.intp] | None = None  # (n²,) индекс bin для v_j+v_k
        self._pair_v_new: NDArray[np.float64] | None = None  # (n²,) сами v_j+v_k
        self._fp_left_bin: NDArray[np.intp] | None = None  # fixed_pivot: левый bin
        self._fp_right_bin: NDArray[np.intp] | None = None  # fixed_pivot: правый bin
        self._fp_w_left: NDArray[np.float64] | None = None  # fixed_pivot: вес левого
        self._fp_w_right: NDArray[np.float64] | None = None  # fixed_pivot: вес правого
        self._precompute_tables()

    def _precompute_tables(self) -> None:
        """Предвычисляет таблицы перераспределения (зависят только от сетки)."""
        x = self.grid.centers
        edges = self.grid.edges
        n = self.grid.n_bins

        v_new = (x[:, None] + x[None, :]).ravel()  # (n²,)
        # Индекс bin для v_new
        target = np.searchsorted(edges, v_new, side="right").astype(np.intp) - 1
        np.clip(target, 0, n - 1, out=target)
        self._pair_v_new = v_new
        self._pair_target_bin = target

        if self.method == "fixed_pivot":
            # Линейная интерполяция между centers[i] и centers[i+1]
            # i_left определяется как searchsorted(centers, v_new) - 1
            i_left = np.searchsorted(x, v_new).astype(np.intp) - 1
            # Граничные случаи: v_new <= x[0] → веса (0, 1.0) и (0, 0.0)
            #                   v_new >= x[-1] → веса (n-1, 1.0) и (n-1, 0.0)
            below = v_new <= x[0]
            above = v_new >= x[-1]
            i_left = np.clip(i_left, 0, n - 2)
            i_right = i_left + 1

            delta = x[i_right] - x[i_left]
            w_left = (x[i_right] - v_new) / delta
            w_right = 1.0 - w_left

            # Принудительно для крайних случаев
            w_left[below] = 1.0
            w_right[below] = 0.0
            i_left[below] = 0
            i_right[below] = 0

            w_left[above] = 1.0
            w_right[above] = 0.0
            i_left[above] = n - 1
            i_right[above] = n - 1

            self._fp_left_bin = i_left
            self._fp_right_bin = i_right
            self._fp_w_left = w_left
            self._fp_w_right = w_right

    def update_kernel(self, kernel_matrix: NDArray[np.float64]) -> None:
        self.Q = np.asarray(kernel_matrix, dtype=np.float64)

    def rhs(self, t: float, N: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.method == "cell_average":
            return self._rhs_cell_average(N)
        return self._rhs_fixed_pivot(N)

    def _pair_rate(self, N: NDArray[np.float64]) -> NDArray[np.float64]:
        """0.5 * Q ⊗ (N N^T) / V_d — скорости рождений по парам (j,k), форма (n²,).

        Множитель 0.5 компенсирует двойной счёт при полном цикле j,k
        (включая диагональ для самоагрегации i+i).
        """
        # outer product через broadcasting
        rate_mat = 0.5 * self.Q * (N[:, None] * N[None, :])
        rate_mat /= self.domain_volume
        return rate_mat.ravel()

    def _rhs_fixed_pivot(self, N: NDArray[np.float64]) -> NDArray[np.float64]:
        n = self.grid.n_bins
        rates = self._pair_rate(N)  # (n²,)

        birth = np.zeros(n)
        np.add.at(birth, self._fp_left_bin, rates * self._fp_w_left)
        np.add.at(birth, self._fp_right_bin, rates * self._fp_w_right)

        # Death = N_i * Σ_j Q_ij N_j / V_d
        death = N * (self.Q @ N) / self.domain_volume
        return birth - death

    def _rhs_cell_average(self, N: NDArray[np.float64]) -> NDArray[np.float64]:
        n = self.grid.n_bins
        rates = self._pair_rate(N)  # (n²,)

        B = np.zeros(n)
        M = np.zeros(n)
        np.add.at(B, self._pair_target_bin, rates)
        np.add.at(M, self._pair_target_bin, rates * self._pair_v_new)

        birth = cell_average(B, M, self.grid)
        death = N * (self.Q @ N) / self.domain_volume
        return birth - death

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
