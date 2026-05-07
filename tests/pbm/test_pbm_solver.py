"""Тесты pbm/pbm_solver.py — PBMSolver (cell_average и fixed_pivot)."""
from __future__ import annotations

import numpy as np
import pytest

from pbm import VolumeGrid, PBMSolver
from pbm.kernels import AnalyticalElectrostaticKernel


# --------------------------------------------------------------------------
# Валидация конструктора
# --------------------------------------------------------------------------
class TestPBMSolverValidation:
    def test_invalid_method_raises(self) -> None:
        g = VolumeGrid(1.0, 10.0, 5)
        Q = np.zeros((5, 5))
        with pytest.raises(ValueError):
            PBMSolver(g, Q, method="bogus")

    def test_invalid_domain_volume_raises(self) -> None:
        g = VolumeGrid(1.0, 10.0, 5)
        Q = np.zeros((5, 5))
        with pytest.raises(ValueError):
            PBMSolver(g, Q, domain_volume=-1.0)


# --------------------------------------------------------------------------
# cell_average — физические инварианты
# --------------------------------------------------------------------------
class TestCellAverageMethod:
    def _setup(self, n_bins: int = 30):
        g = VolumeGrid.from_radii_range(2.5e-6, 7.5e-6 * 5, n_bins)
        rng = np.random.default_rng(0)
        radii = rng.uniform(2.5e-6, 7.5e-6, 100_000)
        N0 = g.histogram(radii)
        Q = AnalyticalElectrostaticKernel().build_matrix(g)
        solver = PBMSolver(g, Q, method="cell_average", domain_volume=1.0)
        return g, N0, solver

    def test_volume_conservation(self) -> None:
        g, N0, solver = self._setup()
        res = solver.solve(N0, (0, 50.0), t_eval=np.linspace(0, 50, 6),
                           rtol=1e-6, atol=1e-3)
        rel_err = (res["total_volume"][-1] - res["total_volume"][0]) / res["total_volume"][0]
        assert abs(rel_err) < 1e-6

    def test_count_decreases(self) -> None:
        g, N0, solver = self._setup()
        res = solver.solve(N0, (0, 50.0), t_eval=np.linspace(0, 50, 6),
                           rtol=1e-6, atol=1e-3)
        diffs = np.diff(res["total_count"])
        assert np.all(diffs <= 1e-6)
        assert res["total_count"][-1] < res["total_count"][0]


# --------------------------------------------------------------------------
# fixed_pivot — физические инварианты
# --------------------------------------------------------------------------
class TestFixedPivotMethod:
    def test_volume_conservation(self) -> None:
        g = VolumeGrid.from_radii_range(2.5e-6, 7.5e-6 * 5, 30)
        rng = np.random.default_rng(0)
        radii = rng.uniform(2.5e-6, 7.5e-6, 100_000)
        N0 = g.histogram(radii)
        Q = AnalyticalElectrostaticKernel().build_matrix(g)
        solver = PBMSolver(g, Q, method="fixed_pivot", domain_volume=1.0)
        res = solver.solve(N0, (0, 50.0), t_eval=np.linspace(0, 50, 6),
                           rtol=1e-6, atol=1e-3)
        rel_err = (res["total_volume"][-1] - res["total_volume"][0]) / res["total_volume"][0]
        assert abs(rel_err) < 1e-6

    def test_simple_aggregation_runs(self) -> None:
        g = VolumeGrid(1.0, 100.0, 12)
        Q = np.full((g.n_bins, g.n_bins), 1e-3)
        solver = PBMSolver(g, Q, method="fixed_pivot", domain_volume=1.0)
        N0 = np.zeros(g.n_bins)
        N0[0] = 100.0
        result = solver.solve(N0, (0.0, 0.5))
        assert result["total_count"][-1] <= result["total_count"][0] + 1e-6


# --------------------------------------------------------------------------
# update_kernel и обработка ошибок интегратора
# --------------------------------------------------------------------------
class TestSolverRuntime:
    def test_update_kernel_replaces_Q(self) -> None:
        g = VolumeGrid(1.0, 100.0, 8)
        solver = PBMSolver(g, np.zeros((g.n_bins, g.n_bins)),
                           method="fixed_pivot", domain_volume=1.0)
        Q_new = np.ones((g.n_bins, g.n_bins))
        solver.update_kernel(Q_new)
        np.testing.assert_array_equal(solver.Q, Q_new)

    def test_solve_failure_raises_runtime_error(self) -> None:
        """Жёсткая система + микроскопические допуски → solve_ivp падает."""
        g = VolumeGrid(1.0, 100.0, 8)
        Q = np.full((g.n_bins, g.n_bins), 1e30)
        solver = PBMSolver(g, Q, method="cell_average", domain_volume=1e-30,
                           integrator="LSODA")
        N0 = np.full(g.n_bins, 1e10)
        with pytest.raises(RuntimeError, match="PBM"):
            solver.solve(N0, (0.0, 1e6), rtol=1e-15, atol=1e-15)
