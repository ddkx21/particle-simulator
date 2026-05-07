"""Тесты pbm/coupling.py — DEMPBMCoupling."""

from __future__ import annotations

import numpy as np

from pbm import PBMSolver, VolumeGrid
from pbm.coupling import DEMPBMCoupling
from pbm.kernels import AnalyticalElectrostaticKernel


def _make(coupling_interval: float = 1.0):
    g = VolumeGrid.from_radii_range(2.5e-6, 7.5e-6 * 5, 30)
    Q = AnalyticalElectrostaticKernel().build_matrix(g)
    solver = PBMSolver(g, Q, method="cell_average", domain_volume=1e-12)
    coupling = DEMPBMCoupling(g, solver, domain_volume=1e-12, coupling_interval=coupling_interval)
    return g, solver, coupling


# --------------------------------------------------------------------------
# Инициализация и базовое состояние
# --------------------------------------------------------------------------
class TestCouplingInit:
    def test_initialize_from_dem_sets_state(self) -> None:
        g, _, coupling = _make()
        rng = np.random.default_rng(0)
        radii = rng.uniform(2.5e-6, 7.5e-6, 100)
        coupling.initialize_from_dem(radii, t0=2.0)
        assert coupling._pbm_time == 2.0
        assert coupling._last_sync_time == 2.0
        assert coupling._pbm_N is not None
        assert coupling._pbm_N.sum() == 100

    def test_get_pbm_distribution_before_init(self) -> None:
        g, _, coupling = _make()
        centers, N = coupling.get_pbm_distribution()
        assert N.shape == (g.n_bins,)
        np.testing.assert_array_equal(N, np.zeros(g.n_bins))


# --------------------------------------------------------------------------
# Шаги и расписание синхронизации
# --------------------------------------------------------------------------
class TestCouplingStep:
    def test_step_below_interval_no_sync(self) -> None:
        g, _, coupling = _make(coupling_interval=10.0)
        rng = np.random.default_rng(0)
        radii = rng.uniform(2.5e-6, 7.5e-6, 100)
        coupling.initialize_from_dem(radii)
        for _ in range(25):
            coupling.step(0.04, radii, 0.04)
        assert len(coupling.history_t) == 0

    def test_step_triggers_sync(self) -> None:
        g, _, coupling = _make(coupling_interval=0.5)
        rng = np.random.default_rng(0)
        radii = rng.uniform(2.5e-6, 7.5e-6, 100)
        coupling.initialize_from_dem(radii)
        t = 0.0
        for _ in range(30):
            t += 0.04
            coupling.step(t, radii, 0.04)
        assert len(coupling.history_t) >= 1


# --------------------------------------------------------------------------
# Запись столкновений и sync edge-cases
# --------------------------------------------------------------------------
class TestCouplingCollisions:
    def test_on_collision_records(self) -> None:
        g, _, coupling = _make()
        radii = np.array([3e-6, 4e-6, 5e-6])
        pairs = np.array([[0, 1], [1, 2]])
        coupling.on_collision(pairs, radii)
        K = coupling.get_dem_kernel()
        np.testing.assert_array_equal(K, np.zeros_like(K))

    def test_sync_with_zero_dt_pbm_returns(self) -> None:
        g, _, coupling = _make(coupling_interval=0.1)
        rng = np.random.default_rng(0)
        radii = rng.uniform(2.5e-6, 7.5e-6, 50)
        coupling.initialize_from_dem(radii, t0=5.0)
        coupling._sync(5.0, radii)
        assert len(coupling.history_t) == 0

    def test_sync_with_collisions_runs(self) -> None:
        g, solver, coupling = _make(coupling_interval=0.1)
        rng = np.random.default_rng(0)
        radii = rng.uniform(2.5e-6, 7.5e-6, 80)
        coupling.initialize_from_dem(radii, t0=0.0)
        pairs = np.array([[i, i + 1] for i in range(0, 20, 2)])
        coupling.on_collision(pairs, radii)
        t = 0.0
        for _ in range(5):
            t += 0.05
            coupling.step(t, radii, 0.05)
        assert len(coupling.history_t) >= 1
        centers, N = coupling.get_pbm_distribution()
        assert N.shape == (g.n_bins,)
