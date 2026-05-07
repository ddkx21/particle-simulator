"""Тесты solver/euler_droplet_solver.py.

EulerDropletSolver — интегратор Эйлера для движения капель.
Используются минимальные стабы для force_calculator/post_processor,
чтобы тестировать сам решатель в изоляции.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dem.particle_state import DropletState
from dem.solution import DropletSolution
from dem.solver import EulerDropletSolver


# --------------------------------------------------------------------------
# Стабы окружения
# --------------------------------------------------------------------------
@dataclass
class StubForceCalculator:
    """Force calculator c постоянной силой (0,0,F) и без конвекции."""

    F: float = 1e-12
    L: float = 1.0
    eta_oil: float = 0.065
    eta_water: float = 0.001
    boundary_mode: str = "open"

    def calculate_forces_and_convection(self, positions, radii):
        n = len(radii)
        forces = np.zeros((n, 3))
        forces[:, 2] = self.F
        convection = np.zeros((n, 3))
        return forces, convection


class StubPostProcessor:
    def __init__(self) -> None:
        self.stop_simulation = False
        self.live_calls = 0

    def update_solution(self, new_solution) -> None:
        self.solution = new_solution

    def update_live_plot(self) -> None:
        self.live_calls += 1

    def live_plot(self) -> None:
        self.live_calls += 1

    def finalize_live_plot(self) -> None:
        pass


def _make_solver(F: float = 1e-12, n: int = 2, collision_detector=None) -> EulerDropletSolver:
    positions = np.zeros((n, 3))
    radii = np.full(n, 50e-6)
    state = DropletState(positions, radii, time=0.0)
    sol = DropletSolution(initial_droplet_state=state, length=20)
    fc = StubForceCalculator(F=F)
    pp = StubPostProcessor()
    return EulerDropletSolver(
        force_calculator=fc, solution=sol, post_processor=pp, collision_detector=collision_detector
    )


# --------------------------------------------------------------------------
# stokes_factor вычисляется в конструкторе
# --------------------------------------------------------------------------
class TestStokesFactor:
    def test_stokes_factor_positive(self) -> None:
        solver = _make_solver()
        assert solver.stokes_factor > 0


# --------------------------------------------------------------------------
# solve(): движение капель в z под постоянной силой
# --------------------------------------------------------------------------
class TestSolveMotion:
    def test_drops_move_in_z(self) -> None:
        solver = _make_solver(F=1e-12)
        solver.solve(dt=1e-3, total_time=1e-2)
        positions = solver.solution.get_current_positions()
        # Капли должны сдвинуться в +z (сила в +z)
        assert np.all(positions[:, 2] > 0)
        # X и Y не меняются
        np.testing.assert_allclose(positions[:, 0], 0.0)
        np.testing.assert_allclose(positions[:, 1], 0.0)

    def test_time_advances(self) -> None:
        solver = _make_solver()
        solver.solve(dt=1e-3, total_time=5e-3)
        assert solver.solution.get_current_time() > 0

    def test_simulation_time_recorded(self) -> None:
        solver = _make_solver()
        solver.solve(dt=1e-3, total_time=2e-3)
        assert solver.simulation_time >= 0


# --------------------------------------------------------------------------
# Прерывание stop_simulation
# --------------------------------------------------------------------------
class TestStopSimulation:
    def test_solver_stops_when_flag_set(self) -> None:
        solver = _make_solver()
        solver.post_processor.stop_simulation = True
        solver.solve(dt=1e-3, total_time=1.0)
        # Никаких новых шагов после флага — current_step остаётся на начальном
        assert solver.solution.current_step == 0


# --------------------------------------------------------------------------
# Periodic boundary mode
# --------------------------------------------------------------------------
class TestPeriodicBoundary:
    def test_positions_wrap_in_periodic_mode(self) -> None:
        solver = _make_solver(F=1e-9)  # Большая сила → быстрый снос
        solver.force_calculator.boundary_mode = "periodic"
        solver.force_calculator.L = 0.001
        solver.solve(dt=1e-4, total_time=1e-3)
        positions = solver.solution.get_current_positions()
        # При periodic positions %= L
        assert np.all(positions[:, 2] >= 0)
        assert np.all(positions[:, 2] < solver.force_calculator.L)
