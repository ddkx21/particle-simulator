"""Тесты force_calculator/direct_droplet_force_calculator.py.

Проверяют DirectDropletForceCalculator на физических инвариантах:
- сумма сил по третьему закону Ньютона ≈ 0,
- shape возвращаемых массивов,
- симметрия дипольной силы между двумя каплями,
- отсутствие сил при единственной частице,
- minimum-image convention при periodic boundary.
"""
from __future__ import annotations

import numpy as np
import pytest

from dem.force_calculator import DirectDropletForceCalculator


# --------------------------------------------------------------------------
# calculate: дипольные силы
# --------------------------------------------------------------------------
class TestDirectForces:
    def test_single_particle_zero_force(self) -> None:
        fc = DirectDropletForceCalculator(num_particles=4, L=1.0,
                                          boundary_mode="open")
        positions = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        radii = np.array([1e-5], dtype=np.float64)
        forces = fc.calculate(positions, radii)
        assert forces.shape == (1, 3)
        np.testing.assert_allclose(forces[0], np.zeros(3))

    def test_pair_newton_third_law(self) -> None:
        """F_ij = -F_ji (третий закон Ньютона)."""
        fc = DirectDropletForceCalculator(num_particles=4, L=1.0,
                                          boundary_mode="open")
        positions = np.array([
            [0.40, 0.50, 0.50],
            [0.60, 0.50, 0.50],
        ], dtype=np.float64)
        radii = np.array([2e-5, 2e-5], dtype=np.float64)
        forces = fc.calculate(positions, radii)
        np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-10)

    def test_returns_correct_shape(self) -> None:
        fc = DirectDropletForceCalculator(num_particles=8, L=1.0,
                                          boundary_mode="open")
        rng = np.random.default_rng(0)
        n = 5
        positions = rng.uniform(0.1, 0.9, size=(n, 3))
        radii = rng.uniform(1e-6, 5e-6, size=n)
        forces = fc.calculate(positions, radii)
        assert forces.shape == (n, 3)


# --------------------------------------------------------------------------
# calculate_forces_and_convection: совместный вызов
# --------------------------------------------------------------------------
class TestForcesAndConvection:
    def test_returns_pair_of_arrays(self) -> None:
        fc = DirectDropletForceCalculator(num_particles=8, L=1.0,
                                          boundary_mode="open")
        positions = np.array([
            [0.40, 0.50, 0.50],
            [0.60, 0.50, 0.50],
        ], dtype=np.float64)
        radii = np.array([2e-5, 2e-5], dtype=np.float64)
        forces, velocities = fc.calculate_forces_and_convection(positions, radii)
        assert forces.shape == (2, 3)
        assert velocities.shape == (2, 3)


# --------------------------------------------------------------------------
# Periodic boundary
# --------------------------------------------------------------------------
class TestPeriodicBoundary:
    def test_periodic_changes_force_vs_open(self) -> None:
        positions = np.array([
            [0.05, 0.50, 0.50],
            [0.95, 0.50, 0.50],
        ], dtype=np.float64)
        radii = np.array([1e-5, 1e-5], dtype=np.float64)

        fc_open = DirectDropletForceCalculator(num_particles=4, L=1.0,
                                               boundary_mode="open")
        f_open = fc_open.calculate(positions, radii)

        fc_per = DirectDropletForceCalculator(num_particles=4, L=1.0,
                                              boundary_mode="periodic")
        f_per = fc_per.calculate(positions, radii)

        # При periodic ближайший образ — через границу,
        # знак x-компоненты силы должен поменяться на противоположный.
        assert f_open[0, 0] * f_per[0, 0] < 0
