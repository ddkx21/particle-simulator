"""Тесты octree/force_tree.py — TreeDropletForceCalculator (Барнс-Хатт O(N log N)).

Cравнивает результат с эталонным DirectDropletForceCalculator на тех же входах.
"""

from __future__ import annotations

import numpy as np

from dem.force_calculator import DirectDropletForceCalculator
from dem.octree import TreeDropletForceCalculator


def _relative_error(v_ref: np.ndarray, v_test: np.ndarray, eps_factor: float = 1e-10) -> np.ndarray:
    diff = np.linalg.norm(v_ref - v_test, axis=1)
    ref = np.linalg.norm(v_ref, axis=1) + np.mean(np.linalg.norm(v_ref, axis=1)) * eps_factor
    return diff / ref


# --------------------------------------------------------------------------
# calculate(): силы должны совпадать с direct
# --------------------------------------------------------------------------
class TestForcesVsDirect:
    def test_forces_match_direct_within_tolerance(self) -> None:
        rng = np.random.default_rng(42)
        n = 30
        L = 0.01
        positions = rng.random((n, 3)) * L
        radii = rng.random(n) * 30e-6 + 20e-6

        direct = DirectDropletForceCalculator(num_particles=n, L=L, boundary_mode="open")
        f_d = direct.calculate(positions, radii)

        tree = TreeDropletForceCalculator(num_particles=n, theta=0.3, mpl=1, L=L, periodic=False)
        f_t = tree.calculate(positions, radii)

        err = _relative_error(f_d, f_t)
        assert np.mean(err) < 0.05


# --------------------------------------------------------------------------
# calculate_convection(): скорости стокслета
# --------------------------------------------------------------------------
class TestConvectionVsDirect:
    def test_convection_match_direct_within_tolerance(self) -> None:
        rng = np.random.default_rng(42)
        n = 30
        L = 0.01
        positions = rng.random((n, 3)) * L
        radii = rng.random(n) * 30e-6 + 20e-6

        direct = DirectDropletForceCalculator(num_particles=n, L=L, boundary_mode="open")
        f_d = direct.calculate(positions, radii)
        v_d = direct.calculate_convection(positions, radii, f_d)

        tree = TreeDropletForceCalculator(num_particles=n, theta=0.3, mpl=1, L=L, periodic=False)
        f_t = tree.calculate(positions, radii)
        v_t = tree.calculate_convection(positions, radii, f_t)

        err = _relative_error(v_d, v_t)
        assert np.mean(err) < 0.1


# --------------------------------------------------------------------------
# Сходимость: меньшее theta → ближе к direct
# --------------------------------------------------------------------------
class TestThetaConvergence:
    def test_smaller_theta_decreases_error(self) -> None:
        rng = np.random.default_rng(42)
        n = 30
        L = 0.01
        positions = rng.random((n, 3)) * L
        radii = rng.random(n) * 30e-6 + 20e-6

        direct = DirectDropletForceCalculator(num_particles=n, L=L, boundary_mode="open")
        f_d = direct.calculate(positions, radii)

        prev_err = float("inf")
        for theta in (0.8, 0.5, 0.3):
            tree = TreeDropletForceCalculator(
                num_particles=n, theta=theta, mpl=1, L=L, periodic=False
            )
            f_t = tree.calculate(positions, radii)
            err = float(np.mean(_relative_error(f_d, f_t)))
            assert err <= prev_err + 1e-4
            prev_err = err
