"""Тесты collision_detector/spatial_hash_collision_detector.py.

Проверяют детектор столкновений на основе spatial hashing:
- отсутствие контактов при разнесённых частицах,
- регистрацию пересекающихся пар,
- minimum-image convention при periodic boundary,
- симметрию результата (i, j) с i < j.
"""
from __future__ import annotations

import numpy as np
import pytest

from dem.collision_detector import SpatialHashCollisionDetector
from dem.collision_detector.spatial_hash_collision_detector import _next_power_of_2


# --------------------------------------------------------------------------
# Утилиты модуля
# --------------------------------------------------------------------------
class TestNextPowerOf2:
    @pytest.mark.parametrize("n,expected", [(1, 1), (2, 2), (3, 4), (5, 8),
                                             (16, 16), (17, 32)])
    def test_pow2_ceiling(self, n: int, expected: int) -> None:
        assert _next_power_of_2(n) == expected


# --------------------------------------------------------------------------
# detect: open boundary
# --------------------------------------------------------------------------
class TestDetectOpenBoundary:
    def test_no_overlap_no_collisions(self) -> None:
        det = SpatialHashCollisionDetector(num_particles=8, L=1.0,
                                           boundary_mode="open")
        positions = np.array([
            [0.10, 0.10, 0.10],
            [0.90, 0.90, 0.90],
        ], dtype=np.float64)
        radii = np.full(2, 0.05, dtype=np.float64)
        is_coll, pairs = det.detect(positions, radii,
                                    L=1.0, boundary_mode="open")
        assert is_coll is False
        assert pairs.shape == (0, 2)

    def test_overlap_returns_pair(self) -> None:
        det = SpatialHashCollisionDetector(num_particles=8, L=1.0,
                                           boundary_mode="open")
        positions = np.array([
            [0.50, 0.50, 0.50],
            [0.52, 0.50, 0.50],
        ], dtype=np.float64)
        radii = np.full(2, 0.05, dtype=np.float64)
        is_coll, pairs = det.detect(positions, radii,
                                    L=1.0, boundary_mode="open")
        assert is_coll is True
        assert len(pairs) == 1
        i, j = sorted(pairs[0])
        assert (i, j) == (0, 1)


# --------------------------------------------------------------------------
# detect: periodic boundary
# --------------------------------------------------------------------------
class TestDetectPeriodicBoundary:
    def test_periodic_minimum_image_records_contact(self) -> None:
        det = SpatialHashCollisionDetector(num_particles=4, L=1.0,
                                           boundary_mode="periodic")
        positions = np.array([
            [0.01, 0.50, 0.50],
            [0.99, 0.50, 0.50],
        ], dtype=np.float64)
        radii = np.full(2, 0.05, dtype=np.float64)

        is_no, _ = det.detect(positions, radii, L=1.0, boundary_mode="open")
        assert is_no is False

        is_yes, pairs = det.detect(positions, radii,
                                   L=1.0, boundary_mode="periodic")
        assert is_yes is True
        assert len(pairs) >= 1


# --------------------------------------------------------------------------
# Краевые случаи
# --------------------------------------------------------------------------
class TestDetectEdgeCases:
    def test_zero_radii_returns_no_collision(self) -> None:
        det = SpatialHashCollisionDetector(num_particles=4, L=1.0,
                                           boundary_mode="open")
        positions = np.zeros((2, 3), dtype=np.float64)
        radii = np.zeros(2, dtype=np.float64)
        is_coll, pairs = det.detect(positions, radii)
        assert is_coll is False
        assert pairs.shape == (0, 2)

    def test_pairs_unique_after_dedup(self) -> None:
        """Hash-коллизии могут привести к повторам — должна срабатывать дедупликация."""
        det = SpatialHashCollisionDetector(num_particles=8, L=1.0,
                                           boundary_mode="open")
        positions = np.array([
            [0.50, 0.50, 0.50],
            [0.51, 0.50, 0.50],
            [0.52, 0.50, 0.50],
        ], dtype=np.float64)
        radii = np.full(3, 0.02, dtype=np.float64)
        is_coll, pairs = det.detect(positions, radii)
        assert is_coll is True
        # Каждая пара появляется ровно один раз
        assert len(pairs) == len({tuple(sorted(p)) for p in pairs})
