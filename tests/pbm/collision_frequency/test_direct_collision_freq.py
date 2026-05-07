"""Тесты pbm/collision_frequency/direct_collision_freq.py."""
from __future__ import annotations

import numpy as np

from pbm import VolumeGrid
from pbm.collision_frequency import DirectCollisionFrequency


class TestDirectCollisionFrequency:
    def test_no_overlap_no_contacts(self) -> None:
        g = VolumeGrid.from_radii_range(1e-6, 5e-6, 5)
        cf = DirectCollisionFrequency(g)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        radii = np.array([1e-6, 1e-6])
        M = cf.compute(positions, radii)
        assert M.sum() == 0

    def test_overlap_records_pair(self) -> None:
        g = VolumeGrid.from_radii_range(1e-6, 5e-6, 5)
        cf = DirectCollisionFrequency(g)
        positions = np.array([[0.0, 0.0, 0.0], [3e-6, 0.0, 0.0]])
        radii = np.array([2e-6, 2e-6])
        M = cf.compute(positions, radii)
        assert M.sum() > 0
        np.testing.assert_array_equal(M, M.T)

    def test_periodic_minimum_image(self) -> None:
        g = VolumeGrid.from_radii_range(1e-6, 5e-6, 5)
        cf = DirectCollisionFrequency(g)
        L = 1e-5
        positions = np.array([[1e-7, 0.0, 0.0], [L - 1e-7, 0.0, 0.0]])
        radii = np.array([3e-7, 3e-7])
        M_no = cf.compute(positions, radii, L=L, periodic=False)
        M_yes = cf.compute(positions, radii, L=L, periodic=True)
        assert M_no.sum() == 0
        assert M_yes.sum() > 0
