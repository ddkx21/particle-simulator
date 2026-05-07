"""Тесты pbm/kernels/analytical_kernel.py — AnalyticalElectrostaticKernel."""

from __future__ import annotations

import numpy as np
import pytest

from pbm import VolumeGrid
from pbm.kernels import AnalyticalElectrostaticKernel


class TestAnalyticalKernel:
    def test_symmetry(self) -> None:
        g = VolumeGrid.from_radii_range(2.5e-6, 7.5e-6, 20)
        Q = AnalyticalElectrostaticKernel().build_matrix(g)
        np.testing.assert_allclose(Q, Q.T)

    def test_positivity(self) -> None:
        g = VolumeGrid.from_radii_range(2.5e-6, 7.5e-6, 20)
        Q = AnalyticalElectrostaticKernel().build_matrix(g)
        assert np.all(Q > 0)

    def test_evaluate_scalar(self) -> None:
        k = AnalyticalElectrostaticKernel()
        v = 1e-15
        expected = k.prefactor * v / 2
        assert k.evaluate(v, v) == pytest.approx(expected, rel=1e-10)
