"""Тесты pbm/redistribution.py — fixed_pivot и cell_average."""

from __future__ import annotations

import numpy as np
import pytest

from pbm import VolumeGrid, cell_average, fixed_pivot


# --------------------------------------------------------------------------
# fixed_pivot
# --------------------------------------------------------------------------
class TestFixedPivot:
    def test_inner_conserves_number_and_volume(self) -> None:
        g = VolumeGrid(1e-18, 1e-12, 30)
        v = g.centers[10] + g.centers[15]
        targets = fixed_pivot(v, g)
        weights = sum(w for _, w in targets)
        vol = sum(w * g.centers[i] for i, w in targets)
        assert weights == pytest.approx(1.0)
        assert vol == pytest.approx(v)

    def test_below_first_center_clamps_to_zero(self) -> None:
        g = VolumeGrid(1.0, 100.0, 10)
        assert fixed_pivot(0.5, g) == [(0, 1.0)]

    def test_above_last_center_clamps_to_last(self) -> None:
        g = VolumeGrid(1.0, 100.0, 10)
        assert fixed_pivot(1e6, g) == [(g.n_bins - 1, 1.0)]


# --------------------------------------------------------------------------
# cell_average
# --------------------------------------------------------------------------
class TestCellAverageInner:
    def test_inner_preserves_number_and_volume(self) -> None:
        g = VolumeGrid(1e-18, 1e-12, 30)
        n = g.n_bins
        i0 = 15
        rate = 1.0
        v_target = g.centers[i0] * 1.2
        B = np.zeros(n)
        M = np.zeros(n)
        B[i0] = rate
        M[i0] = rate * v_target
        out = cell_average(B, M, g)
        assert out.sum() == pytest.approx(rate)
        assert (out * g.centers).sum() == pytest.approx(rate * v_target, rel=1e-10)

    def test_zero_rate_zero_output(self) -> None:
        g = VolumeGrid(1.0, 100.0, 10)
        out = cell_average(np.zeros(10), np.zeros(10), g)
        np.testing.assert_array_equal(out, np.zeros(10))


class TestCellAverageBoundaries:
    def test_left_boundary_preserves_count(self) -> None:
        """Ветка `if left[0]` — a_bar < x[0] на bin 0."""
        g = VolumeGrid(1.0, 10.0, 5, spacing="linear")
        n = g.n_bins
        B = np.zeros(n)
        M = np.zeros(n)
        B[0] = 7.0
        M[0] = B[0] * (g.centers[0] * 0.5)
        out = cell_average(B, M, g)
        assert out[0] == pytest.approx(B[0])

    def test_right_boundary_preserves_count(self) -> None:
        """Ветка `if right[n - 1]` — a_bar >= x[-1] на bin n-1."""
        g = VolumeGrid(1.0, 10.0, 5, spacing="linear")
        n = g.n_bins
        B = np.zeros(n)
        M = np.zeros(n)
        B[n - 1] = 5.0
        M[n - 1] = B[n - 1] * (g.centers[-1] * 1.5)
        out = cell_average(B, M, g)
        assert out[n - 1] == pytest.approx(B[n - 1])
