"""Тесты pbm/volume_grid.py — дискретизация объёмного пространства."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pbm import VolumeGrid


# --------------------------------------------------------------------------
# Конструктор и геометрия сетки
# --------------------------------------------------------------------------
class TestVolumeGridConstruction:
    def test_geometric_centers(self) -> None:
        g = VolumeGrid(1e-18, 1e-12, 30, spacing="geometric")
        np.testing.assert_allclose(g.centers, np.sqrt(g.edges[:-1] * g.edges[1:]))

    def test_linear_centers_arithmetic(self) -> None:
        g = VolumeGrid(1.0, 10.0, 9, spacing="linear")
        np.testing.assert_allclose(g.centers, 0.5 * (g.edges[:-1] + g.edges[1:]))

    def test_logarithmic_matches_geometric(self) -> None:
        g_log = VolumeGrid(1.0, 1000.0, 6, spacing="logarithmic")
        g_geo = VolumeGrid(1.0, 1000.0, 6, spacing="geometric")
        np.testing.assert_allclose(g_log.edges, g_geo.edges)

    def test_widths_equal_edge_diffs(self) -> None:
        g = VolumeGrid(1.0, 100.0, 5, spacing="linear")
        np.testing.assert_allclose(g.widths, np.diff(g.edges))


class TestVolumeGridFromRadii:
    def test_volume_bounds_match_sphere_formula(self) -> None:
        g = VolumeGrid.from_radii_range(2.5e-6, 7.5e-6, 20)
        v_min = (4 / 3) * np.pi * (2.5e-6) ** 3
        v_max = (4 / 3) * np.pi * (7.5e-6) ** 3
        assert g.edges[0] == pytest.approx(v_min)
        assert g.edges[-1] == pytest.approx(v_max)
        assert g.n_bins == 20


# --------------------------------------------------------------------------
# Валидация входных данных
# --------------------------------------------------------------------------
class TestVolumeGridValidation:
    def test_negative_v_min_raises(self) -> None:
        with pytest.raises(ValueError, match="0 < v_min < v_max"):
            VolumeGrid(-1.0, 1.0, 5)

    def test_v_max_le_v_min_raises(self) -> None:
        with pytest.raises(ValueError, match="0 < v_min < v_max"):
            VolumeGrid(1.0, 1.0, 5)

    def test_too_few_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins >= 2"):
            VolumeGrid(1.0, 10.0, 1)

    def test_unknown_spacing_raises(self) -> None:
        with pytest.raises(ValueError, match="Неизвестный тип spacing"):
            VolumeGrid(1.0, 10.0, 5, spacing="unsupported")


# --------------------------------------------------------------------------
# bin_index / bin_indices
# --------------------------------------------------------------------------
class TestBinIndex:
    def test_clamp_below_min(self) -> None:
        g = VolumeGrid(1e-18, 1e-12, 10)
        assert g.bin_index(1e-20) == 0

    def test_clamp_above_max(self) -> None:
        g = VolumeGrid(1e-18, 1e-12, 10)
        assert g.bin_index(1e-10) == 9

    def test_inside_returns_valid_index(self) -> None:
        g = VolumeGrid(1e-18, 1e-12, 10)
        assert 0 <= g.bin_index(1e-15) < 10

    def test_below_min_warns_once(self, caplog) -> None:
        g = VolumeGrid(1.0, 10.0, 5)
        with caplog.at_level(logging.WARNING, logger="pbm.volume_grid"):
            g.bin_index(0.1)
            g.bin_index(0.05)
        low_warns = [r for r in caplog.records if "v_min" in r.getMessage()]
        assert len(low_warns) == 1

    def test_above_max_warns_once(self, caplog) -> None:
        g = VolumeGrid(1.0, 10.0, 5)
        with caplog.at_level(logging.WARNING, logger="pbm.volume_grid"):
            g.bin_index(100.0)
            g.bin_index(200.0)
        high_warns = [r for r in caplog.records if "v_max" in r.getMessage()]
        assert len(high_warns) == 1


class TestBinIndices:
    def test_vectorized_clamp_and_warn(self, caplog) -> None:
        g = VolumeGrid(1.0, 10.0, 5)
        with caplog.at_level(logging.WARNING, logger="pbm.volume_grid"):
            idx = g.bin_indices(np.array([0.5, 5.0, 50.0]))
        assert idx[0] == 0 and idx[2] == g.n_bins - 1
        msgs = [r.getMessage() for r in caplog.records]
        assert any("v_max" in m for m in msgs)
        assert any("v_min" in m for m in msgs)


# --------------------------------------------------------------------------
# histogram
# --------------------------------------------------------------------------
class TestHistogram:
    def test_total_preserved(self) -> None:
        g = VolumeGrid.from_radii_range(2.5e-6, 7.5e-6, 30)
        rng = np.random.default_rng(0)
        radii = rng.uniform(2.5e-6, 7.5e-6, 1000)
        N = g.histogram(radii)
        assert N.sum() == 1000

    def test_includes_max_value(self) -> None:
        g = VolumeGrid.from_radii_range(1e-6, 1e-5, 10)
        N = g.histogram(np.array([1e-5]))
        assert N.sum() == 1

    def test_overflow_underflow_logged(self, caplog) -> None:
        g = VolumeGrid(1.0, 10.0, 5)
        radii = np.array([1e-3, 0.5, 1.5, 2.5])
        with caplog.at_level(logging.WARNING, logger="pbm.volume_grid"):
            counts = g.histogram(radii)
        assert counts.sum() == len(radii)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("v > v_max" in m for m in msgs)
        assert any("v < v_min" in m for m in msgs)
