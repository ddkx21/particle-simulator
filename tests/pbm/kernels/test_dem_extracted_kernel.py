"""Тесты pbm/kernels/dem_extracted_kernel.py — DEMExtractedKernel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pbm import VolumeGrid
from pbm.kernels import AnalyticalElectrostaticKernel, DEMExtractedKernel


# --------------------------------------------------------------------------
# Базовые свойства finalize/reset
# --------------------------------------------------------------------------
class TestDEMKernelBasics:
    def test_symmetry_after_finalize(self) -> None:
        g = VolumeGrid.from_radii_range(2.5e-6, 7.5e-6, 10)
        dem = DEMExtractedKernel(g, domain_volume=1.0)
        dem.update_concentrations(np.array([3e-6, 5e-6, 6e-6]), 0.1)
        dem.record_collision(3e-6, 5e-6)
        K = dem.finalize()
        np.testing.assert_allclose(K, K.T)

    def test_reset_clears_state(self) -> None:
        g = VolumeGrid(1e-18, 1e-12, 5)
        dem = DEMExtractedKernel(g, domain_volume=1.0)
        dem.update_concentrations(np.array([1e-6]), 0.1)
        dem.record_collision(1e-6, 1e-6)
        dem.reset()
        K = dem.finalize()
        np.testing.assert_array_equal(K, np.zeros((5, 5)))


# --------------------------------------------------------------------------
# Восстановление аналитического Q (главный физический тест)
# --------------------------------------------------------------------------
class TestDEMKernelRecovery:
    def test_recovers_analytical_kernel(self) -> None:
        g = VolumeGrid.from_radii_range(2.5e-6, 7.5e-6, 15)
        analytical = AnalyticalElectrostaticKernel().build_matrix(g)

        V_d = 1e-12
        T = 10.0
        dt = 0.04
        dem = DEMExtractedKernel(g, domain_volume=V_d)
        rng = np.random.default_rng(0)
        radii = rng.uniform(2.5e-6, 7.5e-6, 1000)
        for _ in range(int(T / dt)):
            dem.update_concentrations(radii, dt)

        n_avg = dem._conc_time_sum / dem._total_time
        for i in range(g.n_bins):
            for j in range(i, g.n_bins):
                ni, nj = n_avg[i], n_avg[j]
                if ni == 0 or nj == 0:
                    continue
                if i == j:
                    expected = 0.5 * analytical[i, j] * ni * nj * V_d * T
                else:
                    expected = analytical[i, j] * ni * nj * V_d * T
                n_coll = int(round(expected))
                for _ in range(n_coll):
                    r_i = (3 * g.centers[i] / (4 * np.pi)) ** (1 / 3)
                    r_j = (3 * g.centers[j] / (4 * np.pi)) ** (1 / 3)
                    dem.record_collision(r_i, r_j)

        K = dem.finalize()
        ratio = K[analytical > 0] / analytical[analytical > 0]
        assert ratio.mean() == pytest.approx(1.0, rel=0.05)
        assert ratio.std() < 0.05


# --------------------------------------------------------------------------
# Сериализация: load_from_file
# --------------------------------------------------------------------------
class TestDEMKernelLoad:
    def test_load_from_file_roundtrip(self, tmp_path: Path) -> None:
        g = VolumeGrid(1.0, 100.0, 6, spacing="geometric")
        n = 8
        volumes = np.geomspace(0.5, 200.0, n)
        rng = np.random.default_rng(42)
        freq = rng.uniform(0.0, 1.0, size=(n, n))
        freq = 0.5 * (freq + freq.T)

        path = tmp_path / "kernel.npz"
        np.savez(path, volumes=volumes, freq=freq)

        K = DEMExtractedKernel.load_from_file(str(path), g)
        assert K.shape == (g.n_bins, g.n_bins)
        assert np.all(K >= 0.0)

    def test_load_from_file_clips_negatives(self, tmp_path: Path) -> None:
        g = VolumeGrid(1.0, 100.0, 6, spacing="geometric")
        n = 6
        volumes = np.geomspace(0.5, 200.0, n)
        freq = np.full((n, n), -1.0)
        path = tmp_path / "neg_kernel.npz"
        np.savez(path, volumes=volumes, freq=freq)

        K = DEMExtractedKernel.load_from_file(str(path), g)
        assert np.all(K >= 0.0)
