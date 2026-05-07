"""Тесты particle_generator/uniform_droplet_generator.py.

UniformDropletGenerator — генератор случайных капель без пересечений.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dem.particle_generator import UniformDropletGenerator
from dem.particle_state import DropletState


# --------------------------------------------------------------------------
# generate(): корректная генерация без пересечений
# --------------------------------------------------------------------------
class TestGenerate:
    def test_returns_droplet_state(self) -> None:
        np.random.seed(0)
        gen = UniformDropletGenerator(
            coord_range=(0.0, 1.0), radii_range=(0.01, 0.02), num_particles=10, minimum_distance=0.0
        )
        state = gen.generate()
        assert isinstance(state, DropletState)
        assert state.positions.shape == (10, 3)
        assert state.radii.shape == (10,)

    def test_no_overlaps_after_generation(self) -> None:
        np.random.seed(1)
        gen = UniformDropletGenerator(
            coord_range=(0.0, 1.0), radii_range=(0.01, 0.02), num_particles=15, minimum_distance=0.0
        )
        state = gen.generate()
        # Каждая пара должна быть не ближе r_i + r_j
        for i in range(len(state.radii)):
            for j in range(i + 1, len(state.radii)):
                d = np.linalg.norm(state.positions[i] - state.positions[j])
                assert d >= state.radii[i] + state.radii[j] - 1e-12

    def test_radii_within_range(self) -> None:
        np.random.seed(2)
        gen = UniformDropletGenerator(
            coord_range=(0.0, 1.0), radii_range=(0.01, 0.02), num_particles=8, minimum_distance=0.0
        )
        state = gen.generate()
        assert np.all(state.radii >= 0.01 - 1e-12)
        assert np.all(state.radii <= 0.02 + 1e-12)


# --------------------------------------------------------------------------
# Невозможность развести пересечения
# --------------------------------------------------------------------------
class TestImpossiblePacking:
    def test_too_dense_raises(self) -> None:
        """Если радиусы заведомо больше домена — генератор должен сдаться."""
        np.random.seed(3)
        gen = UniformDropletGenerator(
            coord_range=(0.0, 0.05),
            radii_range=(0.04, 0.05),
            num_particles=20,
            minimum_distance=0.01,
        )
        with pytest.raises(ValueError):
            gen.generate()


# --------------------------------------------------------------------------
# Сериализация: save/load в xlsx
# --------------------------------------------------------------------------
class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        np.random.seed(4)
        gen = UniformDropletGenerator(
            coord_range=(0.0, 1.0), radii_range=(0.01, 0.02), num_particles=5
        )
        gen.generate()

        path = tmp_path / "drops.xlsx"
        gen.save(str(path))

        gen2 = UniformDropletGenerator(
            coord_range=(0.0, 1.0), radii_range=(0.01, 0.02), num_particles=5
        )
        gen2.load(str(path))

        np.testing.assert_allclose(gen2.droplet_state.positions, gen.droplet_state.positions)
        np.testing.assert_allclose(gen2.droplet_state.radii, gen.droplet_state.radii)
