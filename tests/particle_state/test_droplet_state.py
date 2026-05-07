"""Тесты particle_state/droplet_state.py — DropletState (хранение состояния капель)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dem.particle_state import DropletState


# --------------------------------------------------------------------------
# Конструктор и валидация
# --------------------------------------------------------------------------
class TestConstruction:
    def test_from_arrays(self) -> None:
        positions = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        radii = np.array([1.0, 2.0])
        state = DropletState(positions, radii, time=1.5)
        np.testing.assert_array_equal(state.positions, positions)
        np.testing.assert_array_equal(state.radii, radii)
        assert state.time == 1.5

    def test_missing_args_raises(self) -> None:
        with pytest.raises(ValueError):
            DropletState()

    def test_invalid_extension_raises(self) -> None:
        with pytest.raises(ValueError):
            DropletState(filename="state.txt")


# --------------------------------------------------------------------------
# copy
# --------------------------------------------------------------------------
class TestCopy:
    def test_copy_is_independent(self) -> None:
        state = DropletState(np.array([[1.0, 2.0, 3.0]]), np.array([4.0]),
                             time=0.0)
        clone = state.copy()
        clone.positions[0, 0] = 999.0
        assert state.positions[0, 0] == 1.0


# --------------------------------------------------------------------------
# Сериализация: npz
# --------------------------------------------------------------------------
class TestNpzRoundtrip:
    def test_save_and_load_npz(self, tmp_path: Path) -> None:
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        radii = np.array([0.5, 1.5])
        state = DropletState(positions, radii, time=2.0)

        path = tmp_path / "state.npz"
        state.save(str(path))

        loaded = DropletState(filename=str(path))
        np.testing.assert_array_equal(loaded.positions, positions)
        np.testing.assert_array_equal(loaded.radii, radii)
        assert loaded.time == 2.0


# --------------------------------------------------------------------------
# Сериализация: xlsx
# --------------------------------------------------------------------------
class TestXlsxRoundtrip:
    def test_export_and_import_xlsx(self, tmp_path: Path) -> None:
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        radii = np.array([0.5, 1.5])
        state = DropletState(positions, radii, time=2.0)

        path = tmp_path / "state.xlsx"
        state.export_to_xlsx(str(path))

        loaded = DropletState(filename=str(path))
        np.testing.assert_allclose(loaded.positions, positions)
        np.testing.assert_allclose(loaded.radii, radii)
        # Время не экспортируется в xlsx — сбрасывается в 0
        assert loaded.time == 0


# --------------------------------------------------------------------------
# Repr
# --------------------------------------------------------------------------
class TestRepr:
    def test_repr_contains_time_and_count(self) -> None:
        state = DropletState(np.zeros((3, 3)), np.zeros(3), time=4.2)
        r = repr(state)
        assert "time=4.2" in r
        assert "num_particles=3" in r
