"""Тесты solution/droplet_solution.py.

DropletSolution — хранение траекторий, цепочка решений после столкновений.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dem.particle_state import DropletState
from dem.solution import DropletSolution


def _initial_state(n: int = 3, time: float = 0.0) -> DropletState:
    rng = np.random.default_rng(0)
    positions = rng.random((n, 3))
    radii = np.full(n, 0.05)
    return DropletState(positions, radii, time=time)


# --------------------------------------------------------------------------
# Конструктор и save_step
# --------------------------------------------------------------------------
class TestConstructionAndSave:
    def test_initial_state_recorded(self) -> None:
        state = _initial_state()
        sol = DropletSolution(initial_droplet_state=state, length=10)
        assert sol.num_particles == 3
        np.testing.assert_array_equal(sol.radii, state.radii)
        # Начальный шаг записан как current_step = 0
        assert sol.current_step == 0

    def test_save_step_grows_buffers(self) -> None:
        state = _initial_state()
        sol = DropletSolution(initial_droplet_state=state, length=2)
        # Заполняем буфер сверх длины — он должен удвоиться
        for k in range(5):
            sol.save_step(k * 0.1, np.full((3, 3), float(k)))
        assert sol.length >= 6
        # Времена в порядке возрастания
        times = sol.get_times().flatten()
        assert times[-1] >= times[0]


# --------------------------------------------------------------------------
# Аксессоры времени и позиций
# --------------------------------------------------------------------------
class TestAccessors:
    def test_get_current_time(self) -> None:
        state = _initial_state(time=2.5)
        sol = DropletSolution(initial_droplet_state=state, length=5)
        assert sol.get_current_time() == 2.5

    def test_get_current_positions_shape(self) -> None:
        state = _initial_state(n=4)
        sol = DropletSolution(initial_droplet_state=state, length=5)
        assert sol.get_current_positions().shape == (4, 3)

    def test_compact_trims_arrays(self) -> None:
        state = _initial_state()
        sol = DropletSolution(initial_droplet_state=state, length=10)
        sol.save_step(0.1, np.zeros((3, 3)))
        sol.compact()
        assert len(sol.times) == sol.current_step + 1
        assert len(sol.trajectories) == sol.current_step + 1


# --------------------------------------------------------------------------
# Цепочка: generate_next_solution
# --------------------------------------------------------------------------
class TestNextSolution:
    def test_collision_merges_two_drops(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.5, 0.5, 0.5]])
        radii = np.array([0.05, 0.05, 0.05])
        state = DropletState(positions, radii, time=0.0)
        sol = DropletSolution(initial_droplet_state=state, length=5)
        sol.collided_droplets = np.array([[0, 1]])
        next_sol = sol.generate_next_solution()

        # 3 капли → 2 (одна слилась)
        assert next_sol.num_particles == 2
        # Сохранение объёма: r_new^3 = r1^3 + r2^3
        expected_v = 2 * (0.05**3)
        # Последний радиус — слившаяся капля
        assert np.isclose(next_sol.radii[-1] ** 3, expected_v)
        # Связь prev/next
        assert next_sol._prev is sol
        assert sol._next is next_sol


# --------------------------------------------------------------------------
# Сохранение/загрузка
# --------------------------------------------------------------------------
class TestPersistence:
    def test_save_and_load_chain(self, tmp_path: Path) -> None:
        state = _initial_state()
        sol = DropletSolution(initial_droplet_state=state, length=5)
        sol.save_step(0.5, np.ones((3, 3)))

        path = tmp_path / "chain.npz"
        sol.save_chain_to_file(str(path))

        loaded = DropletSolution.load_chain_from_file(str(path))
        np.testing.assert_allclose(loaded.radii, sol.radii)
        np.testing.assert_allclose(loaded.trajectories, sol.get_trajectories())


# --------------------------------------------------------------------------
# get_state: интерполяция по времени
# --------------------------------------------------------------------------
class TestGetState:
    def test_interpolated_state(self) -> None:
        state = _initial_state()
        sol = DropletSolution(initial_droplet_state=state, length=5)
        sol.save_step(1.0, np.full((3, 3), 1.0))
        sol.compact()

        mid = sol.get_state(0.5)
        # Между t=0 (state.positions) и t=1 (всё единицы)
        expected = (state.positions + np.ones((3, 3))) / 2
        np.testing.assert_allclose(mid.positions, expected)
        assert mid.time == 0.5

    def test_out_of_range_raises(self) -> None:
        state = _initial_state()
        sol = DropletSolution(initial_droplet_state=state, length=5)
        with pytest.raises(ValueError):
            sol.get_state(-1.0)
