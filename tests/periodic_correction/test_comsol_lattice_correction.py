"""Тесты periodic_correction/comsol_lattice_correction.py.

COMSOLLatticeCorrection — загрузка/интерполяция периодической поправки COMSOL.
"""

from __future__ import annotations

import numpy as np
import pytest

from dem.periodic_correction import COMSOLLatticeCorrection


# Тяжёлая загрузка (несколько секунд) — кэшируем для всех тестов модуля.
@pytest.fixture(scope="module")
def correction() -> COMSOLLatticeCorrection:
    return COMSOLLatticeCorrection.load_default()


# --------------------------------------------------------------------------
# Загрузка
# --------------------------------------------------------------------------
class TestLoadDefault:
    def test_load_default_returns_instance(self, correction: COMSOLLatticeCorrection) -> None:
        assert isinstance(correction, COMSOLLatticeCorrection)

    def test_grid_resolution_positive(self, correction: COMSOLLatticeCorrection) -> None:
        assert correction.grid_resolution > 0

    def test_L_and_Fz_positive(self, correction: COMSOLLatticeCorrection) -> None:
        assert correction.L_comsol > 0
        assert correction.Fz_comsol > 0


# --------------------------------------------------------------------------
# Регулярная сетка
# --------------------------------------------------------------------------
class TestRegularGrid:
    def test_grid_shapes_match_resolution(self, correction: COMSOLLatticeCorrection) -> None:
        N = correction.grid_resolution
        assert correction.grid_u.shape == (N, N, N)
        assert correction.grid_v.shape == (N, N, N)
        assert correction.grid_w.shape == (N, N, N)

    def test_grid_no_nan(self, correction: COMSOLLatticeCorrection) -> None:
        assert not np.any(np.isnan(correction.grid_u))
        assert not np.any(np.isnan(correction.grid_v))
        assert not np.any(np.isnan(correction.grid_w))


# --------------------------------------------------------------------------
# evaluate(): сырой столбец и тензорная свёртка
# --------------------------------------------------------------------------
class TestEvaluate:
    def test_evaluate_no_force_returns_three_components(
        self, correction: COMSOLLatticeCorrection
    ) -> None:
        r = np.array([[0.0, 0.0, 0.0], [correction.L_comsol * 0.1, 0.0, 0.0]])
        out = correction.evaluate(r)
        assert out.shape == (2, 3)

    def test_evaluate_with_force_full_convolution(
        self, correction: COMSOLLatticeCorrection
    ) -> None:
        r = np.array([[correction.L_comsol * 0.1, 0.0, 0.0]])
        F = np.array([[0.0, 0.0, correction.Fz_comsol]])
        out = correction.evaluate(r, F)
        # При F = (0,0,Fz_comsol) свёртка должна дать сырой столбец
        raw = correction.evaluate(r)
        np.testing.assert_allclose(out, raw, atol=1e-12)

    def test_evaluate_outside_grid_zero_fill(self, correction: COMSOLLatticeCorrection) -> None:
        # Точки далеко за пределами сетки → fill_value=0
        far = correction.L_comsol * 10
        r = np.array([[far, far, far]])
        out = correction.evaluate(r)
        np.testing.assert_allclose(out, np.zeros_like(out))


# --------------------------------------------------------------------------
# get_grid_data
# --------------------------------------------------------------------------
class TestGridData:
    def test_grid_data_keys(self, correction: COMSOLLatticeCorrection) -> None:
        data = correction.get_grid_data()
        for key in (
            "grid_u",
            "grid_v",
            "grid_w",
            "grid_min",
            "grid_max",
            "grid_dx",
            "grid_resolution",
            "L_comsol",
            "Fz_comsol",
            "eta_comsol",
        ):
            assert key in data

    def test_grid_min_max_symmetric(self, correction: COMSOLLatticeCorrection) -> None:
        data = correction.get_grid_data()
        np.testing.assert_allclose(data["grid_min"], -data["grid_max"])

    def test_grid_dx_consistent(self, correction: COMSOLLatticeCorrection) -> None:
        data = correction.get_grid_data()
        N = data["grid_resolution"]
        expected_dx = (data["grid_max"] - data["grid_min"]) / (N - 1)
        np.testing.assert_allclose(data["grid_dx"], expected_dx, rtol=1e-12)


# --------------------------------------------------------------------------
# Симметрия зеркалирования (mirror_to_full_cell)
# --------------------------------------------------------------------------
class TestMirrorSymmetry:
    """Проверка симметрии вдоль осей x и y (свойства стоклета по z)."""

    def test_u_odd_in_x(self, correction: COMSOLLatticeCorrection) -> None:
        # u(x,y,z) — нечётная по x
        x = correction.L_comsol * 0.15
        r_pos = np.array([[+x, 0.0, 0.0]])
        r_neg = np.array([[-x, 0.0, 0.0]])
        out_pos = correction.evaluate(r_pos)
        out_neg = correction.evaluate(r_neg)
        np.testing.assert_allclose(out_pos[0, 0], -out_neg[0, 0], atol=1e-10)

    def test_v_odd_in_y(self, correction: COMSOLLatticeCorrection) -> None:
        # v(x,y,z) — нечётная по y
        y = correction.L_comsol * 0.15
        r_pos = np.array([[0.0, +y, 0.0]])
        r_neg = np.array([[0.0, -y, 0.0]])
        out_pos = correction.evaluate(r_pos)
        out_neg = correction.evaluate(r_neg)
        np.testing.assert_allclose(out_pos[0, 1], -out_neg[0, 1], atol=1e-10)

    def test_w_even_in_x_and_y(self, correction: COMSOLLatticeCorrection) -> None:
        # w(x,y,z) — чётная по x и y
        d = correction.L_comsol * 0.15
        r1 = np.array([[+d, +d, 0.0]])
        r2 = np.array([[-d, -d, 0.0]])
        out1 = correction.evaluate(r1)
        out2 = correction.evaluate(r2)
        np.testing.assert_allclose(out1[0, 2], out2[0, 2], atol=1e-10)
