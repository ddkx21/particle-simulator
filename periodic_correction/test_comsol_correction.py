"""
Тесты для COMSOLLatticeCorrection.evaluate() с полным тензором поправки.

Проверяет:
1. Обратную совместимость (F_j=None)
2. Чистый Fz: evaluate(r, F=(0,0,Fz_comsol)) == evaluate(r) * 1.0
3. Кубическую симметрию решётки
4. Линейность по силе
"""
import numpy as np
import pytest

from periodic_correction import COMSOLLatticeCorrection


@pytest.fixture(scope="module")
def correction():
    return COMSOLLatticeCorrection.load_default()


@pytest.fixture(scope="module")
def test_points(correction):
    """Случайные точки внутри ячейки (подальше от границ для точной интерполяции)."""
    rng = np.random.default_rng(42)
    half_L = correction.L_comsol / 2.0
    margin = half_L * 0.3
    pts = rng.uniform(-half_L + margin, half_L - margin, size=(50, 3))
    return pts


class TestBackwardCompatibility:
    """evaluate(r, F_j=None) должен возвращать сырой столбец β=z."""

    def test_none_returns_z_column(self, correction, test_points):
        result_none = correction.evaluate(test_points)
        assert result_none.shape == (50, 3)
        assert not np.all(result_none == 0), "Поле не должно быть нулевым внутри ячейки"


class TestPureFz:
    """При F = (0, 0, Fz_comsol) полная поправка = сырой столбец β=z."""

    def test_pure_fz_matches_raw(self, correction, test_points):
        raw = correction.evaluate(test_points)

        F_j = np.zeros((len(test_points), 3))
        F_j[:, 2] = correction.Fz_comsol
        full = correction.evaluate(test_points, F_j=F_j)

        np.testing.assert_allclose(full, raw, rtol=1e-12,
                                   err_msg="Чистый Fz должен совпадать с сырым столбцом")


class TestCubicSymmetry:
    """Кубическая симметрия: перестановка координат и компонент."""

    def test_x_column_symmetry(self, correction):
        """G_x_component_x(a,b,c) == G_z_component_z(c,b,a)."""
        rng = np.random.default_rng(123)
        half_L = correction.L_comsol / 2.0
        margin = half_L * 0.3
        pts = rng.uniform(-half_L + margin, half_L - margin, size=(30, 3))

        Fz_c = correction.Fz_comsol

        # Поправка для F=(Fz_c, 0, 0) в точке (a, b, c)
        F_x = np.zeros((len(pts), 3))
        F_x[:, 0] = Fz_c
        result_fx = correction.evaluate(pts, F_j=F_x)

        # Поправка для F=(0, 0, Fz_c) в точке (c, b, a)
        pts_permuted = pts[:, [2, 1, 0]]
        F_z = np.zeros((len(pts), 3))
        F_z[:, 2] = Fz_c
        result_fz = correction.evaluate(pts_permuted, F_j=F_z)

        # result_fx[:, 0] (x-компонента от Fx) == result_fz[:, 2] (z-компонента от Fz)
        np.testing.assert_allclose(result_fx[:, 0], result_fz[:, 2], rtol=1e-10,
                                   err_msg="G_x^x(a,b,c) должно == G_z^z(c,b,a)")
        # result_fx[:, 1] (y-компонента от Fx) == result_fz[:, 1] (y-компонента от Fz)
        np.testing.assert_allclose(result_fx[:, 1], result_fz[:, 1], rtol=1e-10,
                                   err_msg="G_x^y(a,b,c) должно == G_z^y(c,b,a)")
        # result_fx[:, 2] (z-компонента от Fx) == result_fz[:, 0] (x-компонента от Fz)
        np.testing.assert_allclose(result_fx[:, 2], result_fz[:, 0], rtol=1e-10,
                                   err_msg="G_x^z(a,b,c) должно == G_z^x(c,b,a)")

    def test_y_column_symmetry(self, correction):
        """G_y_component_y(a,b,c) == G_z_component_z(a,c,b)."""
        rng = np.random.default_rng(456)
        half_L = correction.L_comsol / 2.0
        margin = half_L * 0.3
        pts = rng.uniform(-half_L + margin, half_L - margin, size=(30, 3))

        Fz_c = correction.Fz_comsol

        # Поправка для F=(0, Fz_c, 0) в точке (a, b, c)
        F_y = np.zeros((len(pts), 3))
        F_y[:, 1] = Fz_c
        result_fy = correction.evaluate(pts, F_j=F_y)

        # Поправка для F=(0, 0, Fz_c) в точке (a, c, b)
        pts_permuted = pts[:, [0, 2, 1]]
        F_z = np.zeros((len(pts), 3))
        F_z[:, 2] = Fz_c
        result_fz = correction.evaluate(pts_permuted, F_j=F_z)

        # result_fy[:, 1] (y-компонента от Fy) == result_fz[:, 2] (z-компонента от Fz)
        np.testing.assert_allclose(result_fy[:, 1], result_fz[:, 2], rtol=1e-10,
                                   err_msg="G_y^y(a,b,c) должно == G_z^z(a,c,b)")
        # result_fy[:, 0] (x-компонента от Fy) == result_fz[:, 0] (x-компонента от Fz)
        np.testing.assert_allclose(result_fy[:, 0], result_fz[:, 0], rtol=1e-10,
                                   err_msg="G_y^x(a,b,c) должно == G_z^x(a,c,b)")
        # result_fy[:, 2] (z-компонента от Fy) == result_fz[:, 1] (y-компонента от Fz)
        np.testing.assert_allclose(result_fy[:, 2], result_fz[:, 1], rtol=1e-10,
                                   err_msg="G_y^z(a,b,c) должно == G_z^y(a,c,b)")


class TestLinearity:
    """Линейность: evaluate(r, F1+F2) = evaluate(r, F1) + evaluate(r, F2)."""

    def test_superposition(self, correction, test_points):
        rng = np.random.default_rng(789)
        Fz_c = correction.Fz_comsol

        F1 = rng.uniform(-Fz_c, Fz_c, size=(len(test_points), 3))
        F2 = rng.uniform(-Fz_c, Fz_c, size=(len(test_points), 3))

        result_sum = correction.evaluate(test_points, F_j=F1 + F2)
        result_1 = correction.evaluate(test_points, F_j=F1)
        result_2 = correction.evaluate(test_points, F_j=F2)

        np.testing.assert_allclose(result_sum, result_1 + result_2, rtol=1e-12,
                                   err_msg="Нарушена линейность по силе")

    def test_scaling(self, correction, test_points):
        """evaluate(r, alpha*F) = alpha * evaluate(r, F)."""
        rng = np.random.default_rng(101)
        Fz_c = correction.Fz_comsol
        alpha = 3.7

        F = rng.uniform(-Fz_c, Fz_c, size=(len(test_points), 3))
        result_scaled = correction.evaluate(test_points, F_j=alpha * F)
        result_base = correction.evaluate(test_points, F_j=F)

        np.testing.assert_allclose(result_scaled, alpha * result_base, rtol=1e-12,
                                   err_msg="Нарушено масштабирование по силе")


class TestZeroForce:
    """При нулевой силе поправка = 0."""

    def test_zero_force(self, correction, test_points):
        F_zero = np.zeros((len(test_points), 3))
        result = correction.evaluate(test_points, F_j=F_zero)
        np.testing.assert_allclose(result, 0.0, atol=1e-30,
                                   err_msg="Нулевая сила должна давать нулевую поправку")


class TestEtaComsol:
    """Проверка загрузки и передачи eta_comsol."""

    def test_eta_comsol_loaded(self, correction):
        assert hasattr(correction, 'eta_comsol')
        assert correction.eta_comsol > 0
        assert correction.eta_comsol == 0.065

    def test_eta_comsol_in_grid_data(self, correction):
        grid_data = correction.get_grid_data()
        assert 'eta_comsol' in grid_data
        assert grid_data['eta_comsol'] == correction.eta_comsol


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
