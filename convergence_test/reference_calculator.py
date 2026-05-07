"""
Референсный расчёт сил и скоростей на суперячейке K³.

Использует DirectDropletForceCalculator с boundary_mode="open" (без MIC —
все K³·N пар считаются по реальным расстояниям) и COMSOL-коррекцией
на масштабе K·L для скоростей. Извлекает результаты для центральных N частиц.
"""
import numpy as np
import taichi as ti

from .supercell_builder import build_supercell


def compute_reference(positions: np.ndarray, radii: np.ndarray,
                      L: float, K: int,
                      m_const: float, eta_const: float,
                      eps_oil: float, eta_oil: float, E: float,
                      correction,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет силы и скорости на центральных N каплях от всех K³·N капель
    суперячейки (open boundary — без MIC, прямой расчёт по реальным расстояниям).
    COMSOL-коррекция на масштабе K·L добавляет вклад образов за пределами
    суперячейки для скоростей.

    :param positions: (N, 3) позиции в [0, L]³
    :param radii: (N,) радиусы
    :param L: размер ячейки
    :param K: число копий по оси (нечётное)
    :param m_const: 12*pi*eps0*eps_oil*E²
    :param eta_const: 1/(8*pi*eta_oil)
    :param eps_oil, eta_oil, E: физические параметры
    :param correction: объект COMSOLLatticeCorrection
    :return: (forces (N,3), velocities (N,3)) для центральных частиц
    """
    from dem.force_calculator.direct_droplet_force_calculator import DirectDropletForceCalculator

    N = positions.shape[0]
    N_total = K**3 * N

    # Строим суперячейку (центральная ячейка = первые N частиц)
    super_pos, super_radii = build_supercell(positions, radii, L, K)

    # Open boundary (без MIC) + COMSOL-коррекция для скоростей
    calc = DirectDropletForceCalculator(
        num_particles=N_total,
        eps_oil=eps_oil, eta_oil=eta_oil, E=E,
        L=L * K,
        boundary_mode="open",
        correction_grid_resolution=correction.grid_resolution,
    )
    calc.load_periodic_correction(correction, L_sim=L * K)

    # Силы на всех частицах суперячейки
    forces_all = calc.calculate(super_pos, super_radii)

    # Скорости на всех частицах суперячейки
    velocities_all = calc.calculate_convection(super_pos, super_radii, forces_all)

    # Извлекаем результаты только для центральных N частиц
    forces_center = forces_all[:N]
    velocities_center = velocities_all[:N]

    return forces_center, velocities_center
