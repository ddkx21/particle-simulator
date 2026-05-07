"""Standalone PBM — решение уравнений популяционного баланса.

Запуск без DEM: аналитическое ядро или загруженное из файла.
"""

import numpy as np
import matplotlib.pyplot as plt

from pbm import VolumeGrid, PBMSolver
from pbm.kernels import AnalyticalElectrostaticKernel


def main() -> None:
    # Физические параметры (из main.py)
    eps0 = 8.85e-12
    eps_oil = 2.85
    eta_oil = 0.065
    E = 3e5

    # Параметры частиц
    r_min, r_max = 2.5e-6, 7.5e-6
    n_bins = 50
    total_particles = 1_000

    # Время
    t_end = 50.0
    n_time_points = 501

    # Сетка по объёму
    grid = VolumeGrid.from_radii_range(r_min, r_max, n_bins, spacing="geometric")

    # Начальное распределение: равномерное по радиусам
    rng = np.random.default_rng(42)
    radii = rng.uniform(r_min, r_max, total_particles)
    N0 = grid.histogram(radii)

    # Аналитическое ядро
    kernel = AnalyticalElectrostaticKernel(eps0, eps_oil, E, eta_oil)
    Q = kernel.build_matrix(grid)

    # PBM солвер (Cell-Average). domain_volume=1 даёт «стандартный» PBM без
    # деления на физический объём — для standalone сценария это допустимо.
    solver = PBMSolver(
        grid, Q, method="cell_average", integrator="BDF",
        domain_volume=1.0,
    )
    t_eval = np.linspace(0, t_end, n_time_points)

    print(f"Запуск PBM: {n_bins} бинов, {total_particles} начальных частиц")
    print(f"Объёмы: [{grid.edges[0]:.2e}, {grid.edges[-1]:.2e}] м³")

    result = solver.solve(N0, (0, t_end), t_eval=t_eval, rtol=1e-5, atol=1e-5)

    # Результаты
    t = result["t"]
    N = result["N"]
    x = grid.centers

    print(f"Начальное число частиц: {result['total_count'][0]:.0f}")
    print(f"Конечное число частиц:  {result['total_count'][-1]:.0f}")
    print(f"Начальный объём:        {result['total_volume'][0]:.4e}")
    print(f"Конечный объём:         {result['total_volume'][-1]:.4e}")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Распределение в разные моменты
    ax1 = axes[0]
    for t_val in [0, t_end * 0.25, t_end * 0.5, t_end]:
        idx = np.argmin(np.abs(t - t_val))
        ax1.plot(x, N[idx], "o-", markersize=3, label=f"t = {t[idx]:.1f} с")
    ax1.set_xlabel("Объём частиц (м³)")
    ax1.set_ylabel("Число частиц")
    ax1.set_title("Эволюция распределения по объёму")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Общее число частиц
    ax2 = axes[1]
    ax2.plot(t, result["total_count"])
    ax2.set_xlabel("Время (с)")
    ax2.set_ylabel("Общее число частиц")
    ax2.set_title("Динамика общего числа частиц")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pbm_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
