"""
Скрипт для анализа сходимости по временному шагу dt.

Запускает симуляцию с одинаковыми начальными условиями для разных dt,
затем сравнивает результаты и строит графики сходимости.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from dem.collision_detector import SpatialHashCollisionDetector
from dem.force_calculator import *
from dem.particle_generator import *
from dem.particle_state import *
from dem.post_processor import *
from dem.solution import *
from dem.solver import *
import taichi as ti

n_of_threads = 16
ti.init(arch=ti.cpu, cpu_max_num_threads=n_of_threads, default_fp=ti.f64)

def run_simulation(initial_state, dt, t_stop, box_size,
                   eps_oil, eta_oil, eta_water, rho_water, rho_oil, E,
                   boundary_mode):
    """
    Запускает одну симуляцию с заданным dt и возвращает объект solution.
    """
    num_particles = len(initial_state.radii)

    # Используем копию начального состояния, чтобы не мутировать оригинал
    state_copy = initial_state.copy()

    force_calculator = DirectDropletForceCalculator(
        num_particles=num_particles, eps_oil=eps_oil,
        eta_oil=eta_oil, eta_water=eta_water,
        rho_water=rho_water, rho_oil=rho_oil, E=E,
        L=box_size, boundary_mode=boundary_mode
    )

    collision_detector = SpatialHashCollisionDetector(
        num_particles=num_particles, L=box_size,
        boundary_mode=boundary_mode
    )

    solution = DropletSolution(
        initial_droplet_state=state_copy,
        real_time_visualization=False
    )

    post_processor = DropletPostProcessor(solution, box_size=box_size)

    solver = EulerDropletSolver(
        force_calculator=force_calculator,
        solution=solution,
        post_processor=post_processor,
        collision_detector=collision_detector
    )

    print(f"\n{'='*60}")
    print(f"  Запуск симуляции с dt = {dt}")
    print(f"{'='*60}")

    t_start = time.time()
    solver.solve(dt, t_stop)
    elapsed = time.time() - t_start

    print(f"  dt = {dt}: завершено за {elapsed:.1f} сек")
    return solver.solution

def get_droplet_count_over_time(solution, time_points):
    """Количество капель в заданные моменты времени."""
    counts = np.zeros(len(time_points))
    for i, t in enumerate(time_points):
        try:
            state = solution.get_state(t)
            counts[i] = len(state.radii)
        except ValueError:
            counts[i] = np.nan
    return counts

def get_mean_radius_over_time(solution, time_points):
    """Средний радиус капель в заданные моменты времени."""
    means = np.zeros(len(time_points))
    for i, t in enumerate(time_points):
        try:
            state = solution.get_state(t)
            means[i] = np.mean(state.radii)
        except ValueError:
            means[i] = np.nan
    return means

def compute_radius_kde_difference(solution_a, solution_b, t, n_grid=500):
    """
    Разница KDE распределений радиусов между двумя решениями
    в момент времени t (L1 норма).
    """
    try:
        state_a = solution_a.get_state(t)
        state_b = solution_b.get_state(t)
    except ValueError:
        return np.nan

    radii_a = state_a.radii
    radii_b = state_b.radii

    if len(radii_a) < 2 or len(radii_b) < 2:
        return np.nan

    kde_a = gaussian_kde(radii_a)
    kde_b = gaussian_kde(radii_b)

    all_radii = np.concatenate([radii_a, radii_b])
    r_grid = np.linspace(all_radii.min(), all_radii.max(), n_grid)
    dr = r_grid[1] - r_grid[0]

    pdf_a = kde_a(r_grid)
    pdf_b = kde_b(r_grid)

    return np.sum(np.abs(pdf_a - pdf_b)) * dr

def main():
    print('\n' * 3)

    # === Параметры ===
    num_particles = 10000
    water_volume_content = 0.02
    radii_range = np.array([2.5e-6, 7.5e-6])

    box_size = np.cbrt(
        (np.pi * num_particles * np.sum(radii_range) * np.sum(np.square(radii_range)))
        / (3 * water_volume_content)
    )
    coord_range = (0, box_size)

    t_stop = 100
    boundary_mode = "open"

    # Физические параметры
    eps_oil = 2.85
    eta_oil = 0.065
    eta_water = 0.001
    rho_water = 1000
    rho_oil = 900
    E = 3e5

    # Шаги для исследования сходимости
    dt_values = [0.1, 0.05, 0.04, 0.02, 0.01, 0.005]

    # === Загрузка или генерация начальных условий ===
    initial_data_file = os.path.join(PROJECT_ROOT, f"results/droplets_N{num_particles}_vol{water_volume_content}_0.xlsx")

    try:
        initial_state = DropletState(filename=initial_data_file)
        print(f"Начальные условия загружены из {initial_data_file}")
    except Exception:
        print("Файл начальных условий не найден, генерируем новые...")
        particle_generator = UniformDropletGenerator(
            coord_range=coord_range, radii_range=radii_range,
            num_particles=num_particles, minimum_distance=1e-6
        )
        initial_state = particle_generator.generate()
        initial_state.export_to_xlsx(initial_data_file)
        print(f"Начальные условия сохранены в {initial_data_file}")

    num_particles = len(initial_state.radii)

    # === Запуск симуляций ===
    solutions = {}
    wall_times = {}

    for dt in dt_values:
        t0 = time.time()
        sol = run_simulation(
            initial_state, dt, t_stop, box_size,
            eps_oil, eta_oil, eta_water, rho_water, rho_oil, E,
            boundary_mode
        )
        wall_times[dt] = time.time() - t0
        solutions[dt] = sol

        # Сохраняем результат
        stamp = f'convergence_N{num_particles}_dt{dt}'
        results_filename = os.path.join(PROJECT_ROOT, f"results/results_{stamp}.npz")
        sol.save_chain_to_file(results_filename, precision='float32')

    # === Анализ сходимости ===
    print(f"\n{'='*60}")
    print("  Анализ сходимости")
    print(f"{'='*60}")

    # Временные точки для анализа
    time_points = np.arange(0, t_stop + 1, 1.0)

    # Собираем метрики
    droplet_counts = {}
    mean_radii = {}
    for dt in dt_values:
        droplet_counts[dt] = get_droplet_count_over_time(solutions[dt], time_points)
        mean_radii[dt] = get_mean_radius_over_time(solutions[dt], time_points)

    # Эталонное решение (самый мелкий шаг)
    dt_ref = min(dt_values)
    ref_solution = solutions[dt_ref]

    # Разность KDE между каждым решением и эталоном в финальный момент
    kde_diffs = {}
    for dt in dt_values:
        if dt == dt_ref:
            kde_diffs[dt] = 0.0
        else:
            kde_diffs[dt] = compute_radius_kde_difference(
                solutions[dt], ref_solution, t_stop
            )

    # Относительная разность количества капель в финальный момент
    count_diffs = {}
    ref_count = droplet_counts[dt_ref][-1]
    for dt in dt_values:
        count_diffs[dt] = abs(droplet_counts[dt][-1] - ref_count) / ref_count * 100

    # Относительная разность среднего радиуса в финальный момент
    radius_diffs = {}
    ref_mean_r = mean_radii[dt_ref][-1]
    for dt in dt_values:
        radius_diffs[dt] = abs(mean_radii[dt][-1] - ref_mean_r) / ref_mean_r * 100

    # === Вывод таблицы ===
    print(f"\n{'dt':>10} | {'N капель':>10} | {'dN, %':>8} | {'<r>':>12} | {'d<r>, %':>8} | {'KDE diff':>10} | {'t, сек':>8}")
    print("-" * 85)
    for dt in dt_values:
        n = int(droplet_counts[dt][-1])
        dn = count_diffs[dt]
        mr = mean_radii[dt][-1]
        dr = radius_diffs[dt]
        kd = kde_diffs[dt]
        wt = wall_times[dt]
        print(f"{dt:>10.4f} | {n:>10d} | {dn:>7.2f}% | {mr:>12.6e} | {dr:>7.2f}% | {kd:>10.6f} | {wt:>7.1f}")

    # === Графики ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Анализ сходимости по dt (N={num_particles}, t_stop={t_stop})', fontsize=14)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(dt_values)))

    # 1. Количество капель от времени для разных dt
    ax = axes[0, 0]
    for i, dt in enumerate(dt_values):
        ax.plot(time_points, droplet_counts[dt], label=f'dt={dt}', color=colors[i], linewidth=1.5)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Количество капель')
    ax.set_title('Количество капель от времени')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Средний радиус от времени для разных dt
    ax = axes[0, 1]
    for i, dt in enumerate(dt_values):
        ax.plot(time_points, mean_radii[dt], label=f'dt={dt}', color=colors[i], linewidth=1.5)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Средний радиус')
    ax.set_title('Средний радиус от времени')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. KDE распределения радиусов в финальный момент
    ax = axes[0, 2]
    for i, dt in enumerate(dt_values):
        try:
            state = solutions[dt].get_state(t_stop)
            radii = state.radii
            if len(radii) >= 2:
                kde = gaussian_kde(radii)
                r_grid = np.linspace(radii.min(), radii.max(), 300)
                ax.plot(r_grid, kde(r_grid), label=f'dt={dt}', color=colors[i], linewidth=1.5)
        except ValueError:
            pass
    ax.set_xlabel('Радиус')
    ax.set_ylabel('Плотность вероятности')
    ax.set_title(f'Распределение радиусов (t={t_stop})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Сходимость: относительная ошибка количества капель vs dt
    ax = axes[1, 0]
    dts = [dt for dt in dt_values if dt != dt_ref]
    dn_vals = [count_diffs[dt] for dt in dts]
    ax.plot(dts, dn_vals, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.set_xlabel('dt')
    ax.set_ylabel('Относительная ошибка, %')
    ax.set_title(f'Ошибка N капель (эталон: dt={dt_ref})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # 5. Сходимость: относительная ошибка среднего радиуса vs dt
    ax = axes[1, 1]
    dr_vals = [radius_diffs[dt] for dt in dts]
    ax.plot(dts, dr_vals, 'o-', color='darkorange', linewidth=2, markersize=8)
    ax.set_xlabel('dt')
    ax.set_ylabel('Относительная ошибка, %')
    ax.set_title(f'Ошибка среднего радиуса (эталон: dt={dt_ref})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # 6. KDE L1 разность и время расчёта
    ax = axes[1, 2]
    kd_vals = [kde_diffs[dt] for dt in dts]
    ax.plot(dts, kd_vals, 'o-', color='crimson', linewidth=2, markersize=8, label='KDE L1 diff')
    ax.set_xlabel('dt')
    ax.set_ylabel('KDE L1 разность')
    ax.set_title(f'Разность распределений (эталон: dt={dt_ref})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Время расчёта на вторичной оси
    ax2 = ax.twinx()
    wt_vals = [wall_times[dt] for dt in dt_values]
    ax2.bar([str(dt) for dt in dt_values], wt_vals, alpha=0.2, color='gray', label='Время расчёта')
    ax2.set_ylabel('Время расчёта, сек')

    fig.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, 'results/convergence_study.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nГрафик сохранён: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
