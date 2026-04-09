"""
Исследование ошибки метода Эйлера для симуляции капель.

Для каждого dt из списка [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025] запускается
NUM_RUNS = 50 симуляций с одинаковым начальным распределением капель.
Результаты усредняются, выводится таблица и графики каждые 10 секунд симуляционного
времени: среднее количество капель, средний радиус, распределение по радиусам.

Запуск: python euler_method_error.py
"""

import sys
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from collision_detector import SpatialHashCollisionDetector
from force_calculator import DirectDropletForceCalculator
from particle_generator import UniformDropletGenerator
from particle_state import DropletState
from post_processor import DropletPostProcessor
from solution import DropletSolution
from solver import EulerDropletSolver

import taichi as ti

ti.init(arch=ti.cpu, cpu_max_num_threads=16, default_fp=ti.f64)

# ============================================================
#  Параметры исследования
# ============================================================
DT_VALUES = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]
NUM_RUNS = 50
T_STOP = 100.0                       # полное время симуляции
SNAPSHOT_INTERVAL = 10.0              # интервал снятия метрик (сек)
NUM_PARTICLES = 10000
WATER_VOLUME_CONTENT = 0.02
RADII_RANGE = np.array([2.5e-6, 7.5e-6])
BOUNDARY_MODE = "open"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/euler_error")

# Физические параметры
EPS_OIL = 2.85
ETA_OIL = 0.065
ETA_WATER = 0.001
RHO_WATER = 1000
RHO_OIL = 900
E_FIELD = 3e5


# ============================================================
#  Вспомогательные функции
# ============================================================

def compute_box_size(num_particles, radii_range, water_volume_content):
    return np.cbrt(
        (np.pi * num_particles * np.sum(radii_range) * np.sum(np.square(radii_range)))
        / (3 * water_volume_content)
    )


def run_single_simulation(initial_state, dt, t_stop, box_size):
    """Запускает одну симуляцию, возвращает объект solution (цепочку)."""
    state_copy = initial_state.copy()
    num_particles = len(state_copy.radii)

    force_calculator = DirectDropletForceCalculator(
        num_particles=num_particles, eps_oil=EPS_OIL,
        eta_oil=ETA_OIL, eta_water=ETA_WATER,
        rho_water=RHO_WATER, rho_oil=RHO_OIL, E=E_FIELD,
        L=box_size, boundary_mode=BOUNDARY_MODE
    )

    collision_detector = SpatialHashCollisionDetector(
        num_particles=num_particles, L=box_size,
        boundary_mode=BOUNDARY_MODE
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

    solver.solve(dt, t_stop)
    return solver.solution


def extract_metrics(solution, time_points):
    """
    Извлекает количество капель и радиусы для заданных моментов времени.

    Возвращает:
        counts  — массив int, количество капель в каждый момент
        radii_lists — список массивов радиусов для каждого момента
    """
    counts = np.zeros(len(time_points), dtype=int)
    radii_lists = []
    for i, t in enumerate(time_points):
        try:
            state = solution.get_state(t)
            counts[i] = len(state.radii)
            radii_lists.append(state.radii.copy())
        except ValueError:
            counts[i] = 0
            radii_lists.append(np.array([]))
    return counts, radii_lists


# ============================================================
#  Основной скрипт
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    box_size = compute_box_size(NUM_PARTICLES, RADII_RANGE, WATER_VOLUME_CONTENT)
    coord_range = (0, box_size)

    # Временные точки для снятия метрик (каждые 10 секунд)
    time_points = np.arange(0, T_STOP + SNAPSHOT_INTERVAL, SNAPSHOT_INTERVAL)

    # --- Загрузка или генерация ФИКСИРОВАННОГО начального состояния ---
    initial_data_file = os.path.join(PROJECT_ROOT, f"results/droplets_N{NUM_PARTICLES}_vol{WATER_VOLUME_CONTENT}_0.xlsx")
    try:
        initial_state = DropletState(filename=initial_data_file)
        print(f"Начальные условия загружены из {initial_data_file}")
    except Exception:
        print("Генерируем начальные условия...")
        gen = UniformDropletGenerator(
            coord_range=coord_range, radii_range=RADII_RANGE,
            num_particles=NUM_PARTICLES, minimum_distance=1e-6
        )
        initial_state = gen.generate()
        initial_state.export_to_xlsx(initial_data_file)
        print(f"Начальные условия сохранены в {initial_data_file}")

    actual_num_particles = len(initial_state.radii)
    print(f"Число частиц: {actual_num_particles}")
    print(f"Размер области: {box_size:.6e}")
    print(f"Значения dt: {DT_VALUES}")
    print(f"Число запусков для усреднения: {NUM_RUNS}")
    print(f"Временные точки: {time_points}")

    # ============================================================
    #  Цикл по dt → запуски → сбор метрик
    # ============================================================

    # Структуры для хранения результатов
    # all_counts[dt] — массив (NUM_RUNS, len(time_points))
    # all_radii[dt]  — список списков массивов радиусов
    all_counts = {}
    all_mean_radii = {}
    all_std_radii = {}
    all_radii_at_end = {}   # для финального KDE
    wall_times = {}

    for dt in DT_VALUES:
        print(f"\n{'='*70}")
        print(f"  dt = {dt}  |  {NUM_RUNS} запусков")
        print(f"{'='*70}")

        counts_matrix = np.zeros((NUM_RUNS, len(time_points)), dtype=int)
        mean_radii_matrix = np.zeros((NUM_RUNS, len(time_points)))
        std_radii_matrix = np.zeros((NUM_RUNS, len(time_points)))
        radii_at_end_runs = []

        dt_wall_start = time.time()

        for run_idx in range(NUM_RUNS):
            t0 = time.time()
            solution = run_single_simulation(initial_state, dt, T_STOP, box_size)
            elapsed = time.time() - t0

            counts, radii_lists = extract_metrics(solution, time_points)
            counts_matrix[run_idx] = counts

            for j, radii in enumerate(radii_lists):
                if len(radii) > 0:
                    mean_radii_matrix[run_idx, j] = np.mean(radii)
                    std_radii_matrix[run_idx, j] = np.std(radii)

            # Сохраняем радиусы в последний момент для KDE
            if len(radii_lists[-1]) > 0:
                radii_at_end_runs.append(radii_lists[-1])

            print(f"  Запуск {run_idx + 1}/{NUM_RUNS}: "
                  f"N_end={counts[-1]}, <r>_end={mean_radii_matrix[run_idx, -1]:.4e}, "
                  f"время={elapsed:.1f} сек")

        dt_wall_total = time.time() - dt_wall_start
        wall_times[dt] = dt_wall_total

        all_counts[dt] = counts_matrix
        all_mean_radii[dt] = mean_radii_matrix
        all_std_radii[dt] = std_radii_matrix
        all_radii_at_end[dt] = radii_at_end_runs

    # ============================================================
    #  Вывод сводной таблицы
    # ============================================================

    print(f"\n\n{'='*100}")
    print("  СВОДНАЯ ТАБЛИЦА: средние значения по {NUM_RUNS} запускам")
    print(f"{'='*100}")

    # Заголовок таблицы
    header = f"{'t, сек':>8}"
    for dt in DT_VALUES:
        header += f" | {'dt=' + str(dt):>18}"
    print(header)
    print("-" * len(header))

    # Таблица количества капель
    print("\n--- Среднее количество капель ---")
    header = f"{'t, сек':>8}"
    for dt in DT_VALUES:
        header += f" | {'N±σ':>18}"
    print(header)
    print("-" * len(header))

    for j, t in enumerate(time_points):
        row = f"{t:>8.1f}"
        for dt in DT_VALUES:
            mean_n = np.mean(all_counts[dt][:, j])
            std_n = np.std(all_counts[dt][:, j])
            row += f" | {mean_n:>8.1f}±{std_n:>6.1f}"
        print(row)

    # Таблица среднего радиуса
    print("\n--- Средний радиус капель ---")
    header = f"{'t, сек':>8}"
    for dt in DT_VALUES:
        header += f" | {'<r>±σ':>18}"
    print(header)
    print("-" * len(header))

    for j, t in enumerate(time_points):
        row = f"{t:>8.1f}"
        for dt in DT_VALUES:
            mean_r = np.mean(all_mean_radii[dt][:, j])
            std_r = np.std(all_mean_radii[dt][:, j])
            row += f" | {mean_r:>8.3e}±{std_r:>6.2e}"
        print(row)

    # Таблица относительных ошибок (эталон = наименьший dt)
    dt_ref = min(DT_VALUES)
    print(f"\n--- Относительные ошибки (эталон: dt={dt_ref}) ---")
    header = f"{'t, сек':>8}"
    for dt in DT_VALUES:
        if dt == dt_ref:
            header += f" | {'(эталон)':>18}"
        else:
            header += f" | {'δN, %':>8} {'δ<r>, %':>8}"
    print(header)
    print("-" * len(header))

    ref_mean_counts = np.mean(all_counts[dt_ref], axis=0)
    ref_mean_radii = np.mean(all_mean_radii[dt_ref], axis=0)

    for j, t in enumerate(time_points):
        row = f"{t:>8.1f}"
        for dt in DT_VALUES:
            if dt == dt_ref:
                row += f" | {'—':>18}"
            else:
                mean_n = np.mean(all_counts[dt][:, j])
                mean_r = np.mean(all_mean_radii[dt][:, j])
                if ref_mean_counts[j] > 0:
                    err_n = abs(mean_n - ref_mean_counts[j]) / ref_mean_counts[j] * 100
                else:
                    err_n = 0.0
                if ref_mean_radii[j] > 0:
                    err_r = abs(mean_r - ref_mean_radii[j]) / ref_mean_radii[j] * 100
                else:
                    err_r = 0.0
                row += f" | {err_n:>7.2f}% {err_r:>7.2f}%"
        print(row)

    # Сводка: финальные метрики
    print(f"\n--- Итоговая сводка (t = {T_STOP}) ---")
    print(f"{'dt':>8} | {'<N>':>8} | {'σ(N)':>8} | {'<r>':>12} | {'σ(<r>)':>10} | "
          f"{'δN, %':>8} | {'δ<r>, %':>8} | {'Время, сек':>12}")
    print("-" * 95)

    for dt in DT_VALUES:
        mean_n = np.mean(all_counts[dt][:, -1])
        std_n = np.std(all_counts[dt][:, -1])
        mean_r = np.mean(all_mean_radii[dt][:, -1])
        std_r = np.std(all_mean_radii[dt][:, -1])

        if dt == dt_ref:
            err_n_str = "—"
            err_r_str = "—"
        else:
            err_n = abs(mean_n - ref_mean_counts[-1]) / ref_mean_counts[-1] * 100
            err_r = abs(mean_r - ref_mean_radii[-1]) / ref_mean_radii[-1] * 100
            err_n_str = f"{err_n:.2f}%"
            err_r_str = f"{err_r:.2f}%"

        print(f"{dt:>8.4f} | {mean_n:>8.1f} | {std_n:>8.1f} | {mean_r:>12.4e} | {std_r:>10.2e} | "
              f"{err_n_str:>8} | {err_r_str:>8} | {wall_times[dt]:>11.1f}")

    # ============================================================
    #  Графики
    # ============================================================

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(DT_VALUES)))

    # --- Рисунок 1: Среднее количество капель и средний радиус ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle(f'Метод Эйлера: усреднение по {NUM_RUNS} запускам (N₀={actual_num_particles})', fontsize=14)

    ax = axes1[0]
    for i, dt in enumerate(DT_VALUES):
        mean_n = np.mean(all_counts[dt], axis=0).astype(float)
        std_n = np.std(all_counts[dt], axis=0).astype(float)
        ax.plot(time_points, mean_n, label=f'dt={dt}', color=colors[i], linewidth=1.5)
        ax.fill_between(time_points, mean_n - std_n, mean_n + std_n,
                        color=colors[i], alpha=0.15)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Количество капель')
    ax.set_title('Среднее количество капель')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes1[1]
    for i, dt in enumerate(DT_VALUES):
        mean_r = np.mean(all_mean_radii[dt], axis=0)
        std_r = np.std(all_mean_radii[dt], axis=0)
        ax.plot(time_points, mean_r, label=f'dt={dt}', color=colors[i], linewidth=1.5)
        ax.fill_between(time_points, mean_r - std_r, mean_r + std_r,
                        color=colors[i], alpha=0.15)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Средний радиус, м')
    ax.set_title('Средний радиус капель')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(f'{RESULTS_DIR}/euler_error_counts_radii.png', dpi=300, bbox_inches='tight')
    print(f"\nГрафик сохранён: {RESULTS_DIR}/euler_error_counts_radii.png")

    # --- Рисунок 2: KDE распределений радиусов в финальный момент ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_title(f'Распределение радиусов при t={T_STOP} (усреднённое KDE, {NUM_RUNS} запусков)')

    for i, dt in enumerate(DT_VALUES):
        # Объединяем все радиусы из всех запусков
        all_r = np.concatenate(all_radii_at_end[dt]) if all_radii_at_end[dt] else np.array([])
        if len(all_r) >= 2:
            kde = gaussian_kde(all_r)
            r_grid = np.linspace(all_r.min(), all_r.max(), 500)
            ax2.plot(r_grid, kde(r_grid), label=f'dt={dt}', color=colors[i], linewidth=1.5)
    ax2.set_xlabel('Радиус, м')
    ax2.set_ylabel('Плотность вероятности')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(f'{RESULTS_DIR}/euler_error_kde.png', dpi=300, bbox_inches='tight')
    print(f"График сохранён: {RESULTS_DIR}/euler_error_kde.png")

    # --- Рисунок 3: Ошибки относительно эталонного dt ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle(f'Ошибки метода Эйлера (эталон: dt={dt_ref}, {NUM_RUNS} запусков)', fontsize=14)

    dts_no_ref = [dt for dt in DT_VALUES if dt != dt_ref]

    # Ошибка количества капель при t=T_STOP
    ax = axes3[0]
    err_n_vals = []
    for dt in dts_no_ref:
        mean_n = np.mean(all_counts[dt][:, -1])
        err_n_vals.append(abs(mean_n - ref_mean_counts[-1]) / ref_mean_counts[-1] * 100)
    ax.plot(dts_no_ref, err_n_vals, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.set_xlabel('dt')
    ax.set_ylabel('δN, %')
    ax.set_title(f'Ошибка количества капель (t={T_STOP})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Ошибка среднего радиуса при t=T_STOP
    ax = axes3[1]
    err_r_vals = []
    for dt in dts_no_ref:
        mean_r = np.mean(all_mean_radii[dt][:, -1])
        err_r_vals.append(abs(mean_r - ref_mean_radii[-1]) / ref_mean_radii[-1] * 100)
    ax.plot(dts_no_ref, err_r_vals, 'o-', color='darkorange', linewidth=2, markersize=8)
    ax.set_xlabel('dt')
    ax.set_ylabel('δ<r>, %')
    ax.set_title(f'Ошибка среднего радиуса (t={T_STOP})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Время расчёта
    ax = axes3[2]
    ax.bar([str(dt) for dt in DT_VALUES],
           [wall_times[dt] / NUM_RUNS for dt in DT_VALUES],
           color='gray', alpha=0.7)
    ax.set_xlabel('dt')
    ax.set_ylabel('Среднее время одного запуска, сек')
    ax.set_title('Производительность')
    ax.grid(True, alpha=0.3, axis='y')

    fig3.tight_layout()
    fig3.savefig(f'{RESULTS_DIR}/euler_error_convergence.png', dpi=300, bbox_inches='tight')
    print(f"График сохранён: {RESULTS_DIR}/euler_error_convergence.png")

    # --- Сохраняем сырые данные для воспроизводимости ---
    np.savez(
        f'{RESULTS_DIR}/euler_error_raw_data.npz',
        dt_values=DT_VALUES,
        time_points=time_points,
        num_runs=NUM_RUNS,
        **{f'counts_dt{dt}': all_counts[dt] for dt in DT_VALUES},
        **{f'mean_radii_dt{dt}': all_mean_radii[dt] for dt in DT_VALUES},
        **{f'std_radii_dt{dt}': all_std_radii[dt] for dt in DT_VALUES},
    )
    print(f"Сырые данные сохранены: {RESULTS_DIR}/euler_error_raw_data.npz")

    print(f"\nГотово!")


if __name__ == "__main__":
    main()
