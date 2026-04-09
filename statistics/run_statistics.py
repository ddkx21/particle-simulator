"""
Сбор статистики по NUM_RUNS реализациям: direct vs tree метод.

Запуск: python run_statistics.py
"""

import sys
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

from collision_detector import SpatialHashCollisionDetector
from force_calculator import DirectDropletForceCalculator
from octree.force_tree import TreeDropletForceCalculator
from particle_generator import UniformDropletGenerator
from particle_state import DropletState
from post_processor import DropletPostProcessor
from solution import DropletSolution
from solver import EulerDropletSolver

import taichi as ti

ti.init(arch=ti.cpu, cpu_max_num_threads=16, default_fp=ti.f64)

# ── Параметры ────────────────────────────────────────────────
N = 100
dt = 0.04
t_stop = 100
save_interval = 250          # каждые 10 сек при dt=0.04
NUM_RUNS = 2

snapshot_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 11 точек
histogram_times = [0, 20, 40, 60, 80, 100]                       # 6 точек для субплотов

radii_range = np.array([2.5e-6, 7.5e-6])
water_volume_content = 0.02

box_size = np.cbrt(
    (np.pi * N * np.sum(radii_range) * np.sum(np.square(radii_range)))
    / (3 * water_volume_content)
)
coord_range = (0, box_size)

# Физические параметры
eps_oil = 2.85
eta_oil = 0.065
eta_water = 0.001
rho_water = 1000
rho_oil = 900
E = 3e5

boundary_mode = "open"

# Tree параметры
theta = 0.25
mpl = 16

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Одна симуляция ───────────────────────────────────────────
def run_single_simulation(method, run_index):
    """Запуск одной симуляции, возврат агрегированной статистики."""
    # 1. Генерация новой системы
    generator = UniformDropletGenerator(
        coord_range=coord_range,
        radii_range=radii_range,
        num_particles=N,
        minimum_distance=1e-6,
    )
    initial_state = generator.generate()
    num_particles = len(initial_state.radii)

    # 2. Force calculator
    if method == "direct":
        force_calculator = DirectDropletForceCalculator(
            num_particles=num_particles,
            eps_oil=eps_oil, eta_oil=eta_oil, eta_water=eta_water,
            rho_water=rho_water, rho_oil=rho_oil, E=E,
            L=box_size, boundary_mode=boundary_mode,
        )
    else:
        force_calculator = TreeDropletForceCalculator(
            num_particles=num_particles,
            theta=theta, mpl=mpl,
            eps_oil=eps_oil, eta_oil=eta_oil, eta_water=eta_water,
            E=E, L=box_size,
            periodic=(boundary_mode == "periodic"),
        )

    # 3. Остальные компоненты
    collision_detector = SpatialHashCollisionDetector(
        num_particles=num_particles, L=box_size, boundary_mode=boundary_mode,
    )
    solution = DropletSolution(
        initial_droplet_state=initial_state,
        real_time_visualization=False,
    )
    post_processor = DropletPostProcessor(solution, box_size=box_size)
    solver = EulerDropletSolver(
        force_calculator=force_calculator,
        solution=solution,
        post_processor=post_processor,
        collision_detector=collision_detector,
        save_interval=save_interval,
    )

    # 4. Запуск с замером времени
    t_start = time.perf_counter()
    solver.solve(dt, t_stop)
    elapsed = time.perf_counter() - t_start

    # 5. Извлечение статистики по snapshot_times
    droplet_counts = []
    median_radii = []
    radii_snapshots = {}

    for t in snapshot_times:
        state = solution.get_state(float(t))
        n_drops = len(state.radii)
        droplet_counts.append(n_drops)
        median_radii.append(float(np.median(state.radii)))
        if t in histogram_times:
            radii_snapshots[t] = state.radii.copy()

    return {
        "elapsed_time": elapsed,
        "droplet_counts": droplet_counts,
        "median_radii": median_radii,
        "radii_snapshots": radii_snapshots,
    }


# ── Сохранение промежуточных результатов ─────────────────────
def save_phase_results(results, filename):
    """Сохранение результатов фазы в .npz."""
    elapsed_times = np.array([r["elapsed_time"] for r in results])
    droplet_counts = np.array([r["droplet_counts"] for r in results])
    median_radii = np.array([r["median_radii"] for r in results])

    save_dict = {
        "elapsed_times": elapsed_times,
        "droplet_counts": droplet_counts,
        "median_radii": median_radii,
        "snapshot_times": np.array(snapshot_times),
    }

    # Радиусы для гистограммных точек
    for t in histogram_times:
        arrays = [r["radii_snapshots"][t] for r in results]
        # Разные реализации могут иметь разное число капель — сохраняем как object array
        for i, arr in enumerate(arrays):
            save_dict[f"radii_at_t{t}_run{i}"] = arr

    np.savez(filename, **save_dict)
    print(f"Результаты сохранены: {filename}")


# ── Графики ──────────────────────────────────────────────────
def plot_all_statistics(direct_results, tree_results):
    """Построение 4 графиков сравнения."""
    t_arr = np.array(snapshot_times)

    direct_counts = np.array([r["droplet_counts"] for r in direct_results])  # (50, 11)
    tree_counts = np.array([r["droplet_counts"] for r in tree_results])
    direct_medians = np.array([r["median_radii"] for r in direct_results])
    tree_medians = np.array([r["median_radii"] for r in tree_results])

    # ── График 1: N(t) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(NUM_RUNS):
        ax.plot(t_arr, direct_counts[i], color="tab:blue", alpha=0.1, linewidth=0.5)
        ax.plot(t_arr, tree_counts[i], color="tab:red", alpha=0.1, linewidth=0.5)

    d_mean = direct_counts.mean(axis=0)
    d_std = direct_counts.std(axis=0)
    t_mean = tree_counts.mean(axis=0)
    t_std = tree_counts.std(axis=0)

    ax.plot(t_arr, d_mean, color="tab:blue", linewidth=2, label="Direct (mean)")
    ax.fill_between(t_arr, d_mean - d_std, d_mean + d_std, color="tab:blue", alpha=0.2)
    ax.plot(t_arr, t_mean, color="tab:red", linewidth=2, label="Tree (mean)")
    ax.fill_between(t_arr, t_mean - t_std, t_mean + t_std, color="tab:red", alpha=0.2)

    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Число капель N(t)")
    ax.set_title(f"Эволюция числа капель (N₀={N}, {NUM_RUNS} реализаций)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "statistics_droplet_count.png"), dpi=150)
    plt.close(fig)
    print(f"Сохранён: {RESULTS_DIR}/statistics_droplet_count.png")

    # ── График 2: Медианный радиус ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(NUM_RUNS):
        ax.plot(t_arr, direct_medians[i] * 1e6, color="tab:blue", alpha=0.1, linewidth=0.5)
        ax.plot(t_arr, tree_medians[i] * 1e6, color="tab:red", alpha=0.1, linewidth=0.5)

    d_mean = direct_medians.mean(axis=0) * 1e6
    d_std = direct_medians.std(axis=0) * 1e6
    t_mean = tree_medians.mean(axis=0) * 1e6
    t_std = tree_medians.std(axis=0) * 1e6

    ax.plot(t_arr, d_mean, color="tab:blue", linewidth=2, label="Direct (mean)")
    ax.fill_between(t_arr, d_mean - d_std, d_mean + d_std, color="tab:blue", alpha=0.2)
    ax.plot(t_arr, t_mean, color="tab:red", linewidth=2, label="Tree (mean)")
    ax.fill_between(t_arr, t_mean - t_std, t_mean + t_std, color="tab:red", alpha=0.2)

    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Медианный радиус, мкм")
    ax.set_title(f"Эволюция медианного радиуса (N₀={N}, {NUM_RUNS} реализаций)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "statistics_median_radius.png"), dpi=150)
    plt.close(fig)
    print(f"Сохранён: {RESULTS_DIR}/statistics_median_radius.png")

    # ── График 3: Распределение по радиусам (2x3 subplots) ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for idx, t in enumerate(histogram_times):
        ax = axes_flat[idx]

        direct_radii_list = [r["radii_snapshots"][t] * 1e6 for r in direct_results]
        tree_radii_list = [r["radii_snapshots"][t] * 1e6 for r in tree_results]

        # Общий диапазон бинов
        all_radii = np.concatenate(direct_radii_list + tree_radii_list)
        bins = np.linspace(all_radii.min(), all_radii.max(), 31)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Усреднённые гистограммы по 50 реализациям
        direct_hists = np.array([
            np.histogram(r, bins=bins, density=True)[0] for r in direct_radii_list
        ])
        tree_hists = np.array([
            np.histogram(r, bins=bins, density=True)[0] for r in tree_radii_list
        ])

        d_mean_h = direct_hists.mean(axis=0)
        d_std_h = direct_hists.std(axis=0)
        t_mean_h = tree_hists.mean(axis=0)
        t_std_h = tree_hists.std(axis=0)

        ax.plot(bin_centers, d_mean_h, color="tab:blue", linewidth=1.5, label="Direct")
        ax.fill_between(bin_centers, d_mean_h - d_std_h, d_mean_h + d_std_h,
                         color="tab:blue", alpha=0.2)
        ax.plot(bin_centers, t_mean_h, color="tab:red", linewidth=1.5, label="Tree")
        ax.fill_between(bin_centers, t_mean_h - t_std_h, t_mean_h + t_std_h,
                         color="tab:red", alpha=0.2)

        ax.set_title(f"t = {t} сек")
        ax.set_xlabel("Радиус, мкм")
        ax.set_ylabel("Плотность")
        if idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Распределение по радиусам (N₀={N}, {NUM_RUNS} реализаций)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "statistics_radii_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"Сохранён: {RESULTS_DIR}/statistics_radii_distribution.png")

    # ── График 4: Сравнение времени расчёта ──
    direct_times = np.array([r["elapsed_time"] for r in direct_results])
    tree_times = np.array([r["elapsed_time"] for r in tree_results])

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [direct_times, tree_times],
        labels=["Direct", "Tree"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("tab:blue")
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("tab:red")
    bp["boxes"][1].set_alpha(0.5)

    ax.set_ylabel("Время расчёта, сек")
    ax.set_title(f"Время расчёта: Direct vs Tree (N₀={N}, t_stop={t_stop})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "statistics_timing.png"), dpi=150)
    plt.close(fig)
    print(f"Сохранён: {RESULTS_DIR}/statistics_timing.png")

    # Печать статистики
    print(f"\nDirect: {direct_times.mean():.1f} ± {direct_times.std():.1f} сек  "
          f"(min={direct_times.min():.1f}, max={direct_times.max():.1f})")
    print(f"Tree:   {tree_times.mean():.1f} ± {tree_times.std():.1f} сек  "
          f"(min={tree_times.min():.1f}, max={tree_times.max():.1f})")
    speedup = direct_times.mean() / tree_times.mean()
    print(f"Ускорение tree/direct: {speedup:.2f}x")


# ── Главный цикл ────────────────────────────────────────────
def main():
    print(f"Параметры: N={N}, dt={dt}, t_stop={t_stop}, save_interval={save_interval}")
    print(f"box_size={box_size:.6e}, boundary_mode={boundary_mode}")
    print(f"Tree: theta={theta}, mpl={mpl}")
    print(f"Число реализаций: {NUM_RUNS}\n")

    # --- Фаза 1: Direct метод ---
    print("=" * 60)
    print("ФАЗА 1: Direct метод")
    print("=" * 60)
    direct_results = []
    for i in range(NUM_RUNS):
        result = run_single_simulation("direct", i)
        direct_results.append(result)
        print(f"Direct {i+1}/{NUM_RUNS} done, elapsed: {result['elapsed_time']:.1f} сек, "
              f"N_final={result['droplet_counts'][-1]}")

    save_phase_results(direct_results, os.path.join(RESULTS_DIR, "statistics_direct.npz"))

    # --- Фаза 2: Tree метод ---
    print("\n" + "=" * 60)
    print("ФАЗА 2: Tree метод")
    print("=" * 60)
    tree_results = []
    for i in range(NUM_RUNS):
        result = run_single_simulation("tree", i)
        tree_results.append(result)
        print(f"Tree {i+1}/{NUM_RUNS} done, elapsed: {result['elapsed_time']:.1f} сек, "
              f"N_final={result['droplet_counts'][-1]}")

    save_phase_results(tree_results, os.path.join(RESULTS_DIR, "statistics_tree.npz"))

    # --- Графики ---
    print("\n" + "=" * 60)
    print("Построение графиков...")
    print("=" * 60)
    plot_all_statistics(direct_results, tree_results)

    print("\nГотово!")


if __name__ == "__main__":
    main()
