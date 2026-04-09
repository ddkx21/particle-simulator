"""
Бенчмарк: periodic vs open граничные условия, direct vs tree метод.

Сравнивает 4 конфигурации:
  1. Direct + Open
  2. Direct + Periodic (+ COMSOL коррекция)
  3. Tree   + Open
  4. Tree   + Periodic (+ COMSOL коррекция)

Для каждого dt из DT_VALUES запускается NUM_RUNS симуляций.
Результаты сохраняются в results/boundary_benchmark/ как .npz файлы.

Запуск: python benchmarks/run_boundary_benchmark.py
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from collision_detector import SpatialHashCollisionDetector
from force_calculator import DirectDropletForceCalculator
from octree.force_tree import TreeDropletForceCalculator
from particle_generator import UniformDropletGenerator
from particle_state import DropletState
from post_processor import DropletPostProcessor
from solution import DropletSolution
from solver import EulerDropletSolver
from periodic_correction import COMSOLLatticeCorrection

import taichi as ti

ti.init(arch=ti.cpu, cpu_max_num_threads=16, default_fp=ti.f64)

# ============================================================
#  Параметры бенчмарка
# ============================================================

N = 100_000
DT_VALUES = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]
NUM_RUNS = 50
T_STOP = 100.0
SNAPSHOT_INTERVAL = 10.0

RADII_RANGE = np.array([2.5e-6, 7.5e-6])
WATER_VOLUME_CONTENT = 0.02

# Физические параметры
EPS_OIL = 2.85
ETA_OIL = 0.065
ETA_WATER = 0.001
RHO_WATER = 1000
RHO_OIL = 900
E_FIELD = 3e5

# Параметры дерева
THETA = 0.25
MPL = 2

# Конфигурации: (метод, граничные условия)
CONFIGS = [
    ("direct", "open"),
    ("direct", "periodic"),
    ("tree",   "open"),
    ("tree",   "periodic"),
]

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "boundary_benchmark")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
#  Вычисление box_size
# ============================================================

def compute_box_size(num_particles, radii_range, water_volume_content):
    return float(np.cbrt(
        (np.pi * num_particles * np.sum(radii_range) * np.sum(np.square(radii_range)))
        / (3 * water_volume_content)
    ))


BOX_SIZE = compute_box_size(N, RADII_RANGE, WATER_VOLUME_CONTENT)


# ============================================================
#  Загрузка COMSOL коррекции (один раз)
# ============================================================

LATTICE_CORRECTION = COMSOLLatticeCorrection.load_default()
CORR_RES = LATTICE_CORRECTION.grid_resolution


# ============================================================
#  Одна симуляция
# ============================================================

def run_single_simulation(method, boundary, dt):
    """
    Запускает одну симуляцию, возвращает словарь с метриками.
    """
    # 1. Генерация начального состояния
    generator = UniformDropletGenerator(
        coord_range=(0, BOX_SIZE),
        radii_range=RADII_RANGE,
        num_particles=N,
        minimum_distance=1e-6,
    )
    initial_state = generator.generate()
    num_particles = len(initial_state.radii)

    # 2. Force calculator
    is_periodic = (boundary == "periodic")

    if method == "direct":
        corr_res = CORR_RES if is_periodic else 0
        force_calculator = DirectDropletForceCalculator(
            num_particles=num_particles,
            eps_oil=EPS_OIL, eta_oil=ETA_OIL, eta_water=ETA_WATER,
            rho_water=RHO_WATER, rho_oil=RHO_OIL, E=E_FIELD,
            L=BOX_SIZE, boundary_mode=boundary,
            correction_grid_resolution=corr_res,
        )
        if is_periodic:
            force_calculator.load_periodic_correction(LATTICE_CORRECTION, L_sim=BOX_SIZE)
    else:
        corr_res = CORR_RES if is_periodic else 0
        force_calculator = TreeDropletForceCalculator(
            num_particles=num_particles,
            theta=THETA, mpl=MPL,
            eps_oil=EPS_OIL, eta_oil=ETA_OIL, eta_water=ETA_WATER,
            E=E_FIELD, L=BOX_SIZE, periodic=is_periodic,
            correction_grid_resolution=corr_res,
        )
        if is_periodic:
            force_calculator.load_periodic_correction(LATTICE_CORRECTION, L_sim=BOX_SIZE)

    # 3. Остальные компоненты
    collision_detector = SpatialHashCollisionDetector(
        num_particles=num_particles, L=BOX_SIZE, boundary_mode=boundary,
    )
    solution = DropletSolution(
        initial_droplet_state=initial_state,
        real_time_visualization=False,
    )
    post_processor = DropletPostProcessor(solution, box_size=BOX_SIZE)

    save_interval = max(int(SNAPSHOT_INTERVAL / dt), 1)
    solver = EulerDropletSolver(
        force_calculator=force_calculator,
        solution=solution,
        post_processor=post_processor,
        collision_detector=collision_detector,
        save_interval=save_interval,
    )

    # 4. Запуск
    t_start = time.perf_counter()
    solver.solve(dt, T_STOP)
    elapsed = time.perf_counter() - t_start

    # 5. Извлечение метрик
    snapshot_times = np.arange(0, T_STOP + SNAPSHOT_INTERVAL, SNAPSHOT_INTERVAL)
    droplet_counts = []
    mean_radii = []
    median_radii = []
    radii_snapshots = {}

    for t in snapshot_times:
        state = solver.solution.get_state(float(t))
        n_drops = len(state.radii)
        droplet_counts.append(n_drops)
        mean_radii.append(float(np.mean(state.radii)))
        median_radii.append(float(np.median(state.radii)))
        radii_snapshots[float(t)] = state.radii.copy()

    return {
        "elapsed_time": elapsed,
        "droplet_counts": droplet_counts,
        "mean_radii": mean_radii,
        "median_radii": median_radii,
        "radii_snapshots": radii_snapshots,
        "snapshot_times": snapshot_times,
    }


# ============================================================
#  Сохранение результатов
# ============================================================

def result_filename(method, boundary, dt):
    return os.path.join(RESULTS_DIR, f"{method}_{boundary}_dt{dt}.npz")


def save_results(results, method, boundary, dt):
    """Сохраняет список результатов в .npz."""
    snapshot_times = results[0]["snapshot_times"]

    elapsed_times = np.array([r["elapsed_time"] for r in results])
    droplet_counts = np.array([r["droplet_counts"] for r in results])
    mean_radii_arr = np.array([r["mean_radii"] for r in results])
    median_radii_arr = np.array([r["median_radii"] for r in results])

    save_dict = {
        "elapsed_times": elapsed_times,
        "droplet_counts": droplet_counts,
        "mean_radii": mean_radii_arr,
        "median_radii": median_radii_arr,
        "snapshot_times": snapshot_times,
        "method": np.array(method),
        "boundary": np.array(boundary),
        "dt": np.array(dt),
        "num_runs": np.array(len(results)),
        "N": np.array(N),
        "theta": np.array(THETA),
        "mpl": np.array(MPL),
    }

    # Радиусы для KDE
    for t_val in snapshot_times:
        for i, r in enumerate(results):
            key = f"radii_at_t{int(t_val)}_run{i}"
            save_dict[key] = r["radii_snapshots"][float(t_val)]

    fname = result_filename(method, boundary, dt)
    np.savez(fname, **save_dict)
    print(f"Сохранено: {fname}")


# ============================================================
#  Главный цикл
# ============================================================

def main():
    print("=" * 70)
    print("  Бенчмарк: periodic vs open, direct vs tree")
    print(f"  N={N}, dt={DT_VALUES}, NUM_RUNS={NUM_RUNS}, t_stop={T_STOP}")
    print(f"  Tree: theta={THETA}, mpl={MPL}")
    print(f"  box_size={BOX_SIZE:.6e}")
    print("=" * 70)

    total_configs = len(DT_VALUES) * len(CONFIGS)
    config_idx = 0

    for dt in DT_VALUES:
        for method, boundary in CONFIGS:
            config_idx += 1
            fname = result_filename(method, boundary, dt)

            # Пропускаем если уже посчитано
            if os.path.exists(fname):
                print(f"\n[{config_idx}/{total_configs}] {method}+{boundary} dt={dt} "
                      f"— уже существует, пропускаем")
                continue

            print(f"\n{'=' * 70}")
            print(f"  [{config_idx}/{total_configs}] {method}+{boundary}, dt={dt}")
            print(f"  {NUM_RUNS} запусков")
            print(f"{'=' * 70}")

            results = []

            for run_idx in range(NUM_RUNS):
                t0 = time.perf_counter()
                result = run_single_simulation(method, boundary, dt)
                run_time = time.perf_counter() - t0

                results.append(result)

                n_final = result["droplet_counts"][-1]
                print(f"  {method}+{boundary} dt={dt} "
                      f"run {run_idx + 1}/{NUM_RUNS}: "
                      f"N_end={n_final}, "
                      f"время={run_time:.1f} сек")

            save_results(results, method, boundary, dt)

    print("\n" + "=" * 70)
    print("  Бенчмарк завершён!")
    print("=" * 70)


if __name__ == "__main__":
    main()
