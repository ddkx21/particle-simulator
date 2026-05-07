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

from dem.collision_detector import SpatialHashCollisionDetector
from dem.force_calculator import DirectDropletForceCalculator
from dem.octree.force_tree import TreeDropletForceCalculator
from dem.particle_generator import UniformDropletGenerator
from dem.particle_state import DropletState
from dem.post_processor import DropletPostProcessor
from dem.solution import DropletSolution
from dem.solver import EulerDropletSolver
from dem.periodic_correction import COMSOLLatticeCorrection

import taichi as ti

ti.init(arch=ti.cpu, cpu_max_num_threads=16, default_fp=ti.f64)

# ============================================================
#  Параметры бенчмарка
# ============================================================

N = 100_000
DT_VALUES = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]
NUM_RUNS = 50
T_STOP = 100.0
SNAPSHOT_INTERVAL = 10.0        # Тяжёлые снимки (полные массивы радиусов для KDE)
LIGHT_METRIC_INTERVAL = 1.0     # Лёгкие метрики (N, mean_r, median_r) — каждую 1 сек

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

def extract_chain_metrics(solution):
    """
    Обходит цепочку solution и собирает лёгкие метрики на регулярной сетке.

    Каждый узел цепочки хранит num_particles и radii (меняются при столкновениях)
    и times (когда этот узел был активен). Обходим узлы и для каждой точки
    на сетке LIGHT_METRIC_INTERVAL определяем текущее число капель и статистику.

    Возвращает dict с ключами:
        light_times, light_droplet_counts, light_mean_radii, light_median_radii,
        light_max_radii, light_polydispersity, light_volume_ratio,
        light_cumulative_collisions
    """
    # Собираем все узлы цепочки: (t_start, t_end, num_particles, radii)
    first = solution
    while first._prev is not None:
        first = first._prev

    segments = []
    current = first
    while current is not None:
        times = current.get_times().flatten()
        if len(times) > 0:
            t_start = float(times[0])
            t_end = float(times[-1])
            segments.append((t_start, t_end, current.num_particles, current.radii))
        current = current._next

    # Начальный объём (из первого сегмента)
    initial_radii = segments[0][3]
    V0 = float(np.sum(initial_radii ** 3)) * (4.0 / 3.0 * np.pi)

    # Кумулятивные столкновения: каждый новый сегмент = событие столкновения
    # (кроме первого). Число слившихся капель = N_prev - N_current
    # Считаем кумулятивное число столкновений на границе каждого сегмента.
    collision_times = []     # время начала каждого нового сегмента
    collision_cumsum = []    # кумулятивное число столкновений к этому моменту
    cumulative = 0
    for idx in range(len(segments)):
        if idx > 0:
            n_prev = segments[idx - 1][2]
            n_curr = segments[idx][2]
            cumulative += (n_prev - n_curr)
        collision_times.append(segments[idx][0])
        collision_cumsum.append(cumulative)

    # Регулярная сетка для лёгких метрик
    light_times = np.arange(0, T_STOP + LIGHT_METRIC_INTERVAL * 0.5, LIGHT_METRIC_INTERVAL)
    n_points = len(light_times)

    light_counts = np.empty(n_points, dtype=np.int64)
    light_mean_r = np.empty(n_points, dtype=np.float64)
    light_median_r = np.empty(n_points, dtype=np.float64)
    light_max_r = np.empty(n_points, dtype=np.float64)
    light_polydispersity = np.empty(n_points, dtype=np.float64)
    light_volume_ratio = np.empty(n_points, dtype=np.float64)
    light_cum_collisions = np.empty(n_points, dtype=np.int64)

    seg_idx = 0
    coll_idx = 0
    for i, t in enumerate(light_times):
        # Находим подходящий сегмент для этого времени
        while seg_idx < len(segments) - 1 and t > segments[seg_idx][1]:
            seg_idx += 1

        _, _, n_particles, radii = segments[seg_idx]
        light_counts[i] = n_particles
        mean_r = float(np.mean(radii))
        light_mean_r[i] = mean_r
        light_median_r[i] = float(np.median(radii))
        light_max_r[i] = float(np.max(radii))

        # Полидисперсность: std(r) / mean(r)
        std_r = float(np.std(radii))
        light_polydispersity[i] = std_r / mean_r if mean_r > 0 else 0.0

        # Сохранение объёма: V(t) / V(0)
        V_t = float(np.sum(radii ** 3)) * (4.0 / 3.0 * np.pi)
        light_volume_ratio[i] = V_t / V0 if V0 > 0 else 1.0

        # Кумулятивные столкновения к моменту t
        while coll_idx < len(collision_times) - 1 and collision_times[coll_idx + 1] <= t:
            coll_idx += 1
        light_cum_collisions[i] = collision_cumsum[coll_idx]

    return {
        "light_times": light_times,
        "light_droplet_counts": light_counts,
        "light_mean_radii": light_mean_r,
        "light_median_radii": light_median_r,
        "light_max_radii": light_max_r,
        "light_polydispersity": light_polydispersity,
        "light_volume_ratio": light_volume_ratio,
        "light_cumulative_collisions": light_cum_collisions,
    }


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

    # 5. Лёгкие метрики из цепочки solution (высокое разрешение)
    metrics = extract_chain_metrics(solver.solution)

    # 6. Тяжёлые снимки (полные массивы радиусов для KDE) — редко
    snapshot_times = np.arange(0, T_STOP + SNAPSHOT_INTERVAL, SNAPSHOT_INTERVAL)
    radii_snapshots = {}

    for t in snapshot_times:
        state = solver.solution.get_state(float(t))
        radii_snapshots[float(t)] = state.radii.copy()

    return {
        "elapsed_time": elapsed,
        **metrics,
        # Тяжёлые снимки (каждые SNAPSHOT_INTERVAL сек)
        "snapshot_times": snapshot_times,
        "radii_snapshots": radii_snapshots,
    }


# ============================================================
#  Сохранение результатов
# ============================================================

def config_dir(method: str, boundary: str, dt: float) -> str:
    """Директория для промежуточных результатов одной конфигурации."""
    d = os.path.join(RESULTS_DIR, f"{method}_{boundary}_dt{dt}")
    os.makedirs(d, exist_ok=True)
    return d


def run_filename(method: str, boundary: str, dt: float, run_idx: int) -> str:
    """Путь к .npz одного рана."""
    return os.path.join(config_dir(method, boundary, dt), f"run{run_idx:03d}.npz")


def result_filename(method: str, boundary: str, dt: float) -> str:
    """Финальный агрегированный файл."""
    return os.path.join(RESULTS_DIR, f"{method}_{boundary}_dt{dt}.npz")


def count_completed_runs(method: str, boundary: str, dt: float) -> int:
    """Сколько ранов уже посчитано для данной конфигурации."""
    d = config_dir(method, boundary, dt)
    count = 0
    while os.path.exists(os.path.join(d, f"run{count:03d}.npz")):
        count += 1
    return count


def save_single_run(result: dict, method: str, boundary: str, dt: float, run_idx: int) -> None:
    """Сохраняет результат одного рана в отдельный .npz."""
    snapshot_times = result["snapshot_times"]

    save_dict = {
        "elapsed_time": np.array(result["elapsed_time"]),
        # Лёгкие метрики (высокое разрешение)
        "light_times": result["light_times"],
        "light_droplet_counts": result["light_droplet_counts"],
        "light_mean_radii": result["light_mean_radii"],
        "light_median_radii": result["light_median_radii"],
        "light_max_radii": result["light_max_radii"],
        "light_polydispersity": result["light_polydispersity"],
        "light_volume_ratio": result["light_volume_ratio"],
        "light_cumulative_collisions": result["light_cumulative_collisions"],
        # Тяжёлые снимки
        "snapshot_times": snapshot_times,
        "method": np.array(method),
        "boundary": np.array(boundary),
        "dt": np.array(dt),
        "N": np.array(N),
    }

    for t_val in snapshot_times:
        key = f"radii_at_t{int(t_val)}"
        save_dict[key] = result["radii_snapshots"][float(t_val)]

    fname = run_filename(method, boundary, dt, run_idx)
    tmp_fname = fname + ".tmp"
    np.savez(tmp_fname, **save_dict)
    os.replace(tmp_fname, fname)
    print(f"  Сохранён: {fname}")


def merge_runs(method: str, boundary: str, dt: float) -> None:
    """Собирает все отдельные раны в один финальный .npz."""
    d = config_dir(method, boundary, dt)
    run_files = sorted(
        f for f in os.listdir(d) if f.startswith("run") and f.endswith(".npz")
    )

    if len(run_files) < NUM_RUNS:
        print(f"  Пропускаем merge: {len(run_files)}/{NUM_RUNS} ранов готово")
        return

    results = []
    for rf in run_files:
        data = np.load(os.path.join(d, rf), allow_pickle=True)
        results.append(data)

    snapshot_times = results[0]["snapshot_times"]
    light_times = results[0]["light_times"]
    elapsed_times = np.array([r["elapsed_time"].item() for r in results])

    # Лёгкие метрики (высокое разрешение): (num_runs, num_light_points)
    light_droplet_counts = np.array([r["light_droplet_counts"] for r in results])
    light_mean_radii = np.array([r["light_mean_radii"] for r in results])
    light_median_radii = np.array([r["light_median_radii"] for r in results])
    light_max_radii = np.array([r["light_max_radii"] for r in results])
    light_polydispersity = np.array([r["light_polydispersity"] for r in results])
    light_volume_ratio = np.array([r["light_volume_ratio"] for r in results])
    light_cumulative_collisions = np.array([r["light_cumulative_collisions"] for r in results])

    # Обратная совместимость: droplet_counts/mean_radii/median_radii на редкой сетке
    # Берём из light-метрик значения в точках snapshot_times
    light_dt = LIGHT_METRIC_INTERVAL
    snapshot_indices = np.round(snapshot_times / light_dt).astype(int)
    snapshot_indices = np.clip(snapshot_indices, 0, light_droplet_counts.shape[1] - 1)

    droplet_counts = light_droplet_counts[:, snapshot_indices]
    mean_radii_arr = light_mean_radii[:, snapshot_indices]
    median_radii_arr = light_median_radii[:, snapshot_indices]

    save_dict = {
        "elapsed_times": elapsed_times,
        # Лёгкие метрики (высокое разрешение)
        "light_times": light_times,
        "light_droplet_counts": light_droplet_counts,
        "light_mean_radii": light_mean_radii,
        "light_median_radii": light_median_radii,
        "light_max_radii": light_max_radii,
        "light_polydispersity": light_polydispersity,
        "light_volume_ratio": light_volume_ratio,
        "light_cumulative_collisions": light_cumulative_collisions,
        # Обратная совместимость (редкая сетка)
        "droplet_counts": droplet_counts,
        "mean_radii": mean_radii_arr,
        "median_radii": median_radii_arr,
        "snapshot_times": snapshot_times,
        # Мета
        "method": np.array(method),
        "boundary": np.array(boundary),
        "dt": np.array(dt),
        "num_runs": np.array(len(results)),
        "N": np.array(N),
        "theta": np.array(THETA),
        "mpl": np.array(MPL),
    }

    # Тяжёлые снимки (полные радиусы для KDE)
    for t_val in snapshot_times:
        for i, r in enumerate(results):
            key = f"radii_at_t{int(t_val)}_run{i}"
            save_dict[key] = r[f"radii_at_t{int(t_val)}"]

    fname = result_filename(method, boundary, dt)
    tmp_fname = fname + ".tmp"
    np.savez(tmp_fname, **save_dict)
    os.replace(tmp_fname, fname)
    print(f"Финальный файл: {fname}")

    for r in results:
        r.close()


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

            # Пропускаем если финальный файл уже собран
            if os.path.exists(fname):
                print(f"\n[{config_idx}/{total_configs}] {method}+{boundary} dt={dt} "
                      f"— уже существует, пропускаем")
                continue

            # Проверяем сколько ранов уже посчитано
            done = count_completed_runs(method, boundary, dt)
            remaining = NUM_RUNS - done

            if remaining <= 0:
                # Все раны есть, но финальный файл не собран — собираем
                print(f"\n[{config_idx}/{total_configs}] {method}+{boundary} dt={dt} "
                      f"— все {NUM_RUNS} ранов готовы, собираем финальный файл")
                merge_runs(method, boundary, dt)
                continue

            print(f"\n{'=' * 70}")
            print(f"  [{config_idx}/{total_configs}] {method}+{boundary}, dt={dt}")
            if done > 0:
                print(f"  Продолжаем с рана {done}/{NUM_RUNS} ({done} уже посчитано)")
            else:
                print(f"  {NUM_RUNS} запусков")
            print(f"{'=' * 70}")

            for run_idx in range(done, NUM_RUNS):
                t0 = time.perf_counter()
                result = run_single_simulation(method, boundary, dt)
                run_time = time.perf_counter() - t0

                n_final = result["light_droplet_counts"][-1]
                print(f"  {method}+{boundary} dt={dt} "
                      f"run {run_idx + 1}/{NUM_RUNS}: "
                      f"N_end={n_final}, "
                      f"время={run_time:.1f} сек")

                save_single_run(result, method, boundary, dt, run_idx)

            merge_runs(method, boundary, dt)

    print("\n" + "=" * 70)
    print("  Бенчмарк завершён!")
    print("=" * 70)


if __name__ == "__main__":
    main()
