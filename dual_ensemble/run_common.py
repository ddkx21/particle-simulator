"""
Общие утилиты для запуска систем двойного ансамбля.
"""

import json
import os
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize_time(t: float) -> float:
    return round(float(t), 6)


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(SCRIPT_DIR, "config.json")
    with open(path) as f:
        return json.load(f)


def compute_box_size(N: int, radii_range: np.ndarray, wvc: float) -> float:
    return float(np.cbrt(
        (np.pi * N * np.sum(radii_range) * np.sum(np.square(radii_range)))
        / (3 * wvc)
    ))


def compute_N_from_box(L: float, radii_range: np.ndarray, wvc: float) -> int:
    return int(np.round(
        3 * wvc * L ** 3
        / (np.pi * np.sum(radii_range) * np.sum(np.square(radii_range)))
    ))


def make_snapshot_times(t_stop: float, interval: float) -> list[float]:
    times = [normalize_time(t) for t in np.arange(0, t_stop, interval)]
    if len(times) == 0 or not np.isclose(times[-1], t_stop):
        times.append(normalize_time(t_stop))
    return times


def snapshot_key(prefix: str, t: float | int) -> str:
    return f"{prefix}_radii_at_t{float(t):.6g}"


def run_single_system(
    label: str,
    N: int,
    radii_range: np.ndarray,
    box_size: float,
    boundary_mode: str,
    t_stop: float,
    snapshot_times: list[float],
    cfg: dict,
    initial_state_path: str | None = None,
) -> dict:
    from collision_detector import SpatialHashCollisionDetector
    from octree.force_tree import TreeDropletForceCalculator
    from particle_generator import UniformDropletGenerator
    from particle_state import DropletState
    from post_processor import DropletPostProcessor
    from solution import DropletSolution
    from solver import EulerDropletSolver

    phys = cfg["physics"]
    tree_cfg = cfg["tree"]
    sim = cfg["simulation"]

    print(f"\n{'='*60}")
    print(f"Запуск: {label}")
    print(f"N={N}, radii=[{radii_range[0]*1e6:.1f}, {radii_range[1]*1e6:.1f}] мкм")
    print(f"L={box_size*1e6:.1f} мкм, boundary={boundary_mode}, t_stop={t_stop}")
    if initial_state_path:
        print(f"Продолжение из: {initial_state_path}")
    print(f"{'='*60}")

    if initial_state_path is not None:
        initial_state = DropletState(filename=initial_state_path)
        # DropletState.load уже приводит time к float, дублируем для надёжности.
        initial_state.time = float(initial_state.time)
        print(f"Загружено состояние: {len(initial_state.radii)} капель, t0={initial_state.time:.3f}")
    else:
        generator = UniformDropletGenerator(
            coord_range=(0, box_size),
            radii_range=radii_range,
            num_particles=N,
            minimum_distance=1e-6,
        )
        initial_state = generator.generate()

    num_particles = len(initial_state.radii)

    force_calculator = TreeDropletForceCalculator(
        num_particles=num_particles,
        theta=tree_cfg["theta"],
        mpl=tree_cfg["mpl"],
        eps_oil=phys["eps_oil"],
        eta_oil=phys["eta_oil"],
        eta_water=phys["eta_water"],
        E=phys["E"],
        L=box_size,
        periodic=(boundary_mode == "periodic"),
    )

    collision_detector = SpatialHashCollisionDetector(
        num_particles=num_particles,
        L=box_size,
        boundary_mode=boundary_mode,
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
        save_interval=sim["save_interval"],
    )

    t_start = time.perf_counter()
    solver.solve(sim["dt"], t_stop)
    elapsed = time.perf_counter() - t_start

    t0 = float(initial_state.time)
    radii_snapshots = {}
    for t in snapshot_times:
        if t == 0.0:
            radii_snapshots[normalize_time(t)] = initial_state.radii.copy()
        else:
            state = solution.get_state(t0 + float(t))
            radii_snapshots[normalize_time(t)] = state.radii.copy()

    final_state = solution.get_state(float(solution.get_current_time()))

    print(f"\n{label} завершена за {elapsed:.1f} сек")
    print(f"Начальное N={num_particles}, финальное N={len(radii_snapshots[normalize_time(snapshot_times[-1])])}")

    return {
        "elapsed_time": elapsed,
        "N_initial": num_particles,
        "radii_range": radii_range,
        "radii_snapshots": radii_snapshots,
        "final_state": final_state,
    }


def load_snapshots(path: str) -> tuple[dict[float, np.ndarray], list[float], float]:
    """Загрузить накопленные snapshots из .npz."""
    data = np.load(path)
    times = [normalize_time(t) for t in data["snapshot_times"]]
    accumulated = float(data["accumulated_time"])
    prefix = data["prefix"].item()
    radii = {}
    for t in times:
        radii[t] = data[snapshot_key(prefix, t)]
    return radii, times, accumulated


def load_elapsed(path: str) -> float:
    """Прочитать накопленное время расчёта из npz (0.0 если файла или поля нет)."""
    if not os.path.isfile(path):
        return 0.0
    try:
        data = np.load(path)
        if "elapsed" in data.files:
            return float(data["elapsed"])
    except Exception:
        pass
    return 0.0


def save_snapshots(
    path: str,
    prefix: str,
    radii: dict[float, np.ndarray],
    times: list[float],
    accumulated_time: float,
    label: str,
    radii_range: np.ndarray,
    box_size: float,
    N_initial: int,
    elapsed: float,
) -> None:
    """Сохранить накопленные snapshots в .npz."""
    normalized = [normalize_time(t) for t in times]
    save_dict: dict[str, object] = {
        "snapshot_times": np.array(normalized, dtype=np.float64),
        "accumulated_time": np.float64(accumulated_time),
        "prefix": prefix,
        "label": label,
        "radii_range": radii_range,
        "box_size": np.float64(box_size),
        "N_initial": np.int64(N_initial),
        "elapsed": np.float64(elapsed),
    }
    for t in normalized:
        save_dict[snapshot_key(prefix, t)] = radii[t]
    np.savez(path, **save_dict)
    print(f"Snapshots сохранены: {path} ({len(times)} точек, accumulated_time={accumulated_time:.1f})")


def merge_snapshots(
    old_radii: dict[float, np.ndarray],
    old_times: list[float],
    old_accumulated: float,
    new_radii: dict[float, np.ndarray],
    new_snapshot_times: list[float],
    t_stop: float,
) -> tuple[dict[float, np.ndarray], list[float], float]:
    """Объединить старые и новые snapshots. Новые times сдвигаются на old_accumulated."""
    merged_radii = dict(old_radii)
    merged_times = list(old_times)
    existing = set(merged_times)
    for t_rel in new_snapshot_times:
        t_abs = normalize_time(old_accumulated + t_rel)
        if t_abs in existing:
            merged_radii[t_abs] = new_radii[normalize_time(t_rel)]
        else:
            merged_radii[t_abs] = new_radii[normalize_time(t_rel)]
            merged_times.append(t_abs)
            existing.add(t_abs)
    new_accumulated = normalize_time(old_accumulated + t_stop)
    return merged_radii, merged_times, new_accumulated


def compute_system_params(cfg: dict) -> tuple[int, int, float]:
    """Вычислить N1, N2, box_size из конфига."""
    sim = cfg["simulation"]
    radii_range1 = np.array(cfg["system1"]["radii_range"])
    radii_range2 = np.array(cfg["system2"]["radii_range"])
    wvc = sim["water_volume_content"]
    fixed_sys = sim["fixed_system"]
    fixed_N = sim["fixed_N"]

    if fixed_sys == 2:
        N2 = fixed_N
        box_size = compute_box_size(N2, radii_range2, wvc)
        N1 = compute_N_from_box(box_size, radii_range1, wvc)
    else:
        N1 = fixed_N
        box_size = compute_box_size(N1, radii_range1, wvc)
        N2 = compute_N_from_box(box_size, radii_range2, wvc)

    return N1, N2, box_size
