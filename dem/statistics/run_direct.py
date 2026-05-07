"""
Сбор статистики: Direct-метод (NUM_RUNS реализаций).

Запуск: python statistics/run_direct.py
Результат: statistics/results/statistics_direct.npz
"""

import json
import os
import sys
import time

import numpy as np

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def compute_box_size(N, radii_range, water_volume_content):
    rr = np.array(radii_range)
    return float(
        np.cbrt((np.pi * N * np.sum(rr) * np.sum(np.square(rr))) / (3 * water_volume_content))
    )


def run_single_simulation(cfg, box_size):
    """Запуск одной симуляции direct-методом."""
    from dem.collision_detector import SpatialHashCollisionDetector
    from dem.force_calculator import DirectDropletForceCalculator
    from dem.particle_generator import UniformDropletGenerator
    from dem.post_processor import DropletPostProcessor
    from dem.solution import DropletSolution
    from dem.solver import EulerDropletSolver

    sim = cfg["simulation"]
    phys = cfg["physics"]
    snapshot_times = cfg["snapshot_times"]
    histogram_times = cfg["histogram_times"]

    generator = UniformDropletGenerator(
        coord_range=(0, box_size),
        radii_range=np.array(phys["radii_range"]),
        num_particles=sim["N"],
        minimum_distance=1e-6,
    )
    initial_state = generator.generate()
    num_particles = len(initial_state.radii)

    force_calculator = DirectDropletForceCalculator(
        num_particles=num_particles,
        eps_oil=phys["eps_oil"],
        eta_oil=phys["eta_oil"],
        eta_water=phys["eta_water"],
        rho_water=phys["rho_water"],
        rho_oil=phys["rho_oil"],
        E=phys["E"],
        L=box_size,
        boundary_mode=phys["boundary_mode"],
    )

    collision_detector = SpatialHashCollisionDetector(
        num_particles=num_particles,
        L=box_size,
        boundary_mode=phys["boundary_mode"],
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
    solver.solve(sim["dt"], sim["t_stop"])
    elapsed = time.perf_counter() - t_start

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


def save_phase_results(results, filename, cfg):
    """Сохранение результатов в .npz."""
    snapshot_times = cfg["snapshot_times"]
    histogram_times = cfg["histogram_times"]

    elapsed_times = np.array([r["elapsed_time"] for r in results])
    droplet_counts = np.array([r["droplet_counts"] for r in results])
    median_radii = np.array([r["median_radii"] for r in results])

    save_dict = {
        "elapsed_times": elapsed_times,
        "droplet_counts": droplet_counts,
        "median_radii": median_radii,
        "snapshot_times": np.array(snapshot_times),
    }

    for t in histogram_times:
        arrays = [r["radii_snapshots"][t] for r in results]
        for i, arr in enumerate(arrays):
            save_dict[f"radii_at_t{t}_run{i}"] = arr

    np.savez(filename, **save_dict)
    print(f"Результаты сохранены: {filename}")


def main():
    cfg = load_config()
    sim = cfg["simulation"]
    phys = cfg["physics"]
    ti_cfg = cfg["taichi"]

    import taichi as ti

    ti.init(
        arch=getattr(ti, ti_cfg["arch"]),
        cpu_max_num_threads=ti_cfg["cpu_max_num_threads"],
        default_fp=ti.f64,
    )

    box_size = compute_box_size(sim["N"], phys["radii_range"], phys["water_volume_content"])

    print(f"Direct-метод: N={sim['N']}, dt={sim['dt']}, t_stop={sim['t_stop']}")
    print(f"box_size={box_size:.6e}, boundary_mode={phys['boundary_mode']}")
    print(f"Число реализаций: {sim['num_runs']}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for i in range(sim["num_runs"]):
        result = run_single_simulation(cfg, box_size)
        results.append(result)
        print(
            f"Direct {i+1}/{sim['num_runs']} done, "
            f"elapsed: {result['elapsed_time']:.1f} сек, "
            f"N_final={result['droplet_counts'][-1]}"
        )

    output_path = os.path.join(RESULTS_DIR, "statistics_direct.npz")
    save_phase_results(results, output_path, cfg)
    print("\nГотово!")


if __name__ == "__main__":
    main()
