"""Сравнение трёх моделей коалесценции:
  1. DEM — прямое моделирование столкновений каплей.
  2. PBM с аналитическим электростатическим ядром.
  3. PBM с ядром, извлечённым из DEM-статистики (DEMExtractedKernel).

Строит два графика:
  - распределение капель по объёмам (на момент t = t_stop);
  - число капель от времени.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

from dem.collision_detector import SpatialHashCollisionDetector
from dem.force_calculator import *  # noqa: F401,F403
from dem.octree.force_tree import TreeDropletForceCalculator
from dem.particle_generator import *  # noqa: F401,F403
from dem.particle_state import DropletState
from dem.post_processor import *  # noqa: F401,F403
from dem.solution import DropletSolution
from dem.solver import EulerDropletSolver

from pbm import PBMSolver, VolumeGrid
from pbm.coupling import DEMPBMCoupling
from pbm.kernels import AnalyticalElectrostaticKernel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
ti.init(arch=ti.cpu, cpu_max_num_threads=16, default_fp=ti.f64)


def run_dem_with_pbm_dem_kernel(initial_state, params):
    """Запуск DEM + PBM (PBM использует динамически извлекаемое из DEM ядро)."""
    box_size = params["box_size"]
    domain_volume = box_size**3
    radii_range = params["radii_range"]

    force_calculator = TreeDropletForceCalculator(
        num_particles=len(initial_state.radii),
        theta=0.35,
        mpl=4,
        eps_oil=params["eps_oil"],
        eta_oil=params["eta_oil"],
        eta_water=params["eta_water"],
        E=params["E"],
        L=box_size,
        periodic=False,
        correction_grid_resolution=0,
    )
    collision_detector = SpatialHashCollisionDetector(
        num_particles=len(initial_state.radii),
        L=box_size,
        boundary_mode="open",
    )
    solution = DropletSolution(
        initial_droplet_state=deepcopy(initial_state),
        real_time_visualization=False,
    )
    post_processor = DropletPostProcessor(solution, box_size=box_size)  # noqa: F405

    grid = params["grid"]
    kernel = AnalyticalElectrostaticKernel(
        params["eps0"], params["eps_oil"], params["E"], params["eta_oil"]
    )
    Q0 = kernel.build_matrix(grid)
    pbm_solver = PBMSolver(
        grid, Q0, method="cell_average", integrator="BDF",
        domain_volume=domain_volume,
    )
    coupling = DEMPBMCoupling(
        grid=grid,
        pbm_solver=pbm_solver,
        domain_volume=domain_volume,
        coupling_interval=params["pbm_coupling_interval"],
    )
    coupling.initialize_from_dem(initial_state.radii)

    solver = EulerDropletSolver(
        force_calculator=force_calculator,
        solution=solution,
        post_processor=post_processor,
        collision_detector=collision_detector,
        pbm_coupling=coupling,
    )
    solver.solve(params["dt"], params["t_stop"])
    return solution, coupling


def run_pbm_analytical(initial_state, params):
    """Чистый PBM с аналитическим ядром, та же сетка и N0."""
    grid = params["grid"]
    domain_volume = params["box_size"] ** 3
    kernel = AnalyticalElectrostaticKernel(
        params["eps0"], params["eps_oil"], params["E"], params["eta_oil"]
    )
    Q = kernel.build_matrix(grid)
    solver = PBMSolver(
        grid, Q, method="cell_average", integrator="BDF",
        domain_volume=domain_volume,
    )
    N0 = grid.histogram(initial_state.radii)
    n_pts = max(int(params["t_stop"]) + 1, 51)
    t_eval = np.linspace(0, params["t_stop"], n_pts)
    result = solver.solve(N0, (0, params["t_stop"]), t_eval=t_eval, rtol=1e-6, atol=1e-6)
    return result


def main() -> None:
    radii_range = np.array([2.5e-6, 7.5e-6])
    num_particles = 10000
    water_volume_content = 0.02
    box_size = float(
        np.cbrt(
            (np.pi * num_particles * np.sum(radii_range) * np.sum(np.square(radii_range)))
            / (3 * water_volume_content)
        )
    )

    pbm_n_bins = 50
    grid = VolumeGrid.from_radii_range(
        radii_range[0], radii_range[1] * 10, pbm_n_bins, spacing="logarithmic"
    )

    params = {
        "radii_range": radii_range,
        "box_size": box_size,
        "eps0": 8.85e-12,
        "eps_oil": 2.85,
        "eta_oil": 0.065,
        "eta_water": 0.001,
        "E": 3e5,
        "dt": 0.04,
        "t_stop": 19.0,
        "pbm_coupling_interval": 1.0,
        "grid": grid,
    }

    fname = f"results/droplets_N{num_particles}_vol{water_volume_content}_0.xlsx"
    initial_state = DropletState(filename=fname)

    print("=== [1/2] DEM + PBM(DEM-kernel) ===")
    solution_dem, coupling = run_dem_with_pbm_dem_kernel(initial_state, params)

    print("=== [2/2] PBM(analytical) ===")
    pbm_ana = run_pbm_analytical(initial_state, params)

    centers = grid.centers

    # --- График 1: распределение по объёмам в финальный момент ---
    if not coupling.history_t:
        raise RuntimeError("Нет истории coupling — увеличьте t_stop или уменьшите coupling_interval")

    dem_final = coupling.history_dem_N[-1]
    pbm_dem_final = coupling.history_pbm_N[-1]
    pbm_ana_final = pbm_ana["N"][-1]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(centers, dem_final, "o-", label="DEM", markersize=4)
    ax1.plot(centers, pbm_dem_final, "s-", label="PBM (DEM-извлечённое ядро)", markersize=4)
    ax1.plot(centers, pbm_ana_final, "^-", label="PBM (аналитическое ядро)", markersize=4)
    ax1.set_xscale("log")
    ax1.set_xlabel("Объём капли, м³")
    ax1.set_ylabel("Число капель")
    ax1.set_title(f"Распределение капель по объёмам, t = {coupling.history_t[-1]:.1f} с")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig("compare_volume_distribution.png", dpi=150)

    # --- График 2: число капель от времени ---
    t_dem = np.array(coupling.history_t)
    n_dem = np.array([np.sum(h) for h in coupling.history_dem_N])
    n_pbm_dem = np.array([np.sum(h) for h in coupling.history_pbm_N])

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(t_dem, n_dem, "o-", label="DEM", markersize=4)
    ax2.plot(t_dem, n_pbm_dem, "s-", label="PBM (DEM-извлечённое ядро)", markersize=4)
    ax2.plot(pbm_ana["t"], pbm_ana["total_count"], "-", label="PBM (аналитическое ядро)")
    ax2.set_xlabel("Время, с")
    ax2.set_ylabel("Общее число капель")
    ax2.set_title("Динамика числа капель")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("compare_count_vs_time.png", dpi=150)

    print("\nСохранено:")
    print("  compare_volume_distribution.png")
    print("  compare_count_vs_time.png")
    plt.show()


if __name__ == "__main__":
    main()
