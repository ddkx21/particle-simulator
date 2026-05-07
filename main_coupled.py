"""Связанный запуск DEM + PBM.

DEM-симуляция с октодеревом, PBM получает данные о столкновениях
и распределении, решает уравнения популяционного баланса параллельно.
"""

import numpy as np
import taichi as ti
from dem.collision_detector import SpatialHashCollisionDetector
from dem.force_calculator import *
from dem.octree.force_tree import TreeDropletForceCalculator
from dem.particle_generator import *
from dem.particle_state import *
from dem.post_processor import *
from dem.solution import *
from dem.solver import *

from pbm import VolumeGrid, PBMSolver
from pbm.kernels import AnalyticalElectrostaticKernel
from pbm.coupling import DEMPBMCoupling

n_of_threads = 16
ti.init(arch=ti.cpu, cpu_max_num_threads=n_of_threads, default_fp=ti.f64)


def main() -> None:
    # Параметры симуляции
    radii_range = np.array([2.5e-6, 7.5e-6])
    num_particles = 100
    water_volume_content = 0.02

    box_size = np.cbrt(
        (np.pi * num_particles * np.sum(radii_range) * np.sum(np.square(radii_range)))
        / (3 * water_volume_content)
    )
    coord_range = (0, box_size)

    t_stop = 100
    dt = 0.04

    real_time_visualization = True

    should_use_saved_initial_data = True
    filenumber = 0
    filename_to_load = f"results/droplets_N{num_particles}_vol{water_volume_content}_{filenumber}.xlsx"

    # Физические параметры
    eps0 = 8.85e-12
    eps_oil = 2.85
    eta_oil = 0.065
    eta_water = 0.001
    E = 3e5

    boundary_mode = "open"
    use_periodic_correction = False

    # Параметры дерева
    theta = 0.35
    mpl = 4

    # PBM параметры
    pbm_n_bins = 50
    pbm_coupling_interval = 1.0

    # Начальные условия
    initial_state = None
    if should_use_saved_initial_data:
        initial_state = DropletState(filename=filename_to_load)
    else:
        particle_generator = UniformDropletGenerator(
            coord_range=coord_range,
            radii_range=radii_range,
            num_particles=num_particles,
            minimum_distance=1e-6,
        )
        initial_state = particle_generator.generate()

    num_particles = len(initial_state.radii)
    domain_volume = box_size**3

    # Периодическая поправка COMSOL
    lattice_correction = None
    if use_periodic_correction:
        from dem.periodic_correction import COMSOLLatticeCorrection
        lattice_correction = COMSOLLatticeCorrection.load_default()

    # Force calculator
    periodic = boundary_mode == "periodic"
    corr_res = lattice_correction.grid_resolution if lattice_correction else 0
    force_calculator = TreeDropletForceCalculator(
        num_particles=num_particles,
        theta=theta,
        mpl=mpl,
        eps_oil=eps_oil,
        eta_oil=eta_oil,
        eta_water=eta_water,
        E=E,
        L=box_size,
        periodic=periodic,
        correction_grid_resolution=corr_res,
    )
    if lattice_correction:
        force_calculator.load_periodic_correction(lattice_correction, L_sim=box_size)

    collision_detector = SpatialHashCollisionDetector(
        num_particles=num_particles, L=box_size, boundary_mode=boundary_mode,
    )

    solution = DropletSolution(
        initial_droplet_state=initial_state,
        real_time_visualization=real_time_visualization,
    )

    post_processor = DropletPostProcessor(solution, box_size=box_size)

    # PBM setup
    grid = VolumeGrid.from_radii_range(radii_range[0], radii_range[1] * 10, pbm_n_bins, spacing="geometric")
    kernel = AnalyticalElectrostaticKernel(eps0, eps_oil, E, eta_oil)
    Q = kernel.build_matrix(grid)
    pbm_solver = PBMSolver(
        grid, Q, method="cell_average", integrator="BDF",
        domain_volume=domain_volume,
    )

    coupling = DEMPBMCoupling(
        grid=grid,
        pbm_solver=pbm_solver,
        domain_volume=domain_volume,
        coupling_interval=pbm_coupling_interval,
    )
    coupling.initialize_from_dem(initial_state.radii)

    # Solver с PBM
    solver = EulerDropletSolver(
        force_calculator=force_calculator,
        solution=solution,
        post_processor=post_processor,
        collision_detector=collision_detector,
        pbm_coupling=coupling,
    )

    print(f"DEM+PBM: {num_particles} частиц, {pbm_n_bins} PBM-бинов")
    print(f"Box size: {box_size:.4e} m, Domain volume: {domain_volume:.4e} m³")

    solver.solve(dt, t_stop)

    # Результаты PBM
    centers, pbm_N = coupling.get_pbm_distribution()
    print(f"\nPBM итог: {np.sum(pbm_N):.0f} частиц в {pbm_n_bins} бинах")

    # Визуализация сравнения
    if coupling.history_t:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        dem_final = coupling.history_dem_N[-1]
        pbm_final = coupling.history_pbm_N[-1]

        ax.plot(centers, dem_final, "o-", label="DEM", markersize=3)
        ax.plot(centers, pbm_final, "s-", label="PBM", markersize=3)
        ax.set_xlabel("Объём частиц (м³)")
        ax.set_ylabel("Число частиц")
        ax.set_title(f"DEM vs PBM: t = {coupling.history_t[-1]:.1f} с")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("dem_vs_pbm.png", dpi=150)
        plt.show()

    # Сохранение
    stamp = f"N{num_particles}_vol{water_volume_content}_coupled_{filenumber}"
    solution.save_chain_to_file(f"results/results_{stamp}.npz", precision="float32")


if __name__ == "__main__":
    main()
