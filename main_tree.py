import time
import numpy as np
from dem.collision_detector import SpatialHashCollisionDetector
from dem.force_calculator import *
from dem.octree.force_tree import TreeDropletForceCalculator
from dem.particle_generator import *
from dem.particle_state import *
from dem.post_processor import *
from dem.solution import *
from dem.solver import *

import taichi as ti


n_of_threads = 16
ti.init(arch=ti.cpu, cpu_max_num_threads=n_of_threads, default_fp=ti.f64)  # Параллельный расчёт на CPU

#import tracemalloc

def main():
    """
    Главная функция, запускающая моделирование.

    Загружает начальные условия из файла Excel или создаёт новые,
    создает необходимые объекты,
    запускает решатель задачи,
    сохраняет результаты в файл.
    """
    print('\n'*3)

    # tracemalloc.start()

    # Параметры симуляции ___________________________________________________________________________________
    radii_range = np.array([2.5e-6, 7.5e-6])
    num_particles = 1000
    water_volume_content = 0.02

    # Вычисляем размер области моделирования (см заметки, там формула)
    box_size = np.cbrt(  (np.pi*num_particles*np.sum(radii_range)*np.sum(np.square(radii_range)))  /  (3*water_volume_content)  )
    coord_range = (0, box_size)

    t_stop = 100  # Время окончания симуляции
    dt = 0.04  # Временной шаг для интеграции

    real_time_visualization = False

    should_use_saved_initial_data = True
    filenumber = 0
    filename_to_save = f"results/droplets_N{num_particles}_vol{water_volume_content}_0.xlsx"
    filename_to_load = f"results/droplets_N{num_particles}_vol{water_volume_content}_{filenumber}.xlsx"

    # Физические параметры
    eps_oil = 2.85
    eta_oil = 0.065
    eta_water = 0.001
    rho_water = 1000
    rho_oil = 900
    E = 3e5

    # Граничные условия:
    #   "periodic" - minimum image convention + оборачивание позиций (ОБЯЗАТЕЛЬНО при use_periodic_correction=True)
    #   "open"     - открытая коробка, капли могут покидать границы
    boundary_mode = "periodic"

    # Периодическая поправка из COMSOL (псевдо-периодичность)
    # При True: конвекция = прямой стоклет + поправка от периодических образов = полная периодическая функция Грина
    # Требует boundary_mode = "periodic". Параметры в periodic_correction/data/comsol_params.json
    use_periodic_correction = True

    # Параметры дерева
    theta = 0.35  # Параметр Барнса-Хатта (точность аппроксимации)
    mpl = 1      # Максимальное число частиц в листе октодерева

    # ___________________________________________________________________________________

    initial_state = None
    if should_use_saved_initial_data:
        initial_state = DropletState(filename=filename_to_load)
    else:
        particle_generator = UniformDropletGenerator(coord_range=coord_range, radii_range=radii_range, num_particles=num_particles, minimum_distance=1e-6) # type: ignore
        initial_state = particle_generator.generate()
        initial_state.export_to_xlsx(filename_to_save)
        #particle_generator.plot()
        particle_generator.print()
        time.sleep(2)

    num_particles = len(initial_state.radii) # type: ignore


    # Загрузка периодической поправки COMSOL (если включена)
    lattice_correction = None
    if use_periodic_correction:
        if boundary_mode != "periodic":
            raise ValueError("use_periodic_correction=True требует boundary_mode='periodic'")
        from dem.periodic_correction import COMSOLLatticeCorrection
        lattice_correction = COMSOLLatticeCorrection.load_default()

    # Создаем необходимые объекты
    periodic = (boundary_mode == "periodic")
    corr_res = lattice_correction.grid_resolution if lattice_correction else 0
    force_calculator = TreeDropletForceCalculator(num_particles=num_particles, theta=theta, mpl=mpl, eps_oil=eps_oil, eta_oil=eta_oil, eta_water=eta_water, E=E, L=box_size, periodic=periodic, correction_grid_resolution=corr_res)  # Объект для расчета сил (октодерево)

    if lattice_correction:
        force_calculator.load_periodic_correction(lattice_correction, L_sim=box_size)

    collision_detector = SpatialHashCollisionDetector(num_particles=num_particles, L=box_size, boundary_mode=boundary_mode)

    solution = DropletSolution(initial_droplet_state=initial_state, real_time_visualization=real_time_visualization)  # Объект для хранения траекторий

    post_processor = DropletPostProcessor(solution, box_size=box_size)  # Объект для визуализации

    solver = EulerDropletSolver(force_calculator=force_calculator, solution=solution, post_processor=post_processor, collision_detector=collision_detector)  # Основной решатель задачи

    # --- Tree diagnostics (one build + stats) ---
    _positions = initial_state.positions.copy()
    _radii = initial_state.radii.copy()

    _t0 = time.perf_counter()
    force_calculator.octree.build(_positions, _radii, box_size, periodic)
    _build_ms = (time.perf_counter() - _t0) * 1000

    _t0 = time.perf_counter()
    _ = force_calculator.octree.compute_forces(force_calculator.m_const)
    _compute_ms = (time.perf_counter() - _t0) * 1000

    from dem.octree import compute_tree_stats, print_tree_stats
    stats = compute_tree_stats(force_calculator.octree, build_time_ms=_build_ms, compute_time_ms=_compute_ms)
    print_tree_stats(stats)

    VISUALIZE_TREE = True
    if VISUALIZE_TREE:
        from dem.octree import visualize_tree
        visualize_tree(force_calculator.octree, _positions, _radii)

    # Запуск решения
    #profiler = cProfile.Profile()
    #profiler.enable()
    solver.solve(dt, t_stop)
    # profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('tottime')
    #stats.dump_stats('results/profile.prof')
    #print("\n Function call statistics:")
    #stats.print_stats(20)  # Вывести топ-20 затратных функций


    # Снимок текущего состояния памяти
    # snapshot = tracemalloc.take_snapshot()
    # Анализируем топ-10 самых больших источников потребления памяти
    #top_stats = snapshot.statistics('lineno')
    #print("[ Top 10 memory consumers ]")
    #for stat in top_stats[:10]:
    # print(stat)


    # Сохраняем решение в файл
    #stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp = f'N{num_particles}_vol{water_volume_content}_dt{dt}_{filenumber}_t1'
    results_filename = f"results/results_{stamp}.npz"
    solution.save_chain_to_file(results_filename, precision='float32')

if __name__ == "__main__":
    main()
