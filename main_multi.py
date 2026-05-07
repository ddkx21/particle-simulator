import time
import numpy as np
from dem.force_calculator import *
from dem.particle_generator import *
from dem.particle_state import *
from dem.post_processor import *
from dem.solution import *
from dem.solver import *

import taichi as ti
ti.init(arch=ti.cpu, cpu_max_num_threads=16, default_fp=ti.f64)  # Параллельный расчёт на CPU

def main():
    """
    Главная функция, запускающая моделирование для нескольких значений dt и filenumber.
    """
    print('\n'*3)
    
    # Параметры симуляции ___________________________________________________________________________________
    radii_range = np.array([2.5e-6, 7.5e-6])
    num_particles = 10000
    water_volume_content = 0.01

    # Вычисляем размер области моделирования
    box_size = np.cbrt(  (np.pi*num_particles*np.sum(radii_range)*np.sum(np.square(radii_range)))  /  (3*water_volume_content)  )
    coord_range = (0, box_size)
   
    t_stop = 200  # Время окончания симуляции

    real_time_visualization = False

    should_use_saved_initial_data = True

    # Массивы значений dt и filenumber для многократного запуска
    #dt_values = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]  # Пример значений dt
    dt_values = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]  # Пример значений dt
    filenumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]        # Пример значений filenumber

    # Физические параметры
    eps_oil = 2.85
    eta_oil = 0.065
    eta_water = 0.001
    rho_water = 1000
    rho_oil = 900
    E = 3e5

    # Запуск симуляции для каждого сочетания dt и filenumber
    for dt in dt_values:
        for filenumber in filenumbers:
            filename_to_load = f"results/droplets_N{num_particles}_vol{water_volume_content}_{filenumber}.xlsx"
            filename_to_save = f"results/droplets_N{num_particles}_vol{water_volume_content}_0.xlsx"

            initial_state = None
            if should_use_saved_initial_data:
                initial_state = DropletState(filename=filename_to_load)
            else:
                particle_generator = UniformDropletGenerator(coord_range=coord_range, radii_range=radii_range, num_particles=num_particles, minimum_distance= 1e-6)
                initial_state = particle_generator.generate()
                initial_state.export_to_xlsx(filename_to_save)
                time.sleep(2)

            actual_num_particles = len(initial_state.radii)

            # Создаем необходимые объекты
            force_calculator = DirectDropletForceCalculator(num_particles=actual_num_particles, eps_oil=eps_oil, eta_oil=eta_oil, eta_water=eta_water, rho_water=rho_water, rho_oil=rho_oil, E=E)
            solution = DropletSolution(initial_droplet_state=initial_state, real_time_visualization=real_time_visualization)
            post_processor = DropletPostProcessor(solution, box_size=box_size)
            solver = EulerDropletSolver(force_calculator=force_calculator, solution=solution, post_processor=post_processor)

            # Запуск решения
            solver.solve(dt, t_stop)

            # Сохраняем решение в файл
            stamp = f'N{num_particles}_vol{water_volume_content}_dt{dt}_{filenumber}'
            results_filename = f"results/results_multirun_{stamp}.npz"
            solution.save_chain_to_file(results_filename, precision='float32')

if __name__ == "__main__":
    main()
