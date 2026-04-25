import time
import numpy as np
from .solver_base import Solver

class EulerDropletSolver(Solver):

    def __init__(self, force_calculator, solution, post_processor,
                 collision_detector=None, save_interval: int = 1):
        """
        Инициализация класса Solver.

        :param force_calculator: Объект класса ForceCalculator, используемый для расчета сил.
        :param solution: Объект класса Solution для сохранения траекторий.
        :param post_processor: Объект класса PostProcessor для визуализации.
        :param collision_detector: Объект CollisionDetector для обнаружения столкновений (опционально).
        :param save_interval: Сохранять каждый K-й шаг (1 = каждый шаг, 10 = каждый 10-й).
        """
        self.force_calculator = force_calculator
        self.solution = solution
        self.post_processor = post_processor
        self.collision_detector = collision_detector
        self.simulation_time = 0  # Время моделирования, реальное
        self.save_interval = max(save_interval, 1)

        eta_oil = self.force_calculator.eta_oil
        eta_water = self.force_calculator.eta_water
        self.stokes_factor = 2*np.pi*eta_oil*(2*eta_oil+3*eta_water)/(eta_oil+eta_water)

    def solve(self, dt, total_time):
        """
        Основной метод для решения уравнений движения капель в вязкой жидкости методом Эйлера.

        :param dt: Временной шаг симуляции.
        :param total_time: Общее время моделирования.
        """
        t = 0
        is_aborted = False

        # Засекаем время
        time_start = time.time()

        last_percent = -1

        # Главный цикл симуляции
        while (t < total_time) and not is_aborted:
            # Остановка по запросу пользователя
            if self.post_processor.stop_simulation:
                print("Симуляция остановлена пользователем.")
                break

            positions = self.solution.initial_droplet_state.positions
            radii = self.solution.initial_droplet_state.radii

            remaining_steps = int((total_time - t) / dt) + 1

            for step in range(remaining_steps):
                # Вывод прогресса (внутри цикла, чтобы обновлялся каждый шаг)
                percent = int(100 * t / total_time)
                if percent != last_percent:
                    last_percent = percent
                    elapsed = time.time() - time_start
                    print(f"{percent}%: Время: {t:.3f} из {total_time:.3f}, "
                          f"Капель: {self.solution.num_particles}, "
                          f"Прошло: {elapsed:.1f} сек",
                          flush=True)

                # Прерывание симуляции, если окно было закрыто
                if self.post_processor.stop_simulation:
                    print("Симуляция остановлена пользователем.")
                    is_aborted = True
                    break

                # Collision detection — отдельный проход
                if self.collision_detector is not None:
                    is_collision, collided_pairs = self.collision_detector.detect(positions, radii)
                    if is_collision:
                        self.solution.is_collision = True
                        self.solution.collided_droplets = collided_pairs
                        self.solution.compact()
                        new_solution = self.solution.generate_next_solution()
                        self.solution = new_solution
                        self.post_processor.update_solution(self.solution)
                        if self.solution.real_time_visualization:
                            self.post_processor.update_live_plot()
                        break

                # Вычисляем силы и скорости
                if hasattr(self.force_calculator, 'calculate_forces_and_total_velocity'):
                    forces, total_velocities = self.force_calculator.calculate_forces_and_total_velocity(
                        positions, radii, stokes_factor=self.stokes_factor)
                    positions += total_velocities * dt
                else:
                    forces, convection_velocities = self.force_calculator.calculate_forces_and_convection(positions, radii)
                    migration_velocities = forces / (self.stokes_factor * radii[:, np.newaxis])
                    positions += (migration_velocities + convection_velocities) * dt

                # Граничные условия
                L = self.force_calculator.L
                boundary_mode = self.force_calculator.boundary_mode
                if boundary_mode == "periodic":
                    positions %= L
                # "open" — без ограничений, капли могут покидать коробку

                # Сохраняем текущее положение частиц
                t += dt
                if self.save_interval <= 1 or step % self.save_interval == 0:
                    self.solution.save_step(t, positions)
                elif step == remaining_steps - 1:
                    # Всегда сохраняем последний шаг
                    self.solution.save_step(t, positions)

                # Если нужно обновить график
                if self.solution.should_update_visualization():
                    self.post_processor.live_plot()

        # Сжимаем решение
        self.solution.compact_all()

        # Выводим время моделирования
        self.simulation_time = time.time() - time_start
        print(f"Общее время расчёта: {self.simulation_time:.2f} сек")

        # После завершения симуляции останавливаем визуализацию
        if self.solution.real_time_visualization:
            self.post_processor.finalize_live_plot()
