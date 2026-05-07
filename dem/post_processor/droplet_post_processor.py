import numpy as np
import taichi as ti
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from.post_processor_base import PostProcessor

class DropletPostProcessor(PostProcessor):
 
    def __init__(self, solution, box_size):
        """
        Инициализация класса для визуализации результатов симуляции.

        :param solution: Объект класса Solution, содержащий траектории частиц.
        :param box_size: Размер области моделирования (размер куба).
        """
        self.solution = solution
        self.box_size = box_size
        self.fig = None
        self.ax = None
        self.circles = []
        self.stop_simulation = False  # Флаг для остановки симуляции

    def update_solution(self, new_solution):
        """
        Обновляет объект решения (solution) после столкновений.
        
        :param new_solution: Обновленный объект Solution.
        """
        self.solution = new_solution

    def plot(self):
        """
        Визуализация траекторий всех частиц после завершения симуляции.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        ax.set_zlim(0, self.box_size)

        num_particles = self.solution.num_particles
        trajectories = self.solution.get_trajectories()

        for i in range(num_particles):
            particle_trajectory = trajectories[:, i, :]
            ax.plot(particle_trajectory[:, 0], particle_trajectory[:, 1], particle_trajectory[:, 2])

        ax.set_title('Trajectories of Particles')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def live_plot(self):
        """
        Обновление положения частиц в реальном времени.
        """
        if self.fig is None or self.ax is None or self.circles is None:
            self.initialize_live_plot()

        current_positions = self.solution.get_current_positions()
        radii = self.solution.radii

        for i, circle_set in enumerate(self.circles):
            try: 
                self.update_circle_positions(circle_set, current_positions[i], radii[i])
            except IndexError:
                pass

        # Обновляем заголовок с текущим временем
        self.ax.set_title(f'Time: {self.solution.get_current_time():.3f} sec')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def initialize_live_plot(self):
        """
        Инициализация окна для обновления графика положения частиц в реальном времени.
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlim(0, self.box_size)
        self.ax.set_ylim(0, self.box_size)
        self.ax.set_zlim(0, self.box_size)

        current_positions = self.solution.get_current_positions()
        radii = self.solution.radii

        # Создаем три окружности для каждой капли
        self.circles = []
        for pos, r in zip(current_positions, radii):
            circle_set = self.create_circle_set(pos, r)
            self.circles.append(circle_set)
            for circle in circle_set:
                self.ax.add_collection3d(circle)

        # Подключаем обработчик события закрытия окна
        self.fig.canvas.mpl_connect('close_event', self.on_close_live_plot)

        self.ax.set_title(f'Time: {self.solution.current_step} steps')
        plt.ion()  # Включаем интерактивный режим
        plt.show()

    def finalize_live_plot(self):
        """
        Завершение работы окна для обновления графика положения частиц в реальном времени.
        """
        plt.ioff()
        plt.show()

    def on_close_live_plot(self, event):
        """
        Обработчик события закрытия окна.
        """
        print("Окно закрыто. Остановка симуляции.")
        self.stop_simulation = True  # Устанавливаем флаг остановки симуляции

    def create_circle_set(self, position, radius):
        """
        Создание набора из трех окружностей для отображения капли в трех основных плоскостях.

        :param position: Позиция центра окружности (x, y, z)
        :param radius: Радиус окружности
        :return: Список объектов Line3DCollection
        """
        circle_sets = []
        u = np.linspace(0, 2 * np.pi, 100)

        # Окружность в плоскости XY
        x = radius * np.cos(u) + position[0]
        y = radius * np.sin(u) + position[1]
        z = np.full_like(x, position[2])
        circle_sets.append(Line3DCollection([list(zip(x, y, z))], colors='b', linewidths=1.5, alpha=0.7))

        # Окружность в плоскости XZ
        x = radius * np.cos(u) + position[0]
        y = np.full_like(x, position[1])
        z = radius * np.sin(u) + position[2]
        circle_sets.append(Line3DCollection([list(zip(x, y, z))], colors='r', linewidths=1.5, alpha=0.7))

        # Окружность в плоскости YZ
        x = np.full_like(u, position[0])
        y = radius * np.cos(u) + position[1]
        z = radius * np.sin(u) + position[2]
        circle_sets.append(Line3DCollection([list(zip(x, y, z))], colors='g', linewidths=1.5, alpha=0.7))

        return circle_sets

    def update_circle_positions(self, circle_set, position, radius):
        """
        Обновление позиций трех окружностей.

        :param circle_set: Список из трех Line3DCollection объектов
        :param position: Новая позиция центра окружности (x, y, z)
        :param radius: Новый радиус окружности
        """
        u = np.linspace(0, 2 * np.pi, 20)

        # Обновление окружности в плоскости XY
        x = radius * np.cos(u) + position[0]
        y = radius * np.sin(u) + position[1]
        z = np.full_like(x, position[2])
        circle_set[0].set_segments([list(zip(x, y, z))])

        # Обновление окружности в плоскости XZ
        x = radius * np.cos(u) + position[0]
        y = np.full_like(x, position[1])
        z = radius * np.sin(u) + position[2]
        circle_set[1].set_segments([list(zip(x, y, z))])

        # Обновление окружности в плоскости YZ
        x = np.full_like(u, position[0])
        y = radius * np.cos(u) + position[1]
        z = radius * np.sin(u) + position[2]
        circle_set[2].set_segments([list(zip(x, y, z))])

    def update_live_plot(self):
        """
        Обновляет графику после изменения числа частиц в результате столкновений.
        """
        # Проверяем, инициализировано ли окно графики
        if self.fig is None or self.ax is None:
            self.initialize_live_plot()
        
        # Очищаем предыдущие окружности
        for circle_set in self.circles:
            for circle in circle_set:
                circle.remove()

        # Пересоздаем окружности с учетом новых частиц
        self.circles = []
        current_positions = self.solution.get_current_positions()
        radii = self.solution.radii
        for pos, r in zip(current_positions, radii):
            circle_set = self.create_circle_set(pos, r)
            self.circles.append(circle_set)
            for circle in circle_set:
                self.ax.add_collection3d(circle)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_state(self, target_time, plt_show=True):
        """
        Отображает состояние системы капель в 3D для указанного момента времени.
        
        :param target_time: Время, для которого нужно визуализировать состояние капель.
        """
        # Получаем состояние системы на указанное время
        droplet_state = self.solution.get_state(target_time)

        # Извлекаем позиции и радиусы капель
        positions = droplet_state.positions
        radii = droplet_state.radii

        # Создаем новое окно для визуализации
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        ax.set_zlim(0, self.box_size)

        # Добавляем окружности для каждой капли
        for pos, r in zip(positions, radii):
            circle_set = self.create_circle_set(pos, r)
            for circle in circle_set:
                ax.add_collection3d(circle)

        # Устанавливаем заголовок с отображением времени
        ax.set_title(f'Time: {target_time:.3f} sec, N: {len(radii)}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Показываем визуализацию
        if plt_show:
            plt.show()

    def plot_radius_histogram(self, target_time, num_bins=10, plt_show=True):
        """
        Отображает гистограмму распределения радиусов капель в указанный момент времени.

        :param target_time: Время, для которого нужно построить гистограмму.
        :param num_bins: Количество интервалов для гистограммы (по умолчанию 10).
        """

        # Получаем состояние системы для указанного времени
        state = self.solution.get_state(target_time)
        
        # Извлекаем радиусы из состояния
        radii = state.radii
        
        # Создаем гистограмму распределения радиусов
        hist, bin_edges = np.histogram(radii, bins=num_bins)

        # Создаем новое окно для отображения гистограммы
        plt.figure()
        
        # Отрисовка гистограммы
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", align="edge")
        
        # Оформление графика
        plt.title(f'Radius Distribution at Time: {target_time:.2f}')
        plt.xlabel('Radius')
        plt.ylabel('Count of Droplets')
        
        # Отображаем график
        if plt_show:
            plt.show()

    def plot_volume_histogram(self, target_time, num_bins=10, plt_show=True):
        """
        Отображает гистограмму распределения объёма капель в указанный момент времени.

        :param target_time: Время, для которого нужно построить гистограмму.
        :param num_bins: Количество интервалов для гистограммы (по умолчанию 10).
        """
        # Получаем состояние системы для указанного времени
        state = self.solution.get_state(target_time)

        # Извлекаем объёмы из состояния
        unit_scale = 1e-6 # чтобы рисовать в микрометрах
        volumes = (4/3)*np.pi * (state.radii/unit_scale) ** 3

        # Создаем гистограмму распределения объёма
        hist, bin_edges = np.histogram(volumes, bins=num_bins)

        # Создаем новое окно для отображения гистограммы
        plt.figure()

        # Отрисовка гистограммы
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", align="edge")

        # Оформление графика
        plt.title(f'Volume Distribution at Time: {target_time:.2f} s')
        plt.xlabel(r'Volume, $\mu$m^3')
        plt.ylabel('Count of Droplets')

        # Отображаем график
        if plt_show:
            plt.show()


    def plot_radius_percentiles_evolution(self, percentiles=[5, 25, 50, 75, 95], show_min=True, show_max=True):
        """
        Строит график минимального, максимального и процентилей радиусов от времени по всей цепочке решений.

        :param percentiles: Список процентилей для отображения (по умолчанию [5, 25, 50, 75, 95]).
        :param show_min: Флаг для отображения минимального радиуса (по умолчанию True).
        :param show_max: Флаг для отображения максимального радиуса (по умолчанию True).
        """
        times = []
        min_radii = []
        max_radii = []
        percentiles_radii = {p: [] for p in percentiles}

        current_solution = self.solution

        # Ищем самое первое решение
        while current_solution._prev is not None:
            current_solution = current_solution._prev

        # Проходим по всей цепочке решений
        while current_solution is not None:
            # Получаем временные шаги текущего решения
            time_steps = current_solution.get_times()

            # Получаем отображаемую величину
            radius = current_solution.radii

            # Если решение содержит только один временной шаг (вырожденный случай)
            if len(time_steps) == 1:
                time = time_steps[0]
                times.append(time)

                # Рассчитываем минимальный и максимальный радиусы
                if show_min:
                    min_radii.append(np.min(radius))
                if show_max:
                    max_radii.append(np.max(radius))

                # Рассчитываем указанные процентильные значения радиусов
                for p in percentiles:
                    p_value = np.percentile(radius, p)
                    percentiles_radii[p].append(p_value)

            # Если решение содержит больше одного временного шага
            else:
                first_time, last_time = time_steps[0], time_steps[-1]
                times.append(first_time)
                times.append(last_time)

                # Рассчитываем минимальный и максимальный радиусы
                if show_min:
                    min_radii.append(np.min(radius))
                    min_radii.append(np.min(radius))
                if show_max:
                    max_radii.append(np.max(radius))
                    max_radii.append(np.max(radius))

                # Рассчитываем указанные процентильные значения радиусов
                for p in percentiles:
                    p_value = np.percentile(radius, p)
                    percentiles_radii[p].append(p_value)
                    percentiles_radii[p].append(p_value)

            # Переходим к следующему решению в цепочке
            current_solution = current_solution._next

        # Построение графика
        plt.figure(figsize=(10, 6))

        if show_min:
            plt.plot(times, min_radii, label='Min Radius', linestyle='--', color='blue')
        if show_max:
            plt.plot(times, max_radii, label='Max Radius', linestyle='--', color='red')

        for p in percentiles:
            plt.plot(times, percentiles_radii[p], label=f'{p}th Percentile', linestyle='-', marker='o')

        plt.title('Radius Distribution Over Time')
        plt.yscale('log')

        plt.xlabel('Time, s')
        plt.ylabel('Radius, m (log scale)')

        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_radius_evolution(self, percentiles=[5, 25, 50, 75, 95], time_window=None, time_step=None, show_min=True, show_max=True, plt_show=True, plt_data_save=False):
        """
        Строит график минимального и максимального радиусов и объёмных процентилей от времени по всей цепочке решений.

        :param percentiles: Список процентилей для отображения (по умолчанию [5, 25, 50, 75, 95]).
        :param show_min: Флаг для отображения минимального радиуса (по умолчанию True).
        :param show_max: Флаг для отображения максимального радиуса (по умолчанию True).
        """
        times = []
        min_radii = []
        max_radii = []
        percentiles_radii = {p: [] for p in percentiles}

        # Определяем интервал времени, для которого нужно сравнить решения
        if time_window is None:
            time_window = [self.solution.get_chain_start_time(), self.solution.get_chain_end_time()]

        if time_step is None:
            times1 = self.solution.get_times()
            time_step = times1[1] - times1[0]

        # Отсчёт времени
        times = np.arange(time_window[0], time_window[1], time_step)

        for ind, t in enumerate(times):
            state = self.solution.get_state(t)
            
            radii = state.radii
            volumes = (4/3) * np.pi * radii**3

            min_radii.append(np.min(radii))
            max_radii.append(np.max(radii))

            # Рассчитываем указанные процентильные значения радиусов по объему
            total_volume = np.sum(volumes)
            for p in percentiles:
                target_volume = p / 100.0 * total_volume
                sorted_volumes = np.sort(volumes)
                cumulative_volumes = np.cumsum(sorted_volumes)
                percentile_volume = sorted_volumes[np.searchsorted(cumulative_volumes, target_volume)]
                percentile_radius = (3 / (4 * np.pi) * percentile_volume) ** (1/3)
                percentiles_radii[p].append(percentile_radius)


        # Построение графика
        plt.figure(figsize=(10, 6))
        unit_scale = 1e-6 # Масштабирование графика в микрометрах

        if show_max:
            plt.plot(times, np.array(max_radii)/unit_scale, label='Max Radius', linestyle='--', color='red')

        for p in reversed(percentiles):
            plt.plot(times, np.array(percentiles_radii[p])/unit_scale, label=f'{p}th Volume Percentile', linestyle='-', marker='o')

        if show_min:
            plt.plot(times, np.array(min_radii)/unit_scale, label='Min Radius', linestyle='--', color='blue')

        plt.title('Radius Distribution Over Time')
        # plt.yscale('log')

        plt.xlabel('Time, s')
        plt.ylabel(r'Radius, $\mu$m')

        plt.legend()
        plt.grid(True)
        
        if plt_data_save:
            # Сохраняем в excel данные
            folder = 'results/exportet_data'
            fname_to_save = f'{folder}/{self.solution.get_name()}___radius_evolution.xlsx'
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            df = pd.DataFrame({'Time, s': np.array(times).flatten(), r'Min Radius, $\mu$m': min_radii})
            for p in percentiles:
                df[rf'{p}th Volume Percentile, $\mu$m'] = percentiles_radii[p]
            df[r'Max Radius, $\mu$m'] = max_radii

            df.to_excel(fname_to_save, index=False)
            print(f'Data saved to {fname_to_save}')

        if plt_show:
            plt.show()

    def plot_radius_evolution_old(self, percentiles=[5, 25, 50, 75, 95], show_min=True, show_max=True, plt_show=True, plt_data_save=False):
        """
        Строит график минимального и максимального радиусов и объёмных процентилей от времени по всей цепочке решений.

        :param percentiles: Список процентилей для отображения (по умолчанию [5, 25, 50, 75, 95]).
        :param show_min: Флаг для отображения минимального радиуса (по умолчанию True).
        :param show_max: Флаг для отображения максимального радиуса (по умолчанию True).
        """
        times = []
        min_radii = []
        max_radii = []
        percentiles_radii = {p: [] for p in percentiles}

        current_solution = self.solution

        # Ищем самое первое решение
        while current_solution._prev is not None:
            current_solution = current_solution._prev

        # Проходим по всей цепочке решений
        while current_solution is not None:
            # Получаем временные шаги текущего решения
            time_steps = current_solution.get_times()
            radii = current_solution.radii
            volumes = (4/3) * np.pi * radii**3

            if len(time_steps) == 1:
                time = time_steps[0]
                times.append(time)

                # Рассчитываем минимальный и максимальный радиусы
                if show_min:
                    min_radii.append(np.min(radii))
                if show_max:
                    max_radii.append(np.max(radii))

                # Рассчитываем указанные процентильные значения радиусов по объему
                total_volume = np.sum(volumes)
                for p in percentiles:
                    target_volume = p / 100.0 * total_volume
                    sorted_volumes = np.sort(volumes)
                    cumulative_volumes = np.cumsum(sorted_volumes)
                    percentile_volume = sorted_volumes[np.searchsorted(cumulative_volumes, target_volume)]
                    percentile_radius = (3 / (4 * np.pi) * percentile_volume) ** (1/3)
                    percentiles_radii[p].append(percentile_radius)

            else:
                first_time, last_time = time_steps[0], time_steps[-1]
                times.append(first_time)
                times.append(last_time)

                # Рассчитываем минимальный и максимальный радиусы
                if show_min:
                    min_radii.append(np.min(radii))
                    min_radii.append(np.min(radii))
                if show_max:
                    max_radii.append(np.max(radii))
                    max_radii.append(np.max(radii))

                # Рассчитываем указанные процентильные значения радиусов по объему
                total_volume = np.sum(volumes)
                for p in percentiles:
                    target_volume = p / 100.0 * total_volume
                    sorted_volumes = np.sort(volumes)
                    cumulative_volumes = np.cumsum(sorted_volumes)
                    percentile_volume = sorted_volumes[np.searchsorted(cumulative_volumes, target_volume)]
                    percentile_radius = (3 / (4 * np.pi) * percentile_volume) ** (1/3)
                    percentiles_radii[p].append(percentile_radius)
                    percentiles_radii[p].append(percentile_radius)

            # Переходим к следующему решению в цепочке
            current_solution = current_solution._next


        # Построение графика
        plt.figure(figsize=(10, 6))
        unit_scale = 1e-6 # Масштабирование графика в микрометрах

        if show_max:
            plt.plot(times, np.array(max_radii)/unit_scale, label='Max Radius', linestyle='--', color='red')

        for p in reversed(percentiles):
            plt.plot(times, np.array(percentiles_radii[p])/unit_scale, label=f'{p}th Volume Percentile', linestyle='-', marker='o')

        if show_min:
            plt.plot(times, np.array(min_radii)/unit_scale, label='Min Radius', linestyle='--', color='blue')

        plt.title('Radius Distribution Over Time')
        # plt.yscale('log')

        plt.xlabel('Time, s')
        plt.ylabel(r'Radius, $\mu$m')

        plt.legend()
        plt.grid(True)
        
        if plt_data_save:
            # Сохраняем в excel данные
            folder = 'results/exportet_data'
            fname_to_save = f'{folder}/{self.solution.get_name()}___radius_evolution.xlsx'
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            df = pd.DataFrame({'Time, s': np.array(times).flatten(), r'Min Radius, $\mu$m': min_radii})
            for p in percentiles:
                df[rf'{p}th Volume Percentile, $\mu$m'] = percentiles_radii[p]
            df[r'Max Radius, $\mu$m'] = max_radii

            df.to_excel(fname_to_save, index=False)
            print(f'Data saved to {fname_to_save}')

        if plt_show:
            plt.show()


    def plot_number_of_droplets(self, time_window=None, time_step=None, plt_show=True, plt_data_save=False):
        """
        Построение графика количества частиц в цепочке решений от времени.
        """
        # Определяем интервал времени, для которого нужно сравнить решения
        if time_window is None:
            time_window = [self.solution.get_chain_start_time(), self.solution.get_chain_end_time()]

        if time_step is None:
            times1 = self.solution.get_times()
            time_step = times1[1] - times1[0]

        # Отсчёт времени
        times = np.arange(time_window[0], time_window[1], time_step)
        nums = np.zeros_like(times)

        for ind, t in enumerate(times):
            state = self.solution.get_state(t)
            nums[ind] = state.radii.shape[0]

        # Построение графика
        plt.figure(figsize=(10, 6))
        plt.plot(times, nums, label='Number of Droplets')

        plt.title('Number of Droplets Over Time')
        plt.xlabel('Time, s')
        plt.ylabel('Number of Droplets')

        plt.legend()
        plt.grid(True)

        if plt_data_save:
            # Сохраняем в excel данные
            folder = 'results/exportet_data'
            fname_to_save = f'{folder}/{self.solution.get_name()}___number_of_droplets.xlsx'
            if not os.path.exists(folder):
                os.makedirs(folder)
            df = pd.DataFrame({'Time': times, 'Number of Droplets': nums})
            df.to_excel(fname_to_save, index=False)
            print(f'Data saved to {fname_to_save}')

        if plt_show:
            plt.show()


    def compare(self, other_solution, time_window=None, time_step=None, plt_show=True, plt_data_save=False):
            """
            Сравнивает две цепочки решений и рисует график схожести от времени.

            :param other_solution: Другая цепочка решений.
            :param time_window: Временное окно для сравнения (по умолчанию - все времена основного решения).
            :param time_step: Шаг времени для сравнения (по умолчанию - шаг основного решения).
            """

            # Функция для вычисления объёма пересечения двух шаров
            @ti.func
            def volume_intersection(r1, r2, d):
                """Вычисляет объем пересечения двух шаров радиусами r1 и r2 на расстоянии d."""
                val = 0.0
                if d >= r1 + r2:
                    val = 0
                elif d <= abs(r1 - r2):
                    # Один шар полностью внутри другого
                    r = min(r1, r2)
                    val = 4 / 3 * np.pi * r**3
                else:
                    # Общий объем пересечения двух шаров
                    val = (np.pi/(12*d)) * (r1+r2-d)**2 * ( d**2 + 2*d*(r1+r2) - 3*(r1-r2)**2 )
                return val

            # Определяем интервал времени, для которого нужно сравнить решения
            if time_window is None:
                time_window = [self.solution.get_chain_start_time(), self.solution.get_chain_end_time()]

            time_window = [max(time_window[0], self.solution.get_chain_start_time(), other_solution.get_chain_start_time()), 
                        min(time_window[1], self.solution.get_chain_end_time(), other_solution.get_chain_end_time())]

            # Определяем шаг времени для сравнения
            if time_step is None:
                times1 = self.solution.get_times()
                time_step = times1[1] - times1[0]

            # Отсчёт времени
            times = np.arange(time_window[0], time_window[1], time_step)

            # Инициируем похожесть решений
            similarity = np.zeros(len(times))

            # Определяем полные объёмы капель в двух цепочках
            total_volume1 = np.sum(4 / 3 * np.pi * self.solution.radii**3)
            total_volume2 = np.sum(4 / 3 * np.pi * other_solution.radii**3)
            total_volume_geometric_mean = np.sqrt(total_volume1 * total_volume2)



            # Определяем Taichi поля для хранения радиусов, позиций и накопленного результата
            state1 = self.solution.get_state(times[0])
            state2 = other_solution.get_state(times[0])
            max_particles1 = len(state1.radii)  # Максимальное количество капель в системе 1
            max_particles2 = len(state2.radii)  # Максимальное количество капель в системе 2

            ti_radii1 = ti.field(dtype=ti.f32, shape=max_particles1)
            ti_radii2 = ti.field(dtype=ti.f32, shape=max_particles2)
            ti_positions1 = ti.Vector.field(3, dtype=ti.f32, shape=max_particles1)
            ti_positions2 = ti.Vector.field(3, dtype=ti.f32, shape=max_particles2)
            ti_accum = ti.field(dtype=ti.f32, shape=())


            # Taichi кернел для вычисления объёма пересечения всех капель
            @ti.kernel
            def ti_compute_intersection(num_particles1: ti.i32, num_particles2: ti.i32):
                for i in range(num_particles1):
                    for j in range(num_particles2):
                        pos1 = ti_positions1[i]
                        pos2 = ti_positions2[j]
                        d = (pos1 - pos2).norm()
                        ti_accum[None] += volume_intersection(ti_radii1[i], ti_radii2[j], d)


            # Сравниваем каждый шаг времени и заполняем similarity
            for ind_t, t in enumerate(times):
                
                print(f"Сравниваем шаг {ind_t+1} из {len(times)}, время {t:.2f} с.")

                # Состояния 
                state1 = self.solution.get_state(t)
                state2 = other_solution.get_state(t)

                # Радиусы и позиции
                radii1 = state1.radii
                radii2 = state2.radii
                positions1 = state1.positions
                positions2 = state2.positions

                num_particles1 = len(radii1)
                num_particles2 = len(radii2)

                radii1_padded = np.pad(radii1, (0, max_particles1 - len(radii1)), 'constant')
                radii2_padded = np.pad(radii2, (0, max_particles2 - len(radii2)), 'constant')
                positions1_padded = np.pad(positions1, ((0, max_particles1 - len(positions1)), (0, 0)), 'constant')
                positions2_padded = np.pad(positions2, ((0, max_particles2 - len(positions2)), (0, 0)), 'constant')

                # Инициализация данных в Taichi полях
                ti_radii1.from_numpy(radii1_padded.astype(np.float32))
                ti_radii2.from_numpy(radii2_padded.astype(np.float32))
                ti_positions1.from_numpy(positions1_padded.astype(np.float32))
                ti_positions2.from_numpy(positions2_padded.astype(np.float32))
                ti_accum[None] = 0.0

                # Запускаем кернел для вычисления объёма пересечения
                ti_compute_intersection(num_particles1, num_particles2)

                # Рассчитываем W(t)
                similarity[ind_t] = ti_accum[None] / total_volume_geometric_mean
                
            # Рисуем график схожести
            plt.figure(figsize=(10, 6))
            plt.plot(times, similarity)
            plt.title('Similarity Measure W(t) Over Time')
            plt.xlabel('Time, s')
            plt.ylabel('W(t), 1')
            plt.xlim(times[0], times[-1])
            plt.ylim(0, 1.05)
            plt.grid(True)


            if plt_data_save:
                # Сохраняем в excel данные
                folder = 'results/exportet_data'
                fname_to_save = f'{folder}/{self.solution.get_name()}_vs_{other_solution.get_name()}___comparison.xlsx'
                if not os.path.exists(folder):
                    os.makedirs(folder)
                df = pd.DataFrame({'Time': times, 'Similarity': similarity})
                df.to_excel(fname_to_save, index=False)
                print(f'Data saved to {fname_to_save}')

            if plt_show:
                plt.show()


    def plot_radius_ratio_statistics(self, window_size = 10, overlap = 0.5, ratio_intervals = [2, 8], plt_show=True, plt_data_save=False):
        """
        Строит график долей соударений с различными соотношениями радиусов.
        
        :param window_size: Размер временного окна в секундах.
        :param overlap: Доля перекрытия окон (0 <= overlap < 1).
        :param ratio_intervals: Массив границ соотношений радиусов (например, [1.1, 1.5, 2, 5]).
        :param plt_show: Флаг для отображения графика (по умолчанию True).
        """
        # Получаем время начала и конца цепочки решений
        start_time = self.solution.get_chain_start_time()
        end_time = self.solution.get_chain_end_time()

        # Рассчитываем шаг между окнами
        step_size = window_size * (1 - overlap)

        # Временные промежутки, для которых будем строить статистику
        times = np.arange(start_time, end_time, step_size)
        
        # Подготовка структуры для хранения долей по каждому интервалу
        ratio_counts = {i: np.zeros(len(times)) for i in range(len(ratio_intervals) + 1)}

        # Первое решение
        first_solution = self.solution.get_first_solution()

        # Перебираем временные окна
        for idx, window_start in enumerate(times):

            window_end = window_start + window_size
            current_solution = first_solution

            # Проходим по всей цепочке решений и собираем столкновения в текущем временном окне
            while current_solution is not None:
                time = current_solution.get_last_time()

                if window_start <= time <= window_end and current_solution.is_collision:
                    # Считаем соотношения радиусов для всех столкновений
                    for pair in current_solution.collided_droplets:
                        r1 = max(current_solution.radii[pair[0]], current_solution.radii[pair[1]])
                        r2 = min(current_solution.radii[pair[0]], current_solution.radii[pair[1]])
                        ratio = r1 / r2

                        # Определяем, в какой интервал попадает соотношение радиусов
                        for i, interval in enumerate(ratio_intervals):
                            if ratio <= interval:
                                ratio_counts[i][idx] += 1
                                break
                        else:
                            # Если соотношение больше последнего интервала
                            ratio_counts[len(ratio_intervals)][idx] += 1
                elif time > window_end:
                    break

                current_solution = current_solution._next

            # Нормируем по числу столкновений в окне
            total_collisions = np.sum([ratio_counts[i][idx] for i in range(len(ratio_intervals) + 1)])
            if total_collisions > 0:
                for i in range(len(ratio_intervals) + 1):
                    ratio_counts[i][idx] /= total_collisions

        # Построение графиков
        plt.figure(figsize=(10, 6))
        for i, interval in enumerate(ratio_intervals):
            if i == 0:
                plt.plot(times + window_size / 2, ratio_counts[i], label=f'Ratio <= {interval}')
            else:
                plt.plot(times + window_size / 2, ratio_counts[i], label=f'{ratio_intervals[i-1]} < Ratio <= {interval}')
        plt.plot(times + window_size / 2, ratio_counts[len(ratio_intervals)], label=f'{ratio_intervals[-1]} < Ratio')

        plt.title('Ratio of Droplet Collisions Over Time')
        plt.xlabel('Time, s')
        plt.ylabel('Fraction of Collisions')
        plt.legend()
        plt.grid(True)

        if plt_data_save:
            # Сохраняем в excel данные
            folder = 'results/exportet_data'
            fname_to_save = f'{folder}/{self.solution.get_name()}___radius_ratio_statistics.xlsx'
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            df = pd.DataFrame({'Time': times + window_size / 2})
            for i, interval in enumerate(ratio_intervals):
                if i == 0:
                    df[f'Ratio <= {interval}'] = ratio_counts[i]
                else:
                    df[f'{ratio_intervals[i-1]} < Ratio <= {interval}'] = ratio_counts[i]
            df[f'{ratio_intervals[-1]} < Ratio'] = ratio_counts[len(ratio_intervals)]

            df.to_excel(fname_to_save, index=False)
            print(f'Data saved to {fname_to_save}')

        if plt_show:
            plt.show()


    def plot_angle_statistics(self, window_size=10, overlap=0.5, angle_intervals=[5, 10, 15], plt_show=True, plt_data_save=False):
        """
        Строит график долей соударений с различными углами между прямой, соединяющей капли, и осью z.
        
        :param window_size: Размер временного окна в секундах.
        :param overlap: Доля перекрытия окон (0 <= overlap < 1).
        :param angle_intervals: Массив границ углов (в градусах, например, [5, 10, 15, 25]).
        :param plt_show: Флаг для отображения графика (по умолчанию True).
        """
        start_time = self.solution.get_chain_start_time()
        end_time = self.solution.get_chain_end_time()
        step_size = window_size * (1 - overlap)
        times = np.arange(start_time, end_time, step_size)
        
        # Подготовка структуры для хранения долей по каждому интервалу
        angle_counts = {i: np.zeros(len(times)) for i in range(len(angle_intervals) + 1)}
        first_solution = self.solution.get_first_solution()

        for idx, window_start in enumerate(times):
            window_end = window_start + window_size
            current_solution = first_solution

            while current_solution is not None:
                time = current_solution.get_last_time()

                if window_start <= time <= window_end and current_solution.is_collision:
                    for pair in current_solution.collided_droplets:
                        
                        # Проверяем наличие предпоследнего шага
                        if current_solution.trajectories.shape[0] >= 2:
                            
                            # Получаем радиусы капель
                            r1 = current_solution.radii[pair[0]]
                            r2 = current_solution.radii[pair[1]]
                            collision_distance = r1 + r2

                            # Позиции на последнем и предпоследнем шагах
                            pos1_last = current_solution.trajectories[-1, pair[0], :]
                            pos2_last = current_solution.trajectories[-1, pair[1], :]
                            pos1_prev = current_solution.trajectories[-2, pair[0], :]
                            pos2_prev = current_solution.trajectories[-2, pair[1], :]

                             # Векторы смещения капель между последним и предпоследним шагом
                            delta1 = pos1_last - pos1_prev
                            delta2 = pos2_last - pos2_prev

                            # Текущее расстояние между каплями на последнем шаге
                            current_distance = np.linalg.norm(pos2_last - pos1_last)

                            # Линейная интерполяция для нахождения момента контакта
                            prev_distance = np.linalg.norm(pos2_prev - pos1_prev)
                            denom = current_distance - prev_distance
                            if abs(denom) > 1e-30:
                                t_contact = (current_distance - collision_distance) / denom
                                pos1_contact = pos1_prev + t_contact * delta1
                                pos2_contact = pos2_prev + t_contact * delta2
                                vector = pos2_contact - pos1_contact
                            else:
                                vector = pos2_last - pos1_last
                        
                        # Если нет предпоследнего шага, то вычисляем только по последнему
                        else:
                            pos1 = np.array([current_solution.trajectories[-1, pair[0], 0], 
                                            current_solution.trajectories[-1, pair[0], 1], 
                                            current_solution.trajectories[-1, pair[0], 2]])
                            pos2 = np.array([current_solution.trajectories[-1, pair[1], 0], 
                                            current_solution.trajectories[-1, pair[1], 1], 
                                            current_solution.trajectories[-1, pair[1], 2]])

                            # Вектор между двумя каплями
                            vector = pos2 - pos1


                        # Нормированный вектор z
                        z_vector = np.array([0, 0, 1])

                        # Вычисляем угол между вектором и осью z
                        cos_theta = np.dot(vector, z_vector) / (np.linalg.norm(vector) * np.linalg.norm(z_vector))
                        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                        if angle > 90:
                            angle = 180 - angle

                        # Определяем, в какой интервал попадает угол
                        for i, interval in enumerate(angle_intervals):
                            if angle <= interval:
                                angle_counts[i][idx] += 1
                                break
                        else:
                            # Если угол больше последнего интервала
                            angle_counts[len(angle_intervals)][idx] += 1
                        
                elif time > window_end:
                    break

                current_solution = current_solution._next

            # Нормируем по числу столкновений в окне
            total_collisions = np.sum([angle_counts[i][idx] for i in range(len(angle_intervals) + 1)])
            if total_collisions > 0:
                for i in range(len(angle_intervals) + 1):
                    angle_counts[i][idx] /= total_collisions

        # Построение графиков
        plt.figure(figsize=(10, 6))
        for i, interval in enumerate(angle_intervals):
            if i == 0:
                plt.plot(times + window_size / 2, angle_counts[i], label=f'Angle <= {interval}°')
            else:
                plt.plot(times + window_size / 2, angle_counts[i], label=f'{angle_intervals[i-1]}° < Angle <= {interval}°')
        plt.plot(times + window_size / 2, angle_counts[len(angle_intervals)], label=f'{angle_intervals[-1]}° < Angle')

        plt.title('Angle of Droplet Collisions Over Time')
        plt.xlabel('Time, s')
        plt.ylabel('Fraction of Collisions')
        plt.legend()
        plt.grid(True)

        if plt_data_save:
            # Сохраняем в excel данные
            folder = 'results/exportet_data'
            fname_to_save = f'{folder}/{self.solution.get_name()}___angle_statistics.xlsx'
            if not os.path.exists(folder):
                os.makedirs(folder)

            df = pd.DataFrame({'Time': times + window_size / 2})
            for i, interval in enumerate(angle_intervals):
                if i == 0:
                    df[f'Angle <= {interval}°'] = angle_counts[i]
                else:
                    df[f'{angle_intervals[i-1]}° < Angle <= {interval}°'] = angle_counts[i]
            df[f'{angle_intervals[-1]}° < Angle'] = angle_counts[len(angle_intervals)]

            df.to_excel(fname_to_save, index=False)
            print(f'Data saved to {fname_to_save}')

        if plt_show:
            plt.show()


    def plot_angle_statistics_in_given_intervals(self, windows, angle_intervals=[5, 10, 15], plt_show=True, plt_data_save=False):
        """
        Строит график долей соударений с различными углами между прямой, соединяющей капли, и осью z для заданных временных интервалов.
        
        :param windows: Массив границ временных интервалов (например, [0, 20, 100, 200]).
        :param angle_intervals: Массив границ углов (в градусах, например, [5, 10, 15, 25]).
        :param plt_show: Флаг для отображения графика (по умолчанию True).
        """
        start_time = self.solution.get_chain_start_time()
        end_time = self.solution.get_chain_end_time()

        # Убедимся, что первый элемент массива windows не меньше start_time
        if windows[0] < start_time:
            windows[0] = start_time

        # Убедимся, что последний элемент массива windows не превышает end_time
        if windows[-1] > end_time:
            windows[-1] = end_time

        # Подготовка структуры для хранения долей по каждому интервалу
        angle_counts = {i: np.zeros(len(windows) - 1) for i in range(len(angle_intervals) + 1)}
        first_solution = self.solution.get_first_solution()

        for idx, (window_start, window_end) in enumerate(zip(windows[:-1], windows[1:])):
            current_solution = first_solution

            while current_solution is not None:
                time = current_solution.get_last_time()

                if window_start <= time <= window_end and current_solution.is_collision:
                    for pair in current_solution.collided_droplets:
                        
                        # Проверяем наличие предпоследнего шага
                        if current_solution.trajectories.shape[0] >= 2:
                            
                            # Получаем радиусы капель
                            r1 = current_solution.radii[pair[0]]
                            r2 = current_solution.radii[pair[1]]
                            collision_distance = r1 + r2

                            # Позиции на последнем и предпоследнем шагах
                            pos1_last = current_solution.trajectories[-1, pair[0], :]
                            pos2_last = current_solution.trajectories[-1, pair[1], :]
                            pos1_prev = current_solution.trajectories[-2, pair[0], :]
                            pos2_prev = current_solution.trajectories[-2, pair[1], :]

                            # Векторы смещения капель между последним и предпоследним шагом
                            delta1 = pos1_last - pos1_prev
                            delta2 = pos2_last - pos2_prev

                            # Текущее расстояние между каплями на последнем шаге
                            current_distance = np.linalg.norm(pos2_last - pos1_last)

                            # Линейная интерполяция для нахождения момента контакта
                            t_contact = (current_distance - collision_distance) / (current_distance - np.linalg.norm(pos2_prev - pos1_prev))

                            # Вычисляем позиции капель в момент контакта
                            pos1_contact = pos1_prev + t_contact * delta1
                            pos2_contact = pos2_prev + t_contact * delta2

                            # Вектор между каплями в момент контакта
                            vector = pos2_contact - pos1_contact
                        
                        # Если нет предпоследнего шага, то вычисляем только по последнему
                        else:
                            pos1 = np.array([current_solution.trajectories[-1, pair[0], 0], 
                                            current_solution.trajectories[-1, pair[0], 1], 
                                            current_solution.trajectories[-1, pair[0], 2]])
                            pos2 = np.array([current_solution.trajectories[-1, pair[1], 0], 
                                            current_solution.trajectories[-1, pair[1], 1], 
                                            current_solution.trajectories[-1, pair[1], 2]])

                            # Вектор между двумя каплями
                            vector = pos2 - pos1

                        # Нормированный вектор z
                        z_vector = np.array([0, 0, 1])

                        # Вычисляем угол между вектором и осью z
                        cos_theta = np.dot(vector, z_vector) / (np.linalg.norm(vector) * np.linalg.norm(z_vector))
                        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                        if angle > 90:
                            angle = 180 - angle

                        # Определяем, в какой интервал попадает угол
                        for i, interval in enumerate(angle_intervals):
                            if angle <= interval:
                                angle_counts[i][idx] += 1
                                break
                        else:
                            # Если угол больше последнего интервала
                            angle_counts[len(angle_intervals)][idx] += 1
                        
                elif time > window_end:
                    break

                current_solution = current_solution._next

            # Нормируем по числу столкновений в окне
            total_collisions = np.sum([angle_counts[i][idx] for i in range(len(angle_intervals) + 1)])
            if total_collisions > 0:
                for i in range(len(angle_intervals) + 1):
                    angle_counts[i][idx] /= total_collisions

        # Построение графиков
        plt.figure(figsize=(10, 6))
        for i, interval in enumerate(angle_intervals):
            if i == 0:
                plt.plot((np.array(windows[:-1]) + np.array(windows[1:])) / 2, angle_counts[i], label=f'Angle <= {interval}°')
            else:
                plt.plot((np.array(windows[:-1]) + np.array(windows[1:])) / 2, angle_counts[i], label=f'{angle_intervals[i-1]}° < Angle <= {interval}°')
        plt.plot((np.array(windows[:-1]) + np.array(windows[1:])) / 2, angle_counts[len(angle_intervals)], label=f'{angle_intervals[-1]}° < Angle')

        plt.title('Angle of Droplet Collisions Over Time')
        plt.xlabel('Time, s')
        plt.ylabel('Fraction of Collisions')
        plt.legend()
        plt.grid(True)

        if plt_data_save:
            # Сохраняем в excel данные
            folder = 'results/exportet_data'
            fname_to_save = f'{folder}/{self.solution.get_name()}___angle_statistics_in_given_intervals.xlsx'
            if not os.path.exists(folder):
                os.makedirs(folder)

            #df = pd.DataFrame({'Time': (np.array(windows[:-1]) + np.array(windows[1:])) / 2})
            df = pd.DataFrame({'Start time': np.array(windows[:-1]), 'End time': np.array(windows[1:])})
            for i, interval in enumerate(angle_intervals):
                if i == 0:
                    df[f'Angle <= {interval}°'] = angle_counts[i]
                else:
                    df[f'{angle_intervals[i-1]}° < Angle <= {interval}°'] = angle_counts[i]
            df[f'{angle_intervals[-1]}° < Angle'] = angle_counts[len(angle_intervals)]

            df.to_excel(fname_to_save, index=False)
            print(f'Data saved to {fname_to_save}')

        if plt_show:
            plt.show()


    def plot_angle_histogram(self, num_bins=19, range=(0, 90), plt_show=True):
        """
        Строит гистограмму по углам между линией, соединяющей капли, и осью z за всё время моделирования.
        
        :param num_bins: Количество интервалов для гистограммы (по умолчанию 10).
        :param plt_show: Флаг для отображения гистограммы (по умолчанию True).
        """
        # Подготовка списка для хранения углов
        all_angles = []
        
        # Первое решение
        current_solution = self.solution.get_first_solution()

        # Проходим по всей цепочке решений
        while current_solution is not None:
            if current_solution.is_collision:
                for pair in current_solution.collided_droplets:
                    
                    # Проверяем наличие предпоследнего шага
                    if current_solution.trajectories.shape[0] >= 2:
                        r1 = current_solution.radii[pair[0]]
                        r2 = current_solution.radii[pair[1]]
                        collision_distance = r1 + r2

                        pos1_last = current_solution.trajectories[-1, pair[0], :]
                        pos2_last = current_solution.trajectories[-1, pair[1], :]
                        pos1_prev = current_solution.trajectories[-2, pair[0], :]
                        pos2_prev = current_solution.trajectories[-2, pair[1], :]

                        delta1 = pos1_last - pos1_prev
                        delta2 = pos2_last - pos2_prev

                        current_distance = np.linalg.norm(pos2_last - pos1_last)

                        prev_distance = np.linalg.norm(pos2_prev - pos1_prev)
                        denom = current_distance - prev_distance
                        if abs(denom) > 1e-30:
                            t_contact = (current_distance - collision_distance) / denom
                            pos1_contact = pos1_prev + t_contact * delta1
                            pos2_contact = pos2_prev + t_contact * delta2
                            vector = pos2_contact - pos1_contact
                        else:
                            vector = pos2_last - pos1_last
                    
                    else:
                        pos1 = np.array([current_solution.trajectories[-1, pair[0], 0], 
                                        current_solution.trajectories[-1, pair[0], 1], 
                                        current_solution.trajectories[-1, pair[0], 2]])
                        pos2 = np.array([current_solution.trajectories[-1, pair[1], 0], 
                                        current_solution.trajectories[-1, pair[1], 1], 
                                        current_solution.trajectories[-1, pair[1], 2]])

                        vector = pos2 - pos1

                    z_vector = np.array([0, 0, 1])

                    cos_theta = np.dot(vector, z_vector) / (np.linalg.norm(vector) * np.linalg.norm(z_vector))
                    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    if angle > 90:
                        angle = 180 - angle

                    # Сохраняем вычисленный угол
                    all_angles.append(angle)

            current_solution = current_solution._next

        # Построение гистограммы углов
        plt.figure(figsize=(10, 6))
        hist, bin_edges = np.histogram(all_angles, bins=num_bins, range=range)

        # Отрисовка гистограммы
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", align="edge")
        
        # Оформление графика
        plt.title('Histogram of Droplet Collision Angles')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Number of Collisions')
        
        if plt_show:
            plt.show()
