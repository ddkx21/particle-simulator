import time
import os.path
import numpy as np
from particle_state import DropletState
from .solution_base import Solution

class DropletSolution(Solution):
    def __init__(self, initial_droplet_state, real_time_visualization=False, update_interval=0.05, length=10, previous=None, filename=None):
        """
        Инициализация объекта Solution для хранения траекторий частиц.
        
        :param droplet_state: Объект для хранения нчального состояния системы капель.
        :param real_time_visualization: Флаг отображения графика в реальном времени. Замедляет работу программы!
        :param update_interval: Интервал обновления графика в секундах.
        """

        self.initial_droplet_state = initial_droplet_state

        self.num_particles = len(initial_droplet_state.radii)
        self.radii = initial_droplet_state.radii

        self.length = length

        self.real_time_visualization = real_time_visualization
        self.update_interval = update_interval

        # Массив для хранения отсчетов времени (length x 1)
        self.times = np.zeros((self.length, 1)) + initial_droplet_state.time

        # Массив для хранения траекторий частиц (length x num_particles x 3)
        self.trajectories = np.zeros((self.length, self.num_particles, 3))

        # Переменная для отслеживания последнего времени обновления графика
        self.last_update_time = time.time()

        # Ссылки на предыдущее и следующее решения
        self._prev = previous
        self._next = None
        
        # Данные о столкновениях
        self.is_collision = False
        self.collided_droplets = [] # Индексы столкнувшихся частиц в этом решении
        self.resulting_droplet = [] # Индекс образовавшийся частицы в следующем решении

        # Текущий шаг симуляции
        self.current_step = -1
        self.save_step(initial_droplet_state.time, initial_droplet_state.positions)

        # Из какого файла загружено решение
        self.filename = filename

    def save_step(self, t, positions):
        """
        Сохранение текущих позиций частиц на данном шаге симуляции.

        :param positions: Массив текущих координат частиц (num_particles x 3).
        """
        self.current_step += 1

        if self.current_step >= self.length:
            # Удваиваем длину буферов через np.empty + slice copy.
            # В отличие от np.concatenate это не держит одновременно
            # старый+новый+результат: старые массивы освобождаются сразу
            # после копирования, что снижает пик RAM на больших N.
            new_length = self.length * 2

            new_times = np.empty((new_length, 1), dtype=self.times.dtype)
            new_times[: self.length] = self.times
            self.times = new_times

            new_traj = np.empty(
                (new_length, self.num_particles, 3),
                dtype=self.trajectories.dtype,
            )
            new_traj[: self.length] = self.trajectories
            self.trajectories = new_traj

            self.length = new_length

        self.times[self.current_step] = t
        self.trajectories[self.current_step] = positions

    def get_times(self):
        """
        Получение временных отсчетов.

        :return: Массив временных отсчетов (length x 1).
        """
        # Возвращаем только записанные шаги (по current_step)
        return self.times[:self.current_step + 1]

    def get_trajectories(self):
        """
        Получение всех траекторий частиц.

        :return: Массив траекторий частиц (num_steps x num_particles x 3).
        """
        # Возвращаем только записанные шаги (по current_step)
        return self.trajectories[:self.current_step + 1]

    def get_current_time(self):
        """ 
        Получение текущего времени симуляции.

        :return: Текущее время симуляции.
        """
        return self.times[self.current_step][0]


    def get_last_time(self):
        """
        Получение времени последнего шага симуляции.
        В поспроцессоре лучше использовать эту функцию вместо get_current_time

        :return: Время последнего шага симуляции.
        """
        # Поиск последнего ненулевого отсчёта времеин self.times
        valid_steps = np.where(np.any(self.times != 0, axis=0))[0]
        return self.times[valid_steps[-1]]


    def get_chain_start_time(self):
        """
        Получение времени начала цепи.

        :return: Время начала цепи.
        """
        current_solution = self
        while current_solution._prev is not None:
            current_solution = current_solution._prev
        return current_solution.times[0][0]
    
    def get_chain_end_time(self):
        """
        Получение времени окончания цепи.

        :return: Время окончания цепи.
        """
        current_solution = self
        while current_solution._next is not None:
            current_solution = current_solution._next
        return current_solution.times[-1][0]

    def get_first_solution(self):
        """
        Получение первого решения цепи.

        :return: Первое решение цепи.
        """
        current_solution = self
        while current_solution._prev is not None:
            current_solution = current_solution._prev
        return current_solution
    
    def get_current_positions(self):
        """
        Получение текущих позиций всех частиц.

        :return: Массив текущих позиций частиц (num_particles x 3).
        """
        return self.trajectories[self.current_step]

    def compact(self):
        """
        Сжатие массивов позиций и временных отсчетов.
        """
        self.times = self.get_times()
        self.trajectories = self.get_trajectories()

    def compact_all(self):
        """
        Сжатие всех массивов позиций и временных отсчетов.
        """
        self.compact()

        next = self._next
        while next is not None:
            next.compact()
            next = next._next
        
        prev = self._prev
        while prev is not None:
            prev.compact()
            prev = prev._prev


    def should_update_visualization(self):
        """
        Проверка, нужно ли обновить график в реальном времени.

        :return: True, если нужно обновить график.
        """
        if self.real_time_visualization:
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self.last_update_time = current_time
                return True
        return False

    def generate_next_solution(self):
        """
        Генерация следующего решения после нескольких столкновений.
        
        :return: Следующее решение.
        """
        # Массивы новых радиусов и позиций капель
        new_radii = np.copy(self.radii)
        new_positions = np.copy(self.trajectories[self.current_step])

        # Для сохранения новых капель
        new_droplets_radii = []
        new_droplets_positions = []
        
        # Для сохранения индексов новых капель
        resulting_droplet = []

        # Множество индексов капель, которые уже участвовали в столкновениях
        collided_indices = set()

        # Упорядочивание столкновений по индексу первой частицы из пары
        sorted_indices = np.argsort(self.collided_droplets[:, 0])
        self.collided_droplets = self.collided_droplets[sorted_indices]

        # Проходим по всем столкновениям
        for idx1, idx2 in self.collided_droplets:
            if idx1 in collided_indices or idx2 in collided_indices:
                # Пропускаем столкновения, если одна из капель уже участвовала в предыдущем
                continue

            # Получаем радиусы и позиции столкнувшихся капель
            radius1, radius2 = new_radii[idx1], new_radii[idx2]
            position1, position2 = new_positions[idx1], new_positions[idx2]

            radius1_cube = radius1**3
            radius2_cube = radius2**3

            # Новый радиус исходя из сохранения объема
            new_radius = (radius1_cube + radius2_cube)**(1/3)

            # Новая позиция капли (центр масс)
            new_position = (position1 * radius1_cube + position2 * radius2_cube) / (radius1_cube + radius2_cube)

            # Сохраняем радиусы и позиции новых капель в отдельные массивы
            new_droplets_radii.append(new_radius)
            new_droplets_positions.append(new_position)

            # Обозначаем капли как обработанные
            collided_indices.add(idx1)
            collided_indices.add(idx2)

        # Удаляем столкнувшиеся капли из массива радиусов и позиций
        indices_to_remove = sorted(collided_indices, reverse=True)
        new_radii = np.delete(new_radii, indices_to_remove)
        new_positions = np.delete(new_positions, indices_to_remove, axis=0)

        # Добавляем новые капли в массив радиусов и позиций
        new_radii = np.append(new_radii, new_droplets_radii)
        new_positions = np.vstack([new_positions, new_droplets_positions])

        # Сохраняем индексы новых капель
        resulting_droplet.extend(range(len(new_radii) - len(new_droplets_radii), len(new_radii)))

        # Создаем новое решение, связываем с текущим
        new_initial_state = DropletState(new_positions, new_radii, self.times[self.current_step])
        next_solution = DropletSolution(
            initial_droplet_state=new_initial_state,
            real_time_visualization=self.real_time_visualization,
            update_interval=self.update_interval,
            length=self.length,
            previous=self
        )
        self._next = next_solution

        # Устанавливаем флаг столкновения
        next_solution.is_collision = False

        return next_solution


    def save_to_file(self, filename):
        """
        Сохранение решения (траекторий) в файл.

        :param filename: Имя файла для сохранения решения.
        """
        np.savez(filename, 
                 num_particles=self.num_particles,
                 radii=self.radii,
                 times=self.get_times(), 
                 trajectories=self.get_trajectories()
                 )
        print(f"Решение сохранено в файл {filename}.")

    def load_from_file(self, filename):

        """
        Загрузка решения (траекторий) из файла.

        :param filename: Имя файла для загрузки решения.
        """
        # Не подгружает другие поля, надо доработать
        data = np.load(filename)
        self.num_particles = data['num_particles']
        self.radii = data['radii']
        self.times = data['times']
        self.trajectories = data['trajectories']
        self.length = len(self.times)

        # Измвлекаем имя файла
        fname = os.path.basename(filename)
        fname = os.path.splitext(fname)[0]
        self.filename = fname

        print(f"Решение загружено из файла {filename}.")

    def save_chain_to_file(self, filename, precision='float64'):
        """
        Сохранение всей цепочки решений в файл.
        
        :param filename: Имя файла для сохранения всей цепочки решений.
        :param precision: Точность сохранения данных ('float64', 'float32' или 'float16').
        """

        # Определяем numpy тип на основе заданной точности
        if precision == 'float32':
            dtype = np.float32
        elif precision == 'float16':
            dtype = np.float16
        else:
            dtype = np.float64

        # Находим первое решение
        first_solution = self
        while first_solution._prev is not None:
            first_solution = first_solution._prev

        # Собираем данные всей цепочки решений в списки
        solution_list = []
        current_solution = first_solution
        while current_solution is not None:
            current_solution.compact()
            solution_data = {
                'num_particles': current_solution.num_particles,
                'radii': current_solution.radii.astype(dtype),
                'length': current_solution.length,
                'real_time_visualization': current_solution.real_time_visualization,
                'update_interval': current_solution.update_interval,
                'times': current_solution.get_times().astype(dtype),
                'trajectories': current_solution.get_trajectories().astype(dtype),
                'is_collision': current_solution.is_collision,
                'collided_droplets': current_solution.collided_droplets,
                'resulting_droplet': current_solution.resulting_droplet
            }
            solution_list.append(solution_data)
            current_solution = current_solution._next

        # Сохраняем все решения в один .npz файл
        np.savez(filename, solutions=solution_list)
        print(f"Цепочка решений сохранена в файл {filename}.")

    @staticmethod
    def load_chain_from_file(filename):
        """
        Загрузка всей цепочки решений из файла.
        
        :param filename: Имя файла для загрузки всей цепочки решений.
        :return: Первый объект DropletSolution.
        """
        
        # Измвлекаем имя файла
        fname = os.path.basename(filename)
        fname = os.path.splitext(fname)[0]
        
        # Загружаем данные из файла
        data = np.load(filename, allow_pickle=True)
        solution_list = data['solutions']

        # Восстанавливаем цепочку решений
        first_solution = None
        previous_solution = None

        for solution_data in solution_list:
            # Восстановление начального состояния для каждого решения
            initial_state = DropletState(
                solution_data['trajectories'][0],
                solution_data['radii'], 
                solution_data['times'][0])

            # Создаем новое решение
            new_solution = DropletSolution(
                initial_droplet_state=initial_state,
                real_time_visualization=solution_data['real_time_visualization'],
                update_interval=solution_data['update_interval'],
                length=solution_data['length'],
                previous=previous_solution,
                filename=fname
            )
            # Заполняем атрибуты решения
            new_solution.times = solution_data['times']
            new_solution.trajectories = solution_data['trajectories']
            new_solution.is_collision = solution_data['is_collision']
            new_solution.collided_droplets = solution_data['collided_droplets']
            new_solution.resulting_droplet = solution_data['resulting_droplet']

            if previous_solution is not None:
                previous_solution._next = new_solution
            else:
                first_solution = new_solution  # Сохраняем первое решение

            previous_solution = new_solution

        print(f"Цепочка решений загружена из файла {filename}.")
        return first_solution


    def get_state(self, target_time):
        """
        Возвращает объект DropletState для указанного времени с интерполяцией, если время не точно попадает на шаг.

        :param target_time: Время, для которого нужно получить состояние капель.
        :return: Объект DropletState.
        :raises ValueError: Если время выходит за пределы всей цепочки решений.
        """
        current_solution = self

        # Цикл для поиска решения, которое содержит target_time
        while True:
            min_time = current_solution.times[0][0]
            max_time = current_solution.times[-1][0]

            if target_time < min_time:
                if current_solution._prev is not None:
                    current_solution = current_solution._prev
                else:
                    raise ValueError(f"Время {target_time} меньше, чем минимальное время в цепочке решений.")
            elif target_time > max_time:
                if current_solution._next is not None:
                    current_solution = current_solution._next
                else:
                    raise ValueError(f"Время {target_time} больше, чем максимальное время в цепочке решений.")
            else:
                break  # Время находится внутри текущего решения

        # Время внутри текущего решения, находим индексы соседних шагов
        times = current_solution.times.flatten()
        idx_before = np.searchsorted(times, target_time) - 1
        idx_after = idx_before + 1

        if idx_before < 0:
            idx_before = 0
        if idx_after >= len(times):
            idx_after = len(times) - 1

        time_before = times[idx_before]
        time_after = times[idx_after]

        positions_before = current_solution.trajectories[idx_before]
        positions_after = current_solution.trajectories[idx_after]

        if target_time == time_after:
            # Если время попало на шаг, нет смысла интерполировать
            interpolated_positions = positions_after
        elif target_time == time_before:
            # Если время попало на шаг, нет смысла интерполировать
            interpolated_positions = positions_before
        else:
            # Интерполяция по времени
            factor = (target_time - time_before) / (time_after - time_before)
            interpolated_positions = positions_before + factor * (positions_after - positions_before)

        # Возвращаем интерполированный DropletState
        return DropletState(positions=interpolated_positions, radii=current_solution.radii, time=target_time)


    def get_name(self):
        """
        Возвращает имя цепочки решений.

        :return: Имя цепочки решений.
        """
        first_solution = self.get_first_solution()
        if first_solution.filename is not None:
            return first_solution.filename
        else:
            return ''