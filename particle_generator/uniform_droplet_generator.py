import numpy as np
import pandas as pd
import taichi as ti
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from particle_state import DropletState
from .particle_generator_base import ParticleGenerator

@ti.data_oriented
class UniformDropletGenerator(ParticleGenerator):
    def __init__(self, coord_range=(0, 10), radii_range=(1, 100), num_particles=1, minimum_distance=0):
        """
        Инициализация генератора капель.

        :param coord_range: Диапазон координат (min, max)
        :param radii_range: Диапазон радиусов (min, max)
        :param num_particles: Количество капель
        :param minimum_distance: Минимальное расстояние между каплями
        """

        self.num_particles = num_particles
        self.coord_range = coord_range
        self.radii_range = radii_range
        self.minimum_distance = minimum_distance
        self.droplet_state = None # Текущее состояние капель (объект DropletState)

        # Создаем поля Taichi
        self.ti_n_of_overlaps = ti.field(dtype=ti.i32, shape=())
        self.ti_is_overlapping = ti.field(dtype=ti.i32, shape=num_particles)
        self.ti_overlapping_droplets = ti.field(dtype=ti.i32, shape=num_particles)
        self.ti_num_particles = ti.field(dtype=ti.i32, shape=())
        self.ti_radii = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_minimum_distance = ti.field(dtype=ti.f64, shape=())
        self.ti_xs = ti.field(dtype=ti.f64, shape=num_particles)   
        self.ti_ys = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_zs = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_xlims = ti.field(dtype=ti.f64, shape=(2, ))
        self.ti_ylims = ti.field(dtype=ti.f64, shape=(2, ))
        self.ti_zlims = ti.field(dtype=ti.f64, shape=(2, ))


    def generate(self):
        """
        Генерация случайных координат и радиусов капель.

        :return: Объект DropletState
        """

        # Генерация массива радиусов и координат капель
        #print(f"Генерация {self.num_particles} капель")
        radii = np.random.uniform(self.radii_range[0], self.radii_range[1], size=self.num_particles)
        xs = np.random.uniform(self.coord_range[0], self.coord_range[1], size=self.num_particles)
        ys = np.random.uniform(self.coord_range[0], self.coord_range[1], size=self.num_particles)
        zs = np.random.uniform(self.coord_range[0], self.coord_range[1], size=self.num_particles)

        # Обновляем ti поля с данными
        self.ti_n_of_overlaps[None] = self.num_particles
        self.ti_is_overlapping.from_numpy(np.ones(self.num_particles, dtype=np.int32))
        self.ti_overlapping_droplets.from_numpy(np.arange(self.num_particles, dtype=np.int32))
        self.ti_num_particles[None] = self.num_particles
        self.ti_radii.from_numpy(radii)
        self.ti_minimum_distance[None] = self.minimum_distance
        self.ti_xs.from_numpy(xs)
        self.ti_ys.from_numpy(ys)
        self.ti_zs.from_numpy(zs)
        self.ti_xlims[0] = self.coord_range[0]
        self.ti_xlims[1] = self.coord_range[1]
        self.ti_ylims[0] = self.coord_range[0]
        self.ti_ylims[1] = self.coord_range[1]
        self.ti_zlims[0] = self.coord_range[0]  
        self.ti_zlims[1] = self.coord_range[1]

        # Устраняем пересечения
        attempts = 1000
        while self.ti_n_of_overlaps[None] > 0:
            if attempts == 0:
                raise ValueError("Не удалось устранить пересечения при генерациии капель!")
            self.reseed_overlaping_droplets(self.ti_xs, 
                                            self.ti_ys, 
                                            self.ti_zs, 
                                            self.ti_xlims, 
                                            self.ti_ylims, 
                                            self.ti_zlims, 
                                            self.ti_radii,
                                            self.ti_minimum_distance,
                                            self.ti_num_particles, 
                                            self.ti_n_of_overlaps,  
                                            self.ti_is_overlapping,
                                            self.ti_overlapping_droplets)
            attempts -= 1
            #print(f"Пересечений капель: {self.ti_n_of_overlaps[None]}")   
            
        # Копируем обратно в numpy
        positions = np.stack((self.ti_xs.to_numpy(), self.ti_ys.to_numpy(), self.ti_zs.to_numpy()), axis=1)

        # Создаем и возвращаем объект состояния капель
        self.droplet_state = DropletState(positions, radii, 0)
        return self.droplet_state

    @ti.kernel
    def reseed_overlaping_droplets(self,
                         ti_x: ti.template(),
                         ti_y: ti.template(),
                         ti_z: ti.template(),
                         ti_xlims: ti.template(),
                         ti_ylims: ti.template(),
                         ti_zlims: ti.template(),
                         ti_radii: ti.template(),
                         ti_minimum_distance: ti.template(),
                         ti_num_particles: ti.template(),
                         ti_n_of_overlaps: ti.template(),
                         ti_is_overlapping: ti.template(),
                         ti_overlapping_droplets: ti.template()):
    

        # Перегенерируем положения пересекающихся капель
        for ovrlap_ind in range(ti_n_of_overlaps[None]):
            dr1_ind = ti_overlapping_droplets[ovrlap_ind]
            ti_x[dr1_ind] = ti.random() * (ti_xlims[1] - ti_xlims[0]) + ti_xlims[0]
            ti_y[dr1_ind] = ti.random() * (ti_ylims[1] - ti_ylims[0]) + ti_ylims[0]
            ti_z[dr1_ind] = ti.random() * (ti_zlims[1] - ti_zlims[0]) + ti_zlims[0]
            ti_is_overlapping[dr1_ind] = 0

        # Проверяем, остались ли пересечения
        for ovrlap_ind in range(ti_n_of_overlaps[None]):
            dr1_ind = ti_overlapping_droplets[ovrlap_ind]
            x1 = ti_x[dr1_ind]
            y1 = ti_y[dr1_ind]
            z1 = ti_z[dr1_ind]
            radius1 = ti_radii[dr1_ind]

            for dr2_ind in range(ti_num_particles[None]):

                if dr1_ind == dr2_ind:
                    continue

                x2 = ti_x[dr2_ind]
                y2 = ti_y[dr2_ind]
                z2 = ti_z[dr2_ind]
                radius2 = ti_radii[dr2_ind]

                distance_squared = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
                min_distance_squared = (radius1 + radius2 + ti_minimum_distance[None])**2

                if distance_squared < min_distance_squared:
                    ti_is_overlapping[dr1_ind] = 1
                    break

        # Обновляем список пересекающихся капель
        ti.loop_config(parallelize=1)  # Отключаем распараллеливание для следующего цикла
        new_n_of_overlaps = 0
        for dr1_ind in range(ti_num_particles[None]):
            if ti_is_overlapping[dr1_ind] == 1:
                ti_overlapping_droplets[new_n_of_overlaps] = dr1_ind
                new_n_of_overlaps = new_n_of_overlaps + 1
        ti_n_of_overlaps[None] = new_n_of_overlaps



    def save(self, filename):
        """
        Сохранение координат и радиусов капель в файл Excel.

        :param filename: Имя файла
        """
        if self.droplet_state is None:
            print("Сначала сгенерируйте капли с помощью метода generate().")
            return

        # Создаём DataFrame для хранения координат и радиусов
        positions = self.droplet_state.positions
        radii = self.droplet_state.radii
        df = pd.DataFrame({
            'X': positions[:, 0],
            'Y': positions[:, 1],
            'Z': positions[:, 2],
            'Radius': radii
        })
        
        # Сохранение в файл Excel
        df.to_excel(filename, index=False)
        print(f"Данные сохранены в файл {filename}.")

    def load(self, filename):
        """
        Загрузка координат и радиусов капель из файла Excel.

        :param filename: Имя файла
        """
        df = pd.read_excel(filename)
        positions = np.array(df[['X', 'Y', 'Z']])
        radii = np.array(df['Radius'])
        self.num_particles = len(positions)
        self.droplet_state = DropletState(positions, radii, 0)
        print(f"Данные загружены из файла {filename}.")

    def plot(self):
        """
        Визуализация капель в виде окружностей в 3D пространстве.
        """
        if self.droplet_state is None:
            print("Сначала сгенерируйте капли с помощью метода generate().")
            return
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlim(*self.coord_range)
        self.ax.set_ylim(*self.coord_range)
        self.ax.set_zlim(*self.coord_range)
        self.ax.set_title('Particle Distribution')


        # Создаем три окружности для каждой капли
        self.circles = []
        for pos, r in zip(self.droplet_state.positions, self.droplet_state.radii):
            circle_set = self.create_circle_set(pos, r)
            self.circles.append(circle_set)
            for circle in circle_set:
                self.ax.add_collection3d(circle)

        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_zlabel('Z Coordinate')

        plt.show()

    def create_circle_set(self, position, radius):
        """
        Создание набора из трех окружностей для отображения капли в трех основных плоскостях.

        :param position: Позиция центра окружности (x, y, z)
        :param radius: Радиус окружности
        :return: Список объектов Line3DCollection
        """
        circle_sets = []
        u = np.linspace(0, 2 * np.pi, 20)

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
    
    def print(self):
        water_content = np.sum(self.droplet_state.radii**3 * np.pi * 4 / 3) / (self.coord_range[1]-self.coord_range[0])**3

        print(f"\nКоличество капель: {self.num_particles}")
        print(f"Минимальный радиус капли: {np.min(self.droplet_state.radii)}")
        print(f"Максимальный радиус капли: {np.max(self.droplet_state.radii)}")
        print(f"Средний радиус капли: {np.mean(self.droplet_state.radii)}" )
        print(f"Объёмная доля воды: {water_content:.6f}")
        print(f"Размеры области: {self.coord_range}\n")


