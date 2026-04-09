import numpy as np
import pandas as pd
from .particle_state_base import ParticleState

class DropletState(ParticleState):
    def __init__(self, positions=None, radii=None, time=0, filename=None):
        """
        Класс для хранения единичного состояния системы капель.
        
        Инициализация может происходить через передачу:
        - Позиции, радиусы и время,
        - Путь к файлу (.npz или .xlsx).
        
        :param positions: Позиции частиц (список или numpy-массив).
        :param radii: Радиусы частиц (список или numpy-массив).
        :param time: Время состояния системы.
        :param filename: Имя файла для загрузки состояния (если передан).
        """
        self.positions = None
        self.radii = None
        self.time = 0

        # Инициализация через позиции, радиусы и время
        if positions is not None and radii is not None:
            self.positions = np.array(positions)  # Массив координат частиц
            self.radii = np.array(radii)  # Массив радиусов частиц
            self.time = time  # Текущее время

        # Инициализация через файл
        elif filename is not None:
            if filename.endswith('.xlsx'):
                self.import_from_xlsx(filename)  # Загрузка из Excel-файла
            elif filename.endswith('.npz'):
                self.load(filename)  # Загрузка из файла .npz
            else:
                raise ValueError("Файл должен иметь расширение .xlsx или .npz.")
        else:
            raise ValueError("Необходимо передать либо позиции и радиусы, либо имя файла.")

        
    def copy(self):
        """
        Создание копии текущего состояния.
        
        :return: Копия объекта DropletState.
        """
        return DropletState(self.positions.copy(), self.radii.copy(), self.time)

    def save(self, filename):
        """
        Сохранение текущего состояния в файл.

        :param filename: Имя файла для сохранения.
        """
        np.savez(filename, positions=self.positions, radii=self.radii, time=self.time)

    def load(self, filename):
        """
        Загрузка состояния из файла.

        :param filename: Имя файла для загрузки.
        """
        data = np.load(filename)
        self.positions = data['positions']
        self.radii = data['radii']
        self.time = data['time']

    def export_to_xlsx(self, filename):
        """
        Экспорт текущего состояния в Excel.
        Время не экспортируется.

        :param filename: Имя файла для экспорта.
        """

        df = pd.DataFrame({'x': self.positions[:, 0], 'y': self.positions[:, 1], 'z': self.positions[:, 2], 'r': self.radii})
        df.to_excel(filename, index=False)

    def import_from_xlsx(self, filename):
        """
        Импорт состояния из Excel.
        Время не импортируется.

        :param filename: Имя файла для импорта.
        """
        df = pd.read_excel(filename)
        self.positions = np.array(df[['x', 'y', 'z']])
        self.radii = np.array(df['r'])
        self.time = 0


    def __repr__(self):
        return f"<DropletState: time={self.time}, num_particles={len(self.radii)}>"
