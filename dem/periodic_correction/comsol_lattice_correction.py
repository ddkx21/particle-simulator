import json
import os

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator


class COMSOLLatticeCorrection:
    """
    Загрузка и интерполяция поля скоростей периодической поправки из данных COMSOL.

    COMSOL посчитал поле скоростей от одной капли в периодическом домене
    и вычел свободнопространственный стоклет. Результат — чистая поправка
    от периодических образов (поля u2, v2, w2 в U_lattice.txt).

    Данные хранятся на нерегулярной сетке в четверть-ячейке [0, L/2]² × [-L/2, L/2].
    Класс зеркалирует данные в полную ячейку [-L/2, L/2]³ и пересэмплирует
    на регулярную сетку для быстрой трилинейной интерполяции в Taichi.
    """

    # Путь к директории data/ относительно этого файла
    _DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    @classmethod
    def load_default(cls):
        """Загрузка из встроенной директории periodic_correction/data/."""
        return cls.from_data_dir(cls._DEFAULT_DATA_DIR)

    @classmethod
    def from_data_dir(cls, data_dir: str):
        """
        Создание из директории с данными COMSOL (U_lattice.txt + comsol_params.json).
        Все параметры (включая grid_resolution) читаются из comsol_params.json.

        :param data_dir: путь к директории
        """
        params_path = os.path.join(data_dir, "comsol_params.json")
        with open(params_path) as f:
            params = json.load(f)

        data_path = os.path.join(data_dir, params["velocity_file"])
        L_comsol = params["L"]
        R = params["R"]
        E0 = params["E0"]
        eps_0 = params["eps_0"]
        eps_r = params["eps_r"]
        Fz_comsol = 4 * np.pi * R**2 * eps_0 * eps_r * E0**2
        eta_comsol = params["eta"]
        grid_resolution = params.get("grid_resolution", 48)

        return cls(
            data_path=data_path,
            L_comsol=L_comsol,
            Fz_comsol=Fz_comsol,
            eta_comsol=eta_comsol,
            grid_resolution=grid_resolution,
        )

    def __init__(
        self,
        data_path: str,
        L_comsol: float,
        Fz_comsol: float,
        eta_comsol: float = 0.065,
        grid_resolution: int = 48,
    ):
        """
        :param data_path: путь к U_lattice.txt
        :param L_comsol: размер периодической ячейки COMSOL (м)
        :param Fz_comsol: сила на каплю в расчёте COMSOL (Н)
        :param eta_comsol: вязкость внешней жидкости в расчёте COMSOL (Па·с)
        :param grid_resolution: разрешение регулярной сетки (N³)
        """
        self.L_comsol = L_comsol
        self.Fz_comsol = Fz_comsol
        self.eta_comsol = eta_comsol
        self.grid_resolution = grid_resolution

        # Загрузка и обработка
        points, u, v, w = self._load_and_parse(data_path)
        full_points, full_u, full_v, full_w = self._mirror_to_full_cell(points, u, v, w)

        # Пересэмплирование на регулярную сетку
        self.grid_coords, self.grid_u, self.grid_v, self.grid_w = self._build_regular_grid(
            full_points, full_u, full_v, full_w, grid_resolution
        )

        # Интерполяторы для scipy-оценки (тестирование)
        self._interp_u, self._interp_v, self._interp_w = self._build_interpolators()

        print(
            f"[COMSOLLatticeCorrection] Загружено {len(points)} точек, "
            f"зеркалировано до {len(full_points)}, "
            f"пересэмплировано на сетку {grid_resolution}³"
        )

    def _load_and_parse(self, data_path: str):
        """Загрузка данных из U_lattice.txt."""
        data = np.loadtxt(data_path, comments="%")
        points = data[:, 0:3]  # x, y, z
        u = data[:, 3]  # u2
        v = data[:, 4]  # v2
        w = data[:, 5]  # w2

        # Проверка: четверть-ячейка
        assert np.all(points[:, 0] >= -1e-15), "x должен быть >= 0 (четверть-ячейка)"
        assert np.all(points[:, 1] >= -1e-15), "y должен быть >= 0 (четверть-ячейка)"

        return points, u, v, w

    def _mirror_to_full_cell(self, points, u, v, w):
        """
        Зеркалирование четверть-ячейки [0, L/2]² × [-L/2, L/2]
        в полную ячейку [-L/2, L/2]³.

        Свойства симметрии стоклета в z-направлении:
        - u(x,y,z): нечётная по x, чётная по y
        - v(x,y,z): чётная по x, нечётная по y
        - w(x,y,z): чётная по x, чётная по y
        """
        eps = 1e-15  # порог для исключения точек на границе

        all_points = [points]
        all_u = [u]
        all_v = [v]
        all_w = [w]

        # Маски для точек НЕ на границе x=0 и y=0
        mask_x_pos = points[:, 0] > eps  # x > 0 (исключаем x=0)
        mask_y_pos = points[:, 1] > eps  # y > 0 (исключаем y=0)

        # Отражение x → -x (только точки с x > 0)
        if np.any(mask_x_pos):
            pts_mx = points[mask_x_pos].copy()
            pts_mx[:, 0] = -pts_mx[:, 0]
            all_points.append(pts_mx)
            all_u.append(-u[mask_x_pos])  # u нечётная по x
            all_v.append(v[mask_x_pos])  # v чётная по x
            all_w.append(w[mask_x_pos])  # w чётная по x

        # Отражение y → -y (только точки с y > 0)
        if np.any(mask_y_pos):
            pts_my = points[mask_y_pos].copy()
            pts_my[:, 1] = -pts_my[:, 1]
            all_points.append(pts_my)
            all_u.append(u[mask_y_pos])  # u чётная по y
            all_v.append(-v[mask_y_pos])  # v нечётная по y
            all_w.append(w[mask_y_pos])  # w чётная по y

        # Отражение x → -x И y → -y (только точки с x > 0 И y > 0)
        mask_both = mask_x_pos & mask_y_pos
        if np.any(mask_both):
            pts_mxy = points[mask_both].copy()
            pts_mxy[:, 0] = -pts_mxy[:, 0]
            pts_mxy[:, 1] = -pts_mxy[:, 1]
            all_points.append(pts_mxy)
            all_u.append(-u[mask_both])  # u нечётная по x, чётная по y → -u
            all_v.append(-v[mask_both])  # v чётная по x, нечётная по y → -v
            all_w.append(w[mask_both])  # w чётная по обоим

        full_points = np.concatenate(all_points, axis=0)
        full_u = np.concatenate(all_u)
        full_v = np.concatenate(all_v)
        full_w = np.concatenate(all_w)

        return full_points, full_u, full_v, full_w

    def _build_regular_grid(self, points, u, v, w, resolution):
        """
        Пересэмплирование неструктурированных данных на регулярную сетку.
        Использует LinearNDInterpolator с NearestNDInterpolator как fallback.
        """
        half_L = self.L_comsol / 2.0
        grid_1d = np.linspace(-half_L, half_L, resolution)

        print(f"[COMSOLLatticeCorrection] Построение интерполятора на {len(points)} точках...")

        # Построение интерполяторов (одноразовая операция)
        interp_linear_u = LinearNDInterpolator(points, u)
        interp_linear_v = LinearNDInterpolator(points, v)
        interp_linear_w = LinearNDInterpolator(points, w)

        interp_nearest_u = NearestNDInterpolator(points, u)
        interp_nearest_v = NearestNDInterpolator(points, v)
        interp_nearest_w = NearestNDInterpolator(points, w)

        # Создание регулярной сетки
        gx, gy, gz = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
        query_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        print(
            f"[COMSOLLatticeCorrection] Интерполяция на сетку {resolution}³ ({len(query_pts)} точек)..."
        )

        # Интерполяция
        grid_u = interp_linear_u(query_pts).reshape(resolution, resolution, resolution)
        grid_v = interp_linear_v(query_pts).reshape(resolution, resolution, resolution)
        grid_w = interp_linear_w(query_pts).reshape(resolution, resolution, resolution)

        # Заполнение NaN через nearest-neighbor
        for grid, interp_nn in [
            (grid_u, interp_nearest_u),
            (grid_v, interp_nearest_v),
            (grid_w, interp_nearest_w),
        ]:
            nan_mask = np.isnan(grid)
            if np.any(nan_mask):
                nan_count = np.sum(nan_mask)
                nan_indices = np.argwhere(nan_mask)
                nan_coords = np.column_stack(
                    [
                        grid_1d[nan_indices[:, 0]],
                        grid_1d[nan_indices[:, 1]],
                        grid_1d[nan_indices[:, 2]],
                    ]
                )
                grid[nan_mask] = interp_nn(nan_coords)
                print(f"[COMSOLLatticeCorrection] Заполнено {nan_count} NaN через nearest-neighbor")

        return grid_1d, grid_u, grid_v, grid_w

    def _build_interpolators(self):
        """Создание RegularGridInterpolator для scipy-оценки."""
        coords = self.grid_coords
        interp_u = RegularGridInterpolator(
            (coords, coords, coords),
            self.grid_u,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        interp_v = RegularGridInterpolator(
            (coords, coords, coords),
            self.grid_v,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        interp_w = RegularGridInterpolator(
            (coords, coords, coords),
            self.grid_w,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        return interp_u, interp_v, interp_w

    def evaluate(self, r_rel: np.ndarray, F_j: np.ndarray = None) -> np.ndarray:
        """
        Интерполяция поправочного поля в произвольных точках (в координатах COMSOL).

        Если F_j задан, вычисляется полная тензорная свёртка для произвольного
        направления силы с использованием кубической симметрии решётки.

        :param r_rel: массив (M, 3) относительных позиций в координатах COMSOL
        :param F_j: массив (M, 3) сил на каплю (если None — возвращает сырой столбец β=z)
        :return: массив (M, 3) скоростей поправки
        """
        if F_j is None:
            u_vals = self._interp_u(r_rel)
            v_vals = self._interp_v(r_rel)
            w_vals = self._interp_w(r_rel)
            return np.column_stack([u_vals, v_vals, w_vals])

        Fz_inv = 1.0 / self.Fz_comsol

        # Три набора переставленных координат
        pts_z = r_rel  # (x, y, z) — для G_z
        pts_x = r_rel[:, [2, 1, 0]]  # (z, y, x) — для G_x
        pts_y = r_rel[:, [0, 2, 1]]  # (x, z, y) — для G_y

        # G_z(x,y,z) = (u2, v2, w2)
        gz_x = self._interp_u(pts_z)
        gz_y = self._interp_v(pts_z)
        gz_z = self._interp_w(pts_z)

        # G_x(x,y,z) = (w2(z,y,x), v2(z,y,x), u2(z,y,x))
        gx_x = self._interp_w(pts_x)
        gx_y = self._interp_v(pts_x)
        gx_z = self._interp_u(pts_x)

        # G_y(x,y,z) = (u2(x,z,y), w2(x,z,y), v2(x,z,y))
        gy_x = self._interp_u(pts_y)
        gy_y = self._interp_w(pts_y)
        gy_z = self._interp_v(pts_y)

        # Весовые коэффициенты: F_α / Fz_comsol
        sx = F_j[:, 0] * Fz_inv
        sy = F_j[:, 1] * Fz_inv
        sz = F_j[:, 2] * Fz_inv

        # Тензорная свёртка: U = sx*G_x + sy*G_y + sz*G_z
        result_x = gx_x * sx + gy_x * sy + gz_x * sz
        result_y = gx_y * sx + gy_y * sy + gz_y * sz
        result_z = gx_z * sx + gy_z * sy + gz_z * sz

        return np.column_stack([result_x, result_y, result_z])

    def get_grid_data(self) -> dict:
        """
        Возвращает данные регулярной сетки для загрузки в Taichi.

        :return: dict с ключами:
            - grid_u, grid_v, grid_w: numpy arrays shape (N, N, N)
            - grid_min: float (нижняя граница сетки = -L/2)
            - grid_max: float (верхняя граница сетки = L/2)
            - grid_dx: float (шаг сетки)
            - grid_resolution: int
            - L_comsol: float
            - Fz_comsol: float
            - eta_comsol: float
        """
        grid_min = self.grid_coords[0]
        grid_max = self.grid_coords[-1]
        grid_dx = self.grid_coords[1] - self.grid_coords[0]

        return {
            "grid_u": self.grid_u.copy(),
            "grid_v": self.grid_v.copy(),
            "grid_w": self.grid_w.copy(),
            "grid_min": grid_min,
            "grid_max": grid_max,
            "grid_dx": grid_dx,
            "grid_resolution": self.grid_resolution,
            "L_comsol": self.L_comsol,
            "Fz_comsol": self.Fz_comsol,
            "eta_comsol": self.eta_comsol,
        }
