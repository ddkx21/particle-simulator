import numpy as np
import taichi as ti
from .force_calculator_base import ForceCalculator

@ti.data_oriented
class DirectDropletForceCalculator(ForceCalculator):

    def __init__(self, num_particles = 10000, eps_oil=2.85, eta_oil=0.065, eta_water=0.001, rho_water=1000, rho_oil=910, E=3e5, L=1.0, boundary_mode="periodic", correction_grid_resolution=0):
        self.eps0 = 8.85418781762039e-12  # Электрическая постоянная
        self.eps_oil = eps_oil
        self.eta_oil = eta_oil
        self.eta_water = eta_water
        self.rho_water = rho_water
        self.rho_oil = rho_oil
        self.E = E
        self.m_const = 12 * np.pi * self.eps0 * eps_oil * E**2
        self.eta_const = 1/(8 * np.pi * self.eta_oil)

        self.L = L
        self.boundary_mode = boundary_mode  # "periodic" или "open"
        self.num_particles = num_particles

        # Создаем поля Taichi для позиций, радиусов и сил
        self.ti_num_particles = ti.field(dtype=ti.i32, shape=())
        self.ti_radii = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_x = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_y = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_z = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_vx = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_vy = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_vz = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_fx = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_fy = ti.field(dtype=ti.f64, shape=num_particles)
        self.ti_fz = ti.field(dtype=ti.f64, shape=num_particles)

        # Periodic boundary fields
        self.boundary_mode_int = ti.field(dtype=ti.i32, shape=())
        self.L_val = ti.field(dtype=ti.f64, shape=())
        self.boundary_mode_int[None] = 1 if boundary_mode == "periodic" else 0
        self.L_val[None] = L

        # Периодическая поправка из COMSOL (опционально)
        # Taichi JIT компилирует все ветки ядра — поля должны существовать всегда
        self.correction_grid_resolution = correction_grid_resolution
        N = max(correction_grid_resolution, 2)  # минимум 2³ для корректной компиляции
        self.corr_grid_u = ti.field(dtype=ti.f64, shape=(N, N, N))
        self.corr_grid_v = ti.field(dtype=ti.f64, shape=(N, N, N))
        self.corr_grid_w = ti.field(dtype=ti.f64, shape=(N, N, N))
        self.corr_grid_min = ti.field(dtype=ti.f64, shape=())
        self.corr_grid_inv_dx = ti.field(dtype=ti.f64, shape=())
        self.corr_grid_n = ti.field(dtype=ti.i32, shape=())
        self.corr_Fz_inv = ti.field(dtype=ti.f64, shape=())
        self.corr_L_ratio = ti.field(dtype=ti.f64, shape=())
        self.corr_eta_ratio = ti.field(dtype=ti.f64, shape=())
        self.corr_enabled = ti.field(dtype=ti.i32, shape=())
        self.corr_enabled[None] = 0


    def load_periodic_correction(self, correction, L_sim: float):
        """
        Загрузка данных периодической поправки COMSOL в Taichi-поля.

        :param correction: объект COMSOLLatticeCorrection
        :param L_sim: размер ячейки моделирования (м)
        """
        if self.correction_grid_resolution == 0:
            raise RuntimeError(
                "correction_grid_resolution=0: Taichi-поля для поправки не выделены. "
                "Пересоздайте DirectDropletForceCalculator с correction_grid_resolution > 0."
            )

        grid_data = correction.get_grid_data()

        # Копируем 3D массивы в Taichi-поля
        self.corr_grid_u.from_numpy(grid_data['grid_u'])
        self.corr_grid_v.from_numpy(grid_data['grid_v'])
        self.corr_grid_w.from_numpy(grid_data['grid_w'])

        # Метаданные сетки (сетка одинаковая по всем осям)
        self.corr_grid_min[None] = grid_data['grid_min']
        self.corr_grid_inv_dx[None] = 1.0 / grid_data['grid_dx']
        self.corr_grid_n[None] = grid_data['grid_resolution']
        self.corr_Fz_inv[None] = 1.0 / grid_data['Fz_comsol']
        self.corr_L_ratio[None] = grid_data['L_comsol'] / L_sim
        self.corr_eta_ratio[None] = grid_data['eta_comsol'] / self.eta_oil

        # Включить поправку
        self.corr_enabled[None] = 1

        eta_ratio = grid_data['eta_comsol'] / self.eta_oil
        print(f"[DirectDropletForceCalculator] Периодическая поправка загружена: "
              f"сетка {grid_data['grid_resolution']}³, "
              f"L_ratio={grid_data['L_comsol']/L_sim:.6f}, "
              f"Fz_comsol={grid_data['Fz_comsol']:.4e}, "
              f"eta_ratio={eta_ratio:.6f}")


    @staticmethod
    @ti.func
    def _trilinear_interp(grid: ti.template(), x: ti.f64, y: ti.f64, z: ti.f64,
                          grid_min: ti.f64, inv_dx: ti.f64, n: ti.i32) -> ti.f64:
        """Трилинейная интерполяция на регулярной 3D сетке с clamping."""
        ix, iy, iz, tx, ty, tz = DirectDropletForceCalculator._precompute_grid_idx(x, y, z, grid_min, inv_dx, n)
        return DirectDropletForceCalculator._interp_precomp(grid, ix, iy, iz, tx, ty, tz)

    @staticmethod
    @ti.func
    def _precompute_grid_idx(x: ti.f64, y: ti.f64, z: ti.f64,
                             grid_min: ti.f64, inv_dx: ti.f64, n: ti.i32):
        fx = (x - grid_min) * inv_dx
        fy = (y - grid_min) * inv_dx
        fz = (z - grid_min) * inv_dx
        n_max = ti.cast(n - 1, ti.f64)
        fx = ti.max(0.0, ti.min(fx, n_max))
        fy = ti.max(0.0, ti.min(fy, n_max))
        fz = ti.max(0.0, ti.min(fz, n_max))
        ix = ti.min(ti.cast(ti.floor(fx), ti.i32), n - 2)
        iy = ti.min(ti.cast(ti.floor(fy), ti.i32), n - 2)
        iz = ti.min(ti.cast(ti.floor(fz), ti.i32), n - 2)
        tx = fx - ti.cast(ix, ti.f64)
        ty = fy - ti.cast(iy, ti.f64)
        tz = fz - ti.cast(iz, ti.f64)
        return ix, iy, iz, tx, ty, tz

    @staticmethod
    @ti.func
    def _interp_precomp(grid: ti.template(),
                        ix: ti.i32, iy: ti.i32, iz: ti.i32,
                        tx: ti.f64, ty: ti.f64, tz: ti.f64) -> ti.f64:
        c000 = grid[ix, iy, iz]
        c001 = grid[ix, iy, iz + 1]
        c010 = grid[ix, iy + 1, iz]
        c011 = grid[ix, iy + 1, iz + 1]
        c100 = grid[ix + 1, iy, iz]
        c101 = grid[ix + 1, iy, iz + 1]
        c110 = grid[ix + 1, iy + 1, iz]
        c111 = grid[ix + 1, iy + 1, iz + 1]
        return (c000 * (1 - tx) * (1 - ty) * (1 - tz) +
                c001 * (1 - tx) * (1 - ty) * tz +
                c010 * (1 - tx) * ty * (1 - tz) +
                c011 * (1 - tx) * ty * tz +
                c100 * tx * (1 - ty) * (1 - tz) +
                c101 * tx * (1 - ty) * tz +
                c110 * tx * ty * (1 - tz) +
                c111 * tx * ty * tz)


    @staticmethod
    @ti.func
    def _calculate_one_force(m_const, dx, dy, dz, r_mag, radii_i, radii_j):

        force_on_i = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

        M_ik_per_r_mag7 = m_const * radii_i**3 * radii_j**3 / r_mag**7
 
        dx_squared, dy_squared, dz_squared = dx**2, dy**2, dz**2

        force_xy_per_delta = M_ik_per_r_mag7 * (4 * dz_squared - dx_squared - dy_squared)
        force_on_i[0] = force_xy_per_delta * dx  # Компонента по x
        force_on_i[1] = force_xy_per_delta * dy  # Компонента по y
        force_on_i[2] = M_ik_per_r_mag7 * (2 * dz_squared - 3 * dx_squared - 3 * dy_squared) * dz  # Компонента по z

        return force_on_i


    @ti.kernel
    def _calculate_forces(self,
                          ti_num_particles: ti.template(),
                          ti_x: ti.template(),
                          ti_y: ti.template(),
                          ti_z: ti.template(),
                          radii: ti.template(),
                          ti_fx: ti.template(),
                          ti_fy: ti.template(),
                          ti_fz: ti.template()):

        num_particles = ti_num_particles[None]
        is_periodic = self.boundary_mode_int[None]
        L_v = self.L_val[None]

        force_on_i = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

        for i in range(num_particles):

            force_on_i.fill(0)

            for j in range(num_particles):

                if i != j:

                    dx = ti_x[j] - ti_x[i]
                    dy = ti_y[j] - ti_y[i]
                    dz = ti_z[j] - ti_z[i]

                    # Minimum image convention для periodic boundary
                    if is_periodic == 1:
                        if ti.abs(dx) > L_v * 0.5:
                            dx -= ti.math.sign(dx) * L_v
                        if ti.abs(dy) > L_v * 0.5:
                            dy -= ti.math.sign(dy) * L_v
                        if ti.abs(dz) > L_v * 0.5:
                            dz -= ti.math.sign(dz) * L_v

                    r_mag = ti.sqrt(dx**2 + dy**2 + dz**2)

                    r_sum = radii[i] + radii[j]

                    if r_mag > r_sum:
                        force_on_i += self._calculate_one_force(self.m_const, dx, dy, dz, r_mag, radii[i], radii[j])

            ti_fx[i] = force_on_i[0]
            ti_fy[i] = force_on_i[1]
            ti_fz[i] = force_on_i[2]
    

    @ti.kernel
    def update_values(self, ti_radii: ti.template(), radii: ti.types.ndarray(), size: ti.i32):
        for i in range(size):
            ti_radii[i] = radii[i]

    def calculate(self, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        self.num_particles = positions.shape[0]
        self.ti_num_particles[None] = self.num_particles

        # Преобразование типов numpy массивов в float64
        positions = positions.astype(np.float64)
        radii = radii.astype(np.float64)

        # Копируем numpy данные в Taichi поля
        self.update_values(self.ti_radii, np.ascontiguousarray(radii), self.num_particles)
        self.update_values(self.ti_x, np.ascontiguousarray(positions[:, 0]), self.num_particles)
        self.update_values(self.ti_y, np.ascontiguousarray(positions[:, 1]), self.num_particles)
        self.update_values(self.ti_z, np.ascontiguousarray(positions[:, 2]), self.num_particles)
        
        # Вызываем расчет сил
        self._calculate_forces(self.ti_num_particles, self.ti_x, self.ti_y, self.ti_z, self.ti_radii, self.ti_fx, self.ti_fy, self.ti_fz)

        # Копируем силы обратно в numpy
        fx = self.ti_fx.to_numpy()
        fy = self.ti_fy.to_numpy()
        fz = self.ti_fz.to_numpy()

        forces = np.stack((fx, fy, fz), axis=1)

        # Оставляем только self.num_particles значений силы (хвосты лишние)
        forces = forces[:self.num_particles]

        # Очищаем память
        self.ti_x.fill(0)
        self.ti_y.fill(0)
        self.ti_z.fill(0)
        self.ti_fx.fill(0)
        self.ti_fy.fill(0)
        self.ti_fz.fill(0)
        self.ti_radii.fill(0)

        return forces



    def calculate_convection(self, positions: np.ndarray, radii: np.ndarray, forces: np.ndarray) -> np.ndarray:
        
        self.num_particles = positions.shape[0]
        self.ti_num_particles[None] = self.num_particles

        # Преобразование типов numpy массивов в float64
        positions = positions.astype(np.float64)
        radii = radii.astype(np.float64)
        forces = forces.astype(np.float64)

        # Копируем numpy данные в Taichi поля
        self.update_values(self.ti_radii, np.ascontiguousarray(radii), self.num_particles)
        self.update_values(self.ti_x, np.ascontiguousarray(positions[:, 0]), self.num_particles)
        self.update_values(self.ti_y, np.ascontiguousarray(positions[:, 1]), self.num_particles)
        self.update_values(self.ti_z, np.ascontiguousarray(positions[:, 2]), self.num_particles)
        self.update_values(self.ti_fx, np.ascontiguousarray(forces[:, 0]), self.num_particles)
        self.update_values(self.ti_fy, np.ascontiguousarray(forces[:, 1]), self.num_particles)
        self.update_values(self.ti_fz, np.ascontiguousarray(forces[:, 2]), self.num_particles)

        # Вызываем расчет скоростей
        self._calculate_convection_velocities(self.ti_num_particles, self.ti_x, self.ti_y, self.ti_z, self.ti_fx, self.ti_fy, self.ti_fz, self.ti_vx, self.ti_vy, self.ti_vz)

        # Копируем скорости обратно в numpy
        vx = self.ti_vx.to_numpy()
        vy = self.ti_vy.to_numpy()
        vz = self.ti_vz.to_numpy()

        velocities = np.stack((vx, vy, vz), axis=1)

        # Оставляем только self.num_particles значений скоростей (хвосты лишние)
        velocities = velocities[:self.num_particles]

        return velocities


    @ti.kernel
    def _calculate_convection_velocities(self, 
                              ti_num_particles: ti.template(), 
                              ti_x: ti.template(), 
                              ti_y: ti.template(), 
                              ti_z: ti.template(), 
                              ti_fx: ti.template(),
                              ti_fy: ti.template(),
                              ti_fz: ti.template(),
                              ti_vx: ti.template(), 
                              ti_vy: ti.template(), 
                              ti_vz: ti.template()):
        
        num_particles = ti_num_particles[None]
        is_periodic = self.boundary_mode_int[None]
        L_v = self.L_val[None]

        convection_at_i = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

        for i in range(num_particles):
            convection_at_i.fill(0)

            for j in range(num_particles):

                if i != j:

                    x = ti_x[i] - ti_x[j]
                    y = ti_y[i] - ti_y[j]
                    z = ti_z[i] - ti_z[j]

                    # Minimum image convention для periodic boundary
                    if is_periodic == 1:
                        if ti.abs(x) > L_v * 0.5:
                            x -= ti.math.sign(x) * L_v
                        if ti.abs(y) > L_v * 0.5:
                            y -= ti.math.sign(y) * L_v
                        if ti.abs(z) > L_v * 0.5:
                            z -= ti.math.sign(z) * L_v

                    r = ti.sqrt(x**2 + y**2 + z**2)

                    fx = ti_fx[j]
                    fy = ti_fy[j]
                    fz = ti_fz[j]

                    V_x = self.eta_const * ((1/r + (x**2)/r**3) * fx + (x * y)/r**3 * fy + (x * z)/r**3 * fz)
                    V_y = self.eta_const * ((x * y)/r**3 * fx + (1/r + (y**2)/r**3) * fy + (y * z)/r**3 * fz)
                    V_z = self.eta_const * ((x * z)/r**3 * fx + (y * z)/r**3 * fy + (1/r + (z**2)/r**3) * fz)

                    convection_at_i += ti.Vector([V_x, V_y, V_z])

                    # Поправка от периодических образов (COMSOL) — полный тензор
                    if self.corr_enabled[None] == 1:
                        cL = self.corr_L_ratio[None]
                        xc = x * cL
                        yc = y * cL
                        zc = z * cL

                        gmin = self.corr_grid_min[None]
                        ginv = self.corr_grid_inv_dx[None]
                        gn = self.corr_grid_n[None]
                        Fz_inv = self.corr_Fz_inv[None]
                        eta_r = self.corr_eta_ratio[None]

                        # G_z: (xc, yc, zc) — предвычисление индексов один раз
                        i1x, i1y, i1z, t1x, t1y, t1z = self._precompute_grid_idx(xc, yc, zc, gmin, ginv, gn)
                        gz_x = self._interp_precomp(self.corr_grid_u, i1x, i1y, i1z, t1x, t1y, t1z)
                        gz_y = self._interp_precomp(self.corr_grid_v, i1x, i1y, i1z, t1x, t1y, t1z)
                        gz_z = self._interp_precomp(self.corr_grid_w, i1x, i1y, i1z, t1x, t1y, t1z)

                        # G_x: перестановка x↔z → (zc, yc, xc)
                        i2x, i2y, i2z, t2x, t2y, t2z = self._precompute_grid_idx(zc, yc, xc, gmin, ginv, gn)
                        gx_x = self._interp_precomp(self.corr_grid_w, i2x, i2y, i2z, t2x, t2y, t2z)
                        gx_y = self._interp_precomp(self.corr_grid_v, i2x, i2y, i2z, t2x, t2y, t2z)
                        gx_z = self._interp_precomp(self.corr_grid_u, i2x, i2y, i2z, t2x, t2y, t2z)

                        # G_y: перестановка y↔z → (xc, zc, yc)
                        i3x, i3y, i3z, t3x, t3y, t3z = self._precompute_grid_idx(xc, zc, yc, gmin, ginv, gn)
                        gy_x = self._interp_precomp(self.corr_grid_u, i3x, i3y, i3z, t3x, t3y, t3z)
                        gy_y = self._interp_precomp(self.corr_grid_w, i3x, i3y, i3z, t3x, t3y, t3z)
                        gy_z = self._interp_precomp(self.corr_grid_v, i3x, i3y, i3z, t3x, t3y, t3z)

                        common = Fz_inv * eta_r * cL
                        sx = fx * common
                        sy = fy * common
                        sz = fz * common

                        convection_at_i[0] += gx_x * sx + gy_x * sy + gz_x * sz
                        convection_at_i[1] += gx_y * sx + gy_y * sy + gz_y * sz
                        convection_at_i[2] += gx_z * sx + gy_z * sy + gz_z * sz

            ti_vx[i] = convection_at_i[0]
            ti_vy[i] = convection_at_i[1]
            ti_vz[i] = convection_at_i[2]


    def calculate_forces_and_convection(self, positions: np.ndarray, radii: np.ndarray) -> tuple:
        self.num_particles = positions.shape[0]
        self.ti_num_particles[None] = self.num_particles

        positions = positions.astype(np.float64)
        radii = radii.astype(np.float64)

        self.update_values(self.ti_radii, np.ascontiguousarray(radii), self.num_particles)
        self.update_values(self.ti_x, np.ascontiguousarray(positions[:, 0]), self.num_particles)
        self.update_values(self.ti_y, np.ascontiguousarray(positions[:, 1]), self.num_particles)
        self.update_values(self.ti_z, np.ascontiguousarray(positions[:, 2]), self.num_particles)

        self._calculate_forces(
            self.ti_num_particles, self.ti_x, self.ti_y, self.ti_z,
            self.ti_radii, self.ti_fx, self.ti_fy, self.ti_fz)

        self._calculate_convection_velocities(
            self.ti_num_particles, self.ti_x, self.ti_y, self.ti_z,
            self.ti_fx, self.ti_fy, self.ti_fz,
            self.ti_vx, self.ti_vy, self.ti_vz)

        n = self.num_particles
        forces = np.stack((self.ti_fx.to_numpy()[:n],
                           self.ti_fy.to_numpy()[:n],
                           self.ti_fz.to_numpy()[:n]), axis=1)
        velocities = np.stack((self.ti_vx.to_numpy()[:n],
                               self.ti_vy.to_numpy()[:n],
                               self.ti_vz.to_numpy()[:n]), axis=1)

        self.ti_x.fill(0)
        self.ti_y.fill(0)
        self.ti_z.fill(0)
        self.ti_fx.fill(0)
        self.ti_fy.fill(0)
        self.ti_fz.fill(0)
        self.ti_vx.fill(0)
        self.ti_vy.fill(0)
        self.ti_vz.fill(0)
        self.ti_radii.fill(0)

        return forces, velocities