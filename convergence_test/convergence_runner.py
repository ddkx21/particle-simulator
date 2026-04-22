"""
Основной цикл теста сходимости: для каждого N и K вычисляет
REFERENCE(K) и PSEUDO, сравнивает, собирает результаты в DataFrame.
"""
import time
import numpy as np
import pandas as pd
import taichi as ti

from .reference_calculator import compute_reference


def _generate_particles(N: int, L: float, seed: int):
    """Генерация случайных частиц в [0.1·L, 0.9·L]³ с радиусами ~0.01·L."""
    rng = np.random.default_rng(seed)
    margin = 0.1 * L
    positions = rng.uniform(margin, L - margin, size=(N, 3))
    radii = rng.uniform(0.008 * L, 0.012 * L, size=N)
    return positions, radii


def _rmse_relative(ref: np.ndarray, test: np.ndarray) -> float:
    """RMSE(ref - test) / RMS(ref)."""
    diff = ref - test
    rmse = np.sqrt(np.mean(diff**2))
    rms_ref = np.sqrt(np.mean(ref**2))
    if rms_ref < 1e-30:
        return 0.0
    return rmse / rms_ref


def _compute_pseudo(positions: np.ndarray, radii: np.ndarray,
                    L: float, eps_oil: float, eta_oil: float, E: float,
                    correction, method: str = "direct"):
    """
    Вычисляет силы и скорости нашим псевдо-периодическим методом
    (MIC + COMSOL correction).

    :param method: "direct" или "tree"
    :return: (forces, velocities, elapsed_time)
    """
    N = positions.shape[0]

    if method == "direct":
        from force_calculator.direct_droplet_force_calculator import DirectDropletForceCalculator
        calc = DirectDropletForceCalculator(
            num_particles=N,
            eps_oil=eps_oil, eta_oil=eta_oil, E=E,
            L=L, boundary_mode="periodic",
            correction_grid_resolution=correction.grid_resolution,
        )
        calc.load_periodic_correction(correction, L_sim=L)
    else:
        from octree.force_tree import TreeDropletForceCalculator
        calc = TreeDropletForceCalculator(
            num_particles=N,
            theta=0.5, mpl=1,
            eps_oil=eps_oil, eta_oil=eta_oil, E=E,
            L=L, periodic=True,
            correction_grid_resolution=correction.grid_resolution,
        )
        calc.load_periodic_correction(correction, L_sim=L)

    t0 = time.perf_counter()
    forces = calc.calculate(positions, radii)
    velocities = calc.calculate_convection(positions, radii, forces)
    elapsed = time.perf_counter() - t0

    return forces, velocities, elapsed


class ConvergenceTest:
    """Тест сходимости псевдо-периодического метода."""

    def __init__(self, eps_oil: float = 2.85, eta_oil: float = 0.065, E: float = 3e5,
                 L: float = 1e-3):
        self.eps_oil = eps_oil
        self.eta_oil = eta_oil
        self.E = E
        self.L = L

        eps0 = 8.85418781762039e-12
        self.m_const = 12.0 * np.pi * eps0 * eps_oil * E**2
        self.eta_const = 1.0 / (8.0 * np.pi * eta_oil)

    def run(self, N_list: list[int], K_list: list[int],
            results_dir: str = "convergence_test/results") -> pd.DataFrame:
        """
        Основной цикл: для каждого N и K вычисляет ошибки.

        :return: DataFrame со столбцами:
            N, K, err_F_direct, err_V_direct, time_ref, time_pseudo_direct
        """
        import os
        from periodic_correction import COMSOLLatticeCorrection

        os.makedirs(results_dir, exist_ok=True)
        correction = COMSOLLatticeCorrection.load_default()

        rows = []

        for N in N_list:
            seed = 42 + N
            positions, radii = _generate_particles(N, self.L, seed)

            # Вычисляем PSEUDO один раз (не зависит от K)
            print(f"\n{'='*60}")
            print(f"N={N}: вычисление PSEUDO (direct)...")
            F_pseudo, V_pseudo, t_pseudo = _compute_pseudo(
                positions, radii, self.L,
                self.eps_oil, self.eta_oil, self.E,
                correction, method="direct"
            )
            print(f"  PSEUDO direct: {t_pseudo:.2f} с")

            for K in K_list:
                N_total = K**3 * N
                print(f"\n  K={K} (суперячейка {K}³·{N} = {N_total} частиц)...")

                t0 = time.perf_counter()
                F_ref, V_ref = compute_reference(
                    positions, radii, self.L, K,
                    self.m_const, self.eta_const,
                    self.eps_oil, self.eta_oil, self.E,
                )
                t_ref = time.perf_counter() - t0

                err_F = _rmse_relative(F_ref, F_pseudo)
                err_V = _rmse_relative(V_ref, V_pseudo)

                print(f"    err_F = {err_F:.6e}, err_V = {err_V:.6e}, "
                      f"time_ref = {t_ref:.2f} с")

                rows.append({
                    'N': N,
                    'K': K,
                    'err_F_direct': err_F,
                    'err_V_direct': err_V,
                    'time_ref': t_ref,
                    'time_pseudo_direct': t_pseudo,
                })

                # Сохраняем промежуточные результаты
                df = pd.DataFrame(rows)
                df.to_csv(os.path.join(results_dir, "convergence_results.csv"), index=False)

            # Сохраняем данные для scatter-графика (наибольший K)
            K_max = max(K_list)
            np.save(os.path.join(results_dir, f"F_ref_N{N}_K{K_max}.npy"), F_ref)
            np.save(os.path.join(results_dir, f"F_pseudo_N{N}.npy"), F_pseudo)
            np.save(os.path.join(results_dir, f"V_ref_N{N}_K{K_max}.npy"), V_ref)
            np.save(os.path.join(results_dir, f"V_pseudo_N{N}.npy"), V_pseudo)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(results_dir, "convergence_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nРезультаты сохранены в {csv_path}")
        return df


if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    test = ConvergenceTest(
        eps_oil=2.85, eta_oil=0.065, E=3e5, L=1e-3,
    )

    # Полный тест
    N_list = [1000, 5000, 10000, 20000, 50000]
    K_list = [1, 3, 5, 7, 9]

    df = test.run(N_list, K_list)
    print("\n" + df.to_string(index=False))
