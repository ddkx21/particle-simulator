"""
Бенчмарк: DirectDropletForceCalculator vs TreeDropletForceCalculator.

Измеряет время и относительную погрешность скоростей (миграция + конвекция)
при различных N, theta, mpl. Результаты сохраняются в results/.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import taichi as ti
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ti.init(arch=ti.cpu, default_fp=ti.f64)  # Параллельный расчёт на CPU

from dem.octree.force_tree import TreeDropletForceCalculator
from dem.force_calculator import DirectDropletForceCalculator
from dem.particle_generator import UniformDropletGenerator
from dem.particle_state import DropletState

# ── Параметры ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RADII_RANGE          = np.array([2.5e-6, 7.5e-6])
WATER_VOLUME_CONTENT = 0.02
REPS                 = 5
PHYSICS              = dict(eps_oil=2.85, eta_oil=0.065, eta_water=0.001, E=3e5)
THETA_LIST           = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5 , 0.6, 0.7, 0.8, 0.9, 1.0]
MPL_LIST             = [1, 2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 32]
N_LIST               = [100_000, 250_000, 500_000, 1_000_000]
USE_SAVED_STATE      = False

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Утилиты ──────────────────────────────────────────────────────────────────

def sync_time(fn, reps: int) -> tuple:
    """
    Запустить fn() reps раз с синхронизацией Taichi.
    Возвращает (среднее время в секундах, результат последнего вызова).
    """
    ti.sync()
    t0 = time.perf_counter()
    result = None
    for _ in range(reps):
        result = fn()
    ti.sync()
    return (time.perf_counter() - t0) / reps, result


def hadamard_factor(eta_oil: float, eta_water: float) -> float:
    """Знаменатель формулы Адамара–Рыбчинского: 2π·η·(2η+3η_w)/(η+η_w)."""
    return (2 * np.pi * eta_oil
            * (2 * eta_oil + 3 * eta_water)
            / (eta_oil + eta_water))


def relative_velocity_error(
    v_ref: np.ndarray,
    v_approx: np.ndarray,
    noise_frac: float = 0.01,
) -> tuple:
    """
    Относительная погрешность скоростей с шумовым полом.
    C = noise_frac * mean(|v_ref|) предотвращает деление на ~0.
    Возвращает (mean_err_pct, max_err_pct).
    """
    v_ref_norm = np.linalg.norm(v_ref, axis=1)
    C          = np.mean(v_ref_norm) * noise_frac
    diff_norm  = np.linalg.norm(v_approx - v_ref, axis=1)
    rel        = diff_norm / (v_ref_norm + C)
    return float(np.mean(rel)) * 100, float(np.max(rel)) * 100


def box_size_from_volume(n: int, radii: np.ndarray, wvc: float) -> float:
    """
    Размер куба из условия заданного объёмного содержания воды.
    Приближение: V_drops ≈ N · π/3 · (R_min+R_max) · (R_min²+R_max²).
    """
    return float(np.cbrt(
        np.pi * n * np.sum(radii) * np.sum(np.square(radii))
        / (3 * wvc)
    ))


# ── Заголовок таблицы ────────────────────────────────────────────────────────
HEADER = (f"{'N':>7} {'theta':>6} {'mpl':>4} "
          f"{'direct(ms)':>11} {'tree(ms)':>10} {'speedup':>8} "
          f"{'mean_err%':>10} {'max_err%':>10}")
print(HEADER)
print("─" * len(HEADER))

all_results: list = []

# ── Основной цикл ────────────────────────────────────────────────────────────
for N in N_LIST:
    L = box_size_from_volume(N, RADII_RANGE, WATER_VOLUME_CONTENT)

    # ── Генерация / загрузка состояния ──────────────────────────────────────
    state_path = RESULTS_DIR / f"droplets_N{N}_vol{WATER_VOLUME_CONTENT}_0.xlsx"

    if USE_SAVED_STATE and state_path.exists():
        state = DropletState(filename=str(state_path))
        print(f"  [N={N}] загружено из {state_path}")
    else:
        print(f"  [N={N}] генерация частиц…", end=" ", flush=True)
        gen = UniformDropletGenerator(
            coord_range=(0, L),
            radii_range=RADII_RANGE,
            num_particles=N,
            minimum_distance=1e-6,
        )
        state = gen.generate()
        state.export_to_xlsx(str(state_path))
        print("готово")

    pos = state.positions
    rad = state.radii

    # ── Прямой метод: создаём и греем один раз для всего N ──────────────────
    print(f"  [N={N}] прогрев DirectCalculator…", end=" ", flush=True)
    fc_d = DirectDropletForceCalculator(num_particles=N, **PHYSICS)
    _ = fc_d.calculate(pos, rad)
    _ = fc_d.calculate_convection(pos, rad, _)
    print("готово")

    # Замер прямого метода (один раз — не зависит от theta/mpl)
    def _direct_step():
        f  = fc_d.calculate(pos, rad)
        vc = fc_d.calculate_convection(pos, rad, f)
        return f, vc

    dt_d, (f_d, vc_d) = sync_time(_direct_step, REPS)

    # Адамар–Рыбчинский — один раз на каждое N
    sf    = hadamard_factor(fc_d.eta_oil, fc_d.eta_water)
    vm_d  = f_d / (sf * rad[:, np.newaxis])
    vt_d  = vc_d + vm_d   # полная скорость (эталон)

    # ── Один экземпляр дерева на весь N (Taichi не освобождает выделенную память) ──
    fc_t = TreeDropletForceCalculator(
        num_particles=N, theta=THETA_LIST[0], mpl=MPL_LIST[0], **PHYSICS,
    )
    # Прогрев (компиляция ядер — один раз)
    _wf = fc_t.calculate(pos, rad)
    _ = fc_t.calculate_convection(pos, rad, _wf)

    # ── Перебор theta × mpl ─────────────────────────────────────────────────
    for theta in THETA_LIST:
        for mpl in MPL_LIST:
            fc_t.update_params(theta, mpl)

            # Замер дерева
            def _tree_step():
                f  = fc_t.calculate(pos, rad)
                vc = fc_t.calculate_convection(pos, rad, f)
                return f, vc

            dt_t, (f_t, vc_t) = sync_time(_tree_step, REPS)

            # Погрешность (из последнего вызова замера — не пересчитываем)
            vm_t = f_t / (sf * rad[:, np.newaxis])
            vt_t = vc_t + vm_t

            mean_err, max_err = relative_velocity_error(vt_d, vt_t)
            speedup = dt_d / dt_t if dt_t > 0 else float('inf')

            row = dict(
                N=N, theta=theta, mpl=mpl,
                direct_ms=dt_d * 1e3,
                tree_ms=dt_t * 1e3,
                speedup=speedup,
                mean_rel_pct=mean_err,
                max_rel_pct=max_err,
            )
            all_results.append(row)

            # Немедленный вывод строки
            print(f"{N:7d} {theta:6.2f} {mpl:4d} "
                  f"{dt_d*1e3:11.2f} {dt_t*1e3:10.2f} "
                  f"{speedup:7.2f}x "
                  f"{mean_err:10.3f} {max_err:10.3f}")

            # Частичное сохранение (защита от потери данных при падении)
            pd.DataFrame(all_results).to_csv(
                RESULTS_DIR / "benchmark_partial.csv", index=False
            )

# ── Финальное сохранение ─────────────────────────────────────────────────────
df = pd.DataFrame(all_results)

csv_path  = RESULTS_DIR / "benchmark.csv"
xlsx_path = RESULTS_DIR / "benchmark.xlsx"

df.to_csv(csv_path, index=False)
df.to_excel(xlsx_path, index=False)

print(f"\nРезультаты сохранены:\n  {csv_path}\n  {xlsx_path}")

# ── Сводка: лучший speedup при mean_err < 5% ────────────────────────────────
good = df[df["mean_rel_pct"] < 5.0]
if len(good) > 0:
    print("\nТоп-5 конфигураций (speedup при mean_err < 5%):")
    top = (good
           .sort_values("speedup", ascending=False)
           .head(5)
           [["N", "theta", "mpl", "speedup", "mean_rel_pct", "max_rel_pct"]])
    print(top.to_string(index=False))
else:
    print("\nНет конфигураций с mean_err < 5%.")
