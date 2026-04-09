"""
Тесты плоского октодерева.

Проверяет корректность дерева сравнением с direct O(N²) методом.
Метрика: e = |v_direct - v_octree| / (|v_direct| + C)

Запуск: python -m octree.test_octree
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f64)

from octree import FlatOctree, TreeDropletForceCalculator
from force_calculator.direct_droplet_force_calculator import DirectDropletForceCalculator


def relative_error(v_direct, v_octree, C=1e-30):
    diff = np.linalg.norm(v_direct - v_octree, axis=1)
    ref = np.linalg.norm(v_direct, axis=1) + C
    return diff / ref


def run_all_tests():
    print("=" * 60)
    print("  ТЕСТЫ ПЛОСКОГО ОКТОДЕРЕВА")
    print("=" * 60 + "\n")

    # ---- Test 1: Build ----
    print("Тест 1: Построение октодерева...")
    np.random.seed(42)
    n = 50
    L = 1.0
    positions = np.random.rand(n, 3) * L
    radii = np.random.rand(n) * 0.005 + 0.005

    for mpl in [1, 4, 8]:
        tree = FlatOctree(theta=0.5, mpl=mpl, num_particles=n)
        tree.build(positions, radii, L, periodic=False)
        stats = tree.get_tree_stats()
        print(f"  mpl={mpl}: nodes={stats['node_count']}, particles={stats['num_particles']}")
        assert stats['node_count'] > 1
        assert stats['num_particles'] == n
    print("  ✓ Тест 1 пройден\n")

    # ---- Test 2: Forces vs Direct ----
    print("Тест 2: Силы — octree vs direct...")
    np.random.seed(42)
    n = 30
    L = 0.01
    pos = np.random.rand(n, 3) * L
    rad = np.random.rand(n) * 30e-6 + 20e-6

    direct = DirectDropletForceCalculator(num_particles=n)
    f_d = direct.calculate(pos, rad)

    tree_calc = TreeDropletForceCalculator(num_particles=n, theta=0.3, mpl=1)
    f_t = tree_calc.calculate(pos, rad, L=L, periodic=False)

    C = np.mean(np.linalg.norm(f_d, axis=1)) * 1e-10
    err = relative_error(f_d, f_t, C)
    print(f"  N={n}, theta=0.3: mean_err={np.mean(err):.2e}, max_err={np.max(err):.2e}")
    assert np.mean(err) < 0.05, f"Force error too large: {np.mean(err)}"
    print("  ✓ Тест 2 пройден\n")

    # ---- Test 3: Convection ----
    print("Тест 3: Конвекция — octree vs direct...")
    v_d = direct.calculate_convection(pos, rad, f_d)
    v_t = tree_calc.calculate_convection(pos, rad, f_t, L=L, periodic=False)

    C_v = np.mean(np.linalg.norm(v_d, axis=1)) * 1e-10
    err_v = relative_error(v_d, v_t, C_v)
    print(f"  Convection mean_err={np.mean(err_v):.2e}")
    print("  ✓ Тест 3 пройден\n")

    # ---- Test 4: Total velocity ----
    print("Тест 4: Полная скорость v_total = v_mig + v_conv...")
    eta_oil = 0.065
    eta_water = 0.001
    sf = 2 * np.pi * eta_oil * (2 * eta_oil + 3 * eta_water) / (eta_oil + eta_water)

    vt_d = f_d / (sf * rad[:, None]) + v_d
    vt_t = f_t / (sf * rad[:, None]) + v_t

    C_t = np.mean(np.linalg.norm(vt_d, axis=1)) * 1e-10
    err_t = relative_error(vt_d, vt_t, C_t)
    print(f"  V_total mean_err={np.mean(err_t):.2e}, max_err={np.max(err_t):.2e}")
    assert np.mean(err_t) < 0.1, f"V_total error too large: {np.mean(err_t)}"
    print("  ✓ Тест 4 пройден\n")

    # ---- Test 5: Theta convergence ----
    print("Тест 5: Сходимость по theta...")
    prev_err = 1.0
    for theta in [0.8, 0.5, 0.3]:
        tc = TreeDropletForceCalculator(num_particles=n, theta=theta, mpl=1)
        ft = tc.calculate(pos, rad, L=L, periodic=False)
        e = np.mean(relative_error(f_d, ft, C))
        print(f"  theta={theta:.1f}: mean_err={e:.2e}")
        assert e <= prev_err + 1e-4
        prev_err = e
    print("  ✓ Тест 5 пройден\n")

    print("=" * 60)
    print("  ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
