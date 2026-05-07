from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pbm.volume_grid import VolumeGrid


def fixed_pivot(
    v_new: float,
    grid: VolumeGrid,
) -> list[tuple[int, float]]:
    """Перераспределение fixed-pivot (Kumar–Ramkrishna 1996).

    Линейная интерполяция между двумя ближайшими центрами.
    Сохраняет число и объём одновременно.
    """
    x = grid.centers

    if v_new <= x[0]:
        return [(0, 1.0)]
    if v_new >= x[-1]:
        return [(grid.n_bins - 1, 1.0)]

    i = int(np.searchsorted(x, v_new)) - 1
    i = max(0, min(i, grid.n_bins - 2))

    delta = x[i + 1] - x[i]
    a = (x[i + 1] - v_new) / delta
    b = (v_new - x[i]) / delta
    return [(i, a), (i + 1, b)]


def cell_average(
    B: NDArray[np.float64],
    M: NDArray[np.float64],
    grid: VolumeGrid,
) -> NDArray[np.float64]:
    """Метод Cell-Average (Kumar 2006) — векторизованная реализация.

    Args:
        B[i]: суммарная скорость рождения, накопленная в бине i (1/с).
        M[i]: суммарный объём новорождённых в бине i, = Σ rate * v_new (м³/с).

    Returns:
        birth_contrib[i] — итоговый вклад в каждый бин.

    Внутри сетки:
        a_bar = M[i] / B[i] — средний объём новорождённого.
        Если a_bar < x[i]: распределяем между (i-1, i).
        Если a_bar ≥ x[i]: распределяем между (i, i+1).
        Веса подобраны так, чтобы сохранить и число (B[i]), и средний объём (a_bar).

    На границах сетки (i==0 при a_bar<x[0] или i==n-1 при a_bar>x[n-1]):
        сохраняем ЧИСЛО — `birth_contrib[i] += B[i]`. Потерю объёма
        логируем (опционально), чтобы пользователь видел расширение
        диапазона сетки v_min/v_max.
    """
    x = grid.centers
    n = grid.n_bins
    birth_contrib = np.zeros(n)

    valid = B > 0
    if not np.any(valid):
        return birth_contrib

    a_bar = np.zeros(n)
    a_bar[valid] = M[valid] / B[valid]

    left = valid & (a_bar < x)   # средний объём ниже центра
    right = valid & ~(a_bar < x)  # средний объём выше или равен центру

    # ----- left случай: a_bar < x[i] -----
    # Внутренние i>0: распределяем между (i-1, i)
    inner_left = left.copy()
    inner_left[0] = False
    if np.any(inner_left):
        ii = np.where(inner_left)[0]
        a_val = B[ii] * (x[ii] - a_bar[ii]) / (x[ii] - x[ii - 1])
        b_val = B[ii] - a_val
        np.add.at(birth_contrib, ii - 1, a_val)
        np.add.at(birth_contrib, ii, b_val)
    # Граница i==0: сохраняем число
    if left[0]:
        birth_contrib[0] += B[0]

    # ----- right случай: a_bar >= x[i] -----
    # Внутренние i<n-1: распределяем между (i, i+1)
    inner_right = right.copy()
    inner_right[n - 1] = False
    if np.any(inner_right):
        ii = np.where(inner_right)[0]
        b_val = B[ii] * (a_bar[ii] - x[ii]) / (x[ii + 1] - x[ii])
        a_val = B[ii] - b_val
        np.add.at(birth_contrib, ii, a_val)
        np.add.at(birth_contrib, ii + 1, b_val)
    # Граница i==n-1: сохраняем число
    if right[n - 1]:
        birth_contrib[n - 1] += B[n - 1]

    return birth_contrib
