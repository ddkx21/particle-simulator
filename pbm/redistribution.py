from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pbm.volume_grid import VolumeGrid


def fixed_pivot(
    v_new: float,
    grid: VolumeGrid,
) -> list[tuple[int, float]]:
    """Перераспределение fixed-pivot: линейная интерполяция между ближайшими центрами.

    Сохраняет общее число и общий объём.
    Возвращает список (bin_index, weight).
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
    """Метод Cell-Average: перераспределение с сохранением числа и массы.

    B[i] — суммарная скорость рождения в ячейке i.
    M[i] — суммарный объём новорождённых в ячейке i (= Σ rate * v_new).
    Возвращает birth_contrib[i] — итоговые вклады в каждый узел.
    """
    x = grid.centers
    n = grid.n_bins
    birth_contrib = np.zeros(n)

    for i in range(n):
        if B[i] <= 0:
            continue

        a_bar = M[i] / B[i]

        if a_bar < x[i]:
            if i == 0:
                birth_contrib[0] += B[i]
            else:
                a_val = B[i] * (x[i] - a_bar) / (x[i] - x[i - 1])
                b_val = B[i] - a_val
                birth_contrib[i - 1] += a_val
                birth_contrib[i] += b_val
        else:
            if i == n - 1:
                birth_contrib[i] += B[i]
            else:
                b_val = B[i] * (a_bar - x[i]) / (x[i + 1] - x[i])
                a_val = B[i] - b_val
                birth_contrib[i] += a_val
                birth_contrib[i + 1] += b_val

    return birth_contrib
