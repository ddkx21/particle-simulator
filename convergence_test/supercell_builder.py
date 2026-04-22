"""
Построение суперячейки K³ из периодических копий исходной ячейки.
"""
import numpy as np


def build_supercell(positions: np.ndarray, radii: np.ndarray,
                    L: float, K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит суперячейку K³ копий ячейки [0, L]³.

    Смещения по каждой оси: i ∈ [-(K//2), K//2].
    Центральная ячейка = смещение (0,0,0), её индексы: [0:N].

    :param positions: (N, 3) позиции частиц в [0, L]³
    :param radii: (N,) радиусы
    :param L: размер ячейки
    :param K: число копий по каждой оси (нечётное)
    :return: (super_positions (K³·N, 3), super_radii (K³·N,))
             Первые N строк — центральная ячейка (смещение 0,0,0).
    """
    assert K % 2 == 1, f"K должен быть нечётным, получено {K}"
    N = positions.shape[0]
    half = K // 2

    all_positions = []
    all_radii = []

    # Центральная ячейка первой (смещение 0,0,0)
    all_positions.append(positions.copy())
    all_radii.append(radii.copy())

    # Остальные смещения
    for ix in range(-half, half + 1):
        for iy in range(-half, half + 1):
            for iz in range(-half, half + 1):
                if ix == 0 and iy == 0 and iz == 0:
                    continue
                offset = np.array([ix * L, iy * L, iz * L])
                all_positions.append(positions + offset)
                all_radii.append(radii.copy())

    super_positions = np.concatenate(all_positions, axis=0)
    super_radii = np.concatenate(all_radii, axis=0)

    assert super_positions.shape[0] == K**3 * N
    return super_positions, super_radii
