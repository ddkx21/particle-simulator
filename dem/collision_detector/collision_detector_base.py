import numpy as np


class CollisionDetector:
    """Абстрактный базовый класс для детекторов столкновений."""

    def detect(self, positions: np.ndarray, radii: np.ndarray,
               *, L: float | None = None, boundary_mode: str | None = None) -> tuple[bool, np.ndarray]:
        """Обнаружить столкновения между частицами.

        Args:
            positions: (N, 3) массив позиций
            radii: (N,) массив радиусов
            L: размер домена (опционально)
            boundary_mode: "periodic" или "open" (опционально)

        Возвращает (is_collision, collided_pairs) где collided_pairs имеет форму (K, 2).
        """
        raise NotImplementedError
