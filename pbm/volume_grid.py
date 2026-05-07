from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class VolumeGrid:
    """Дискретизация объёмного пространства для PBM."""

    def __init__(
        self,
        v_min: float,
        v_max: float,
        n_bins: int,
        spacing: str = "geometric",
    ) -> None:
        if v_min <= 0 or v_max <= v_min:
            raise ValueError("Требуется 0 < v_min < v_max")
        if n_bins < 2:
            raise ValueError("Требуется n_bins >= 2")

        self.n_bins = n_bins
        self.spacing = spacing

        if spacing == "geometric":
            self.edges: NDArray[np.float64] = np.geomspace(v_min, v_max, n_bins + 1)
        elif spacing == "logarithmic":
            log_min = np.log10(v_min)
            log_max = np.log10(v_max)
            self.edges = 10 ** np.linspace(log_min, log_max, n_bins + 1)
        elif spacing == "linear":
            self.edges = np.linspace(v_min, v_max, n_bins + 1)
        else:
            raise ValueError(f"Неизвестный тип spacing: {spacing}")

        # Для лог/гео сетки — геометрический центр (Kumar 2006). Для линейной — арифметический.
        if spacing in ("geometric", "logarithmic"):
            self.centers: NDArray[np.float64] = np.sqrt(self.edges[:-1] * self.edges[1:])
        else:
            self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.widths: NDArray[np.float64] = np.diff(self.edges)

        self._warned_clamp_low = False
        self._warned_clamp_high = False

    @classmethod
    def from_radii_range(
        cls,
        r_min: float,
        r_max: float,
        n_bins: int,
        spacing: str = "geometric",
    ) -> VolumeGrid:
        v_min = (4.0 / 3.0) * np.pi * r_min**3
        v_max = (4.0 / 3.0) * np.pi * r_max**3
        return cls(v_min, v_max, n_bins, spacing)

    def bin_index(self, volume: float) -> int:
        raw = int(np.searchsorted(self.edges, volume, side="right")) - 1
        if raw < 0 and not self._warned_clamp_low:
            logger.warning(
                "VolumeGrid.bin_index: volume=%.3e < v_min=%.3e, зажимаем в bin 0",
                volume,
                self.edges[0],
            )
            self._warned_clamp_low = True
        elif raw >= self.n_bins and not self._warned_clamp_high:
            logger.warning(
                "VolumeGrid.bin_index: volume=%.3e > v_max=%.3e, зажимаем в bin %d. "
                "Увеличьте v_max, иначе переростки молча накапливаются в крайнем бине.",
                volume,
                self.edges[-1],
                self.n_bins - 1,
            )
            self._warned_clamp_high = True
        return max(0, min(raw, self.n_bins - 1))

    def bin_indices(self, volumes: NDArray[np.float64]) -> NDArray[np.intp]:
        raw = np.searchsorted(self.edges, volumes, side="right").astype(np.intp) - 1
        if not self._warned_clamp_high and np.any(raw >= self.n_bins):
            logger.warning(
                "VolumeGrid.bin_indices: %d объёмов > v_max=%.3e, зажимаем в bin %d.",
                int(np.sum(raw >= self.n_bins)),
                self.edges[-1],
                self.n_bins - 1,
            )
            self._warned_clamp_high = True
        if not self._warned_clamp_low and np.any(raw < 0):
            logger.warning(
                "VolumeGrid.bin_indices: %d объёмов < v_min=%.3e, зажимаем в bin 0.",
                int(np.sum(raw < 0)),
                self.edges[0],
            )
            self._warned_clamp_low = True
        np.clip(raw, 0, self.n_bins - 1, out=raw)
        return raw

    def histogram(self, radii: NDArray[np.float64]) -> NDArray[np.float64]:
        volumes = (4.0 / 3.0) * np.pi * radii**3
        counts, _ = np.histogram(volumes, bins=self.edges)
        # np.histogram включает правую границу в последний бин, но СТРОГИЕ переростки
        # (v > v_max) теряются. Учитываем их явно.
        n_overflow = int(np.sum(volumes > self.edges[-1]))
        if n_overflow > 0:
            counts[-1] += n_overflow
            if not self._warned_clamp_high:
                logger.warning(
                    "VolumeGrid.histogram: %d частиц с v > v_max=%.3e сложены в крайний бин.",
                    n_overflow,
                    self.edges[-1],
                )
                self._warned_clamp_high = True
        # Симметрично — частицы строго ниже v_min теряются: складываем в bin 0.
        n_underflow = int(np.sum(volumes < self.edges[0]))
        if n_underflow > 0:
            counts[0] += n_underflow
            if not self._warned_clamp_low:
                logger.warning(
                    "VolumeGrid.histogram: %d частиц с v < v_min=%.3e сложены в bin 0.",
                    n_underflow,
                    self.edges[0],
                )
                self._warned_clamp_low = True
        return counts.astype(np.float64)
