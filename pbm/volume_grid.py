from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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

        self.centers: NDArray[np.float64] = (self.edges[:-1] + self.edges[1:]) / 2
        self.widths: NDArray[np.float64] = np.diff(self.edges)

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
        idx = int(np.searchsorted(self.edges, volume, side="right")) - 1
        return max(0, min(idx, self.n_bins - 1))

    def bin_indices(self, volumes: NDArray[np.float64]) -> NDArray[np.intp]:
        idx = np.searchsorted(self.edges, volumes, side="right").astype(np.intp) - 1
        np.clip(idx, 0, self.n_bins - 1, out=idx)
        return idx

    def histogram(self, radii: NDArray[np.float64]) -> NDArray[np.float64]:
        volumes = (4.0 / 3.0) * np.pi * radii**3
        counts, _ = np.histogram(volumes, bins=self.edges)
        return counts.astype(np.float64)
