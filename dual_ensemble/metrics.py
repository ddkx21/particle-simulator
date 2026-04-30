import numpy as np


def volume_weighted_radius(radii: np.ndarray) -> float:
    """r_vw = Σ(r⁴) / Σ(r³). Радиусы в метрах."""
    if len(radii) == 0:
        return 0.0
    r3_sum = np.sum(radii ** 3)
    if r3_sum == 0.0:
        return 0.0
    return float(np.sum(radii ** 4) / r3_sum)


def sauter_diameter(radii: np.ndarray) -> float:
    """D₃₂ = 2·Σ(r³) / Σ(r²). Радиусы в метрах."""
    if len(radii) == 0:
        return 0.0
    r2_sum = np.sum(radii ** 2)
    if r2_sum == 0.0:
        return 0.0
    return float(2.0 * np.sum(radii ** 3) / r2_sum)


def mean_radius(radii: np.ndarray) -> float:
    if len(radii) == 0:
        return 0.0
    return float(np.mean(radii))


def median_radius(radii: np.ndarray) -> float:
    if len(radii) == 0:
        return 0.0
    return float(np.median(radii))


def droplet_count(radii: np.ndarray) -> int:
    return len(radii)


def percentiles(radii: np.ndarray, ps: list[float]) -> dict[float, float]:
    if len(radii) == 0:
        return {p: 0.0 for p in ps}
    vals = np.percentile(radii, ps)
    return {p: float(v) for p, v in zip(ps, vals)}


def volume_percentiles(radii: np.ndarray, ps: list[float]) -> dict[float, float]:
    """Процентили по объёму: радиус, при котором суммарный объём от меньших капель = p% от общего."""
    if len(radii) == 0:
        return {p: 0.0 for p in ps}
    volumes = (4.0 / 3.0) * np.pi * radii ** 3
    total_volume = np.sum(volumes)
    sorted_volumes = np.sort(volumes)
    cumulative_volumes = np.cumsum(sorted_volumes)
    result = {}
    for p in ps:
        target = p / 100.0 * total_volume
        idx = np.searchsorted(cumulative_volumes, target)
        idx = min(idx, len(sorted_volumes) - 1)
        result[p] = float((3.0 / (4.0 * np.pi) * sorted_volumes[idx]) ** (1.0 / 3.0))
    return result


def ecdf_tail(
    radii: np.ndarray,
    r_threshold: float,
    r_eval: np.ndarray,
) -> np.ndarray:
    """F_tail(r*) = P(R > r* | R > r_threshold) — survival function хвоста."""
    tail_radii = radii[radii > r_threshold]
    if len(tail_radii) == 0:
        return np.zeros_like(r_eval)
    return np.mean(tail_radii[np.newaxis, :] > r_eval[:, np.newaxis], axis=1)


def volume_weighted_histogram(
    radii: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """dV/dr — гистограмма с весами r³ (нормированная)."""
    if len(radii) == 0:
        centers = 0.5 * (bins[:-1] + bins[1:])
        return centers, np.zeros(len(bins) - 1)
    weights = radii ** 3
    hist, bin_edges = np.histogram(radii, bins=bins, weights=weights, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, hist


def detect_tail_convergence(
    radii1: np.ndarray,
    radii2: np.ndarray,
    r_threshold: float,
    epsilon: float,
    n_eval_points: int = 20,
) -> tuple[bool, float]:
    """Проверка |F_tail_1(r*) − F_tail_2(r*)| < ε для всех r* > threshold."""
    all_tail = np.concatenate([
        radii1[radii1 > r_threshold],
        radii2[radii2 > r_threshold],
    ])
    if len(all_tail) == 0:
        return True, 0.0

    r_eval = np.linspace(r_threshold, np.max(all_tail), n_eval_points)
    f1 = ecdf_tail(radii1, r_threshold, r_eval)
    f2 = ecdf_tail(radii2, r_threshold, r_eval)
    max_diff = float(np.max(np.abs(f1 - f2)))
    return max_diff < epsilon, max_diff


def cross_time_convergence(
    sys1_radii: dict[float, np.ndarray],
    sys2_radii: dict[float, np.ndarray],
    sys1_times: list[float],
    sys2_times: list[float],
    r_threshold: float,
    epsilon: float,
    n_eval_points: int = 20,
) -> tuple[np.ndarray, float | None, float | None]:
    """Кросс-временная матрица сходимости хвостов.

    Для каждой пары (t1, t2) вычисляет max|F_tail_1(r*, t1) − F_tail_2(r*, t2)|.
    Находит первую пару (t1, t2) где diff < epsilon.

    Returns:
        diff_matrix: shape (len(sys1_times), len(sys2_times))
        t1_converge: время системы 1 при сходимости (None если не сошлись)
        t2_converge: время системы 2 при сходимости (None если не сошлись)
    """
    n1 = len(sys1_times)
    n2 = len(sys2_times)
    diff_matrix = np.full((n1, n2), np.nan)

    for i, t1 in enumerate(sys1_times):
        r1 = sys1_radii[t1]
        for j, t2 in enumerate(sys2_times):
            r2 = sys2_radii[t2]
            _, max_diff = detect_tail_convergence(r1, r2, r_threshold, epsilon, n_eval_points)
            diff_matrix[i, j] = max_diff

    best_t1 = None
    best_t2 = None
    best_diff = float("inf")
    for i in range(n1):
        for j in range(n2):
            if diff_matrix[i, j] < epsilon and diff_matrix[i, j] < best_diff:
                best_diff = diff_matrix[i, j]
                best_t1 = sys1_times[i]
                best_t2 = sys2_times[j]

    return diff_matrix, best_t1, best_t2


def compute_all_scalar_metrics(radii: np.ndarray) -> dict[str, float]:
    ps = [5.0, 25.0, 50.0, 75.0, 95.0]
    p_vals = percentiles(radii, ps)
    vp_vals = volume_percentiles(radii, ps)
    return {
        "mean_radius": mean_radius(radii),
        "median_radius": median_radius(radii),
        "volume_weighted_radius": volume_weighted_radius(radii),
        "sauter_diameter": sauter_diameter(radii),
        "droplet_count": float(droplet_count(radii)),
        **{f"p{int(p)}": v for p, v in p_vals.items()},
        **{f"vp{int(p)}": v for p, v in vp_vals.items()},
        "min_radius": float(np.min(radii)) if len(radii) > 0 else 0.0,
        "max_radius": float(np.max(radii)) if len(radii) > 0 else 0.0,
    }
