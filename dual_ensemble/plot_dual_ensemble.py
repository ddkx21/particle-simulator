"""
Построение графиков для двойного ансамбля капель.

Каждая система имеет свои snapshot_times и t_stop.
Сводные графики для каждой системы + кросс-временная матрица сходимости.

Запуск (автономный):
  python -m dual_ensemble.plot_dual_ensemble
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dual_ensemble.metrics import (
    compute_all_scalar_metrics,
    cross_time_convergence,
    detect_tail_convergence,
    ecdf_tail,
    volume_weighted_histogram,
)
from dual_ensemble.run_common import normalize_time, snapshot_key

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

SYS1_COLOR = "tab:blue"
SYS2_COLOR = "tab:red"


def _load_sys_snapshots(path: str, prefix: str) -> dict:
    """Загрузить snapshots одной системы из .npz."""
    data = np.load(path)
    times = [normalize_time(t) for t in data["snapshot_times"]]
    radii = {}
    for t in times:
        radii[t] = data[snapshot_key(prefix, t)]
    return {
        "snapshot_times": times,
        "radii": radii,
        "label": data["label"].item(),
        "radii_range": data["radii_range"],
        "box_size": float(data["box_size"]),
        "N_initial": int(data["N_initial"]),
        "elapsed": float(data["elapsed"]),
        "accumulated_time": float(data["accumulated_time"]),
    }


def load_results(sys1_path: str | None = None, sys2_path: str | None = None) -> dict:
    if sys1_path is None:
        sys1_path = os.path.join(RESULTS_DIR, "sys1_snapshots.npz")
    if sys2_path is None:
        sys2_path = os.path.join(RESULTS_DIR, "sys2_snapshots.npz")

    s1 = _load_sys_snapshots(sys1_path, "sys1")
    s2 = _load_sys_snapshots(sys2_path, "sys2")

    return {
        "sys1_snapshot_times": s1["snapshot_times"],
        "sys2_snapshot_times": s2["snapshot_times"],
        "box_size": s1["box_size"],
        "N1": s1["N_initial"],
        "N2": s2["N_initial"],
        "sys1_label": s1["label"],
        "sys2_label": s2["label"],
        "sys1_radii": s1["radii"],
        "sys2_radii": s2["radii"],
        "sys1_elapsed": s1["elapsed"],
        "sys2_elapsed": s2["elapsed"],
        "sys1_t_stop": s1["accumulated_time"],
        "sys2_t_stop": s2["accumulated_time"],
    }


def _compute_metrics(radii_dict: dict[float, np.ndarray], times: list[float]) -> dict[float, dict[str, float]]:
    return {t: compute_all_scalar_metrics(radii_dict[t]) for t in times}


def _save_fig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Сохранён: {path}")


# ── Per-system timeseries ──────────────────────────────────────

def _plot_metric_overlay(
    results: dict,
    sys1_metrics: dict,
    sys2_metrics: dict,
    metric_key: str,
    ylabel: str,
    title: str,
    filename: str,
    scale: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    t1 = results["sys1_snapshot_times"]
    t2 = results["sys2_snapshot_times"]
    v1 = [sys1_metrics[t][metric_key] * scale for t in t1]
    v2 = [sys2_metrics[t][metric_key] * scale for t in t2]

    ax.plot(t1, v1, "o-", color=SYS1_COLOR, linewidth=2, label=results["sys1_label"])
    ax.plot(t2, v2, "s-", color=SYS2_COLOR, linewidth=2, label=results["sys2_label"])

    ax.set_xlabel("Время, сек")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(PLOTS_DIR, filename))


def plot_all_timeseries(results: dict, sys1_m: dict, sys2_m: dict) -> None:
    specs = [
        ("mean_radius",             "Средний радиус, мкм",  "Средний радиус <r>(t)",                          "mean_radius.png",             1e6),
        ("median_radius",           "Медианный радиус, мкм","Медианный радиус",                                "median_radius.png",           1e6),
        ("volume_weighted_radius",  "r_vw, мкм",           "Объёмно-взвешенный радиус r_vw(t) = Σ(r⁴)/Σ(r³)","volume_weighted_radius.png",  1e6),
        ("sauter_diameter",         "D₃₂, мкм",           "Диаметр Заутера D₃₂(t) = 2·Σ(r³)/Σ(r²)",        "sauter_diameter.png",         1e6),
        ("droplet_count",           "Число капель N(t)",    "Эволюция числа капель N(t)",                      "droplet_count.png",           1.0),
    ]
    for key, ylabel, title, fname, scale in specs:
        _plot_metric_overlay(results, sys1_m, sys2_m, key, ylabel, title, fname, scale)


# ── Percentiles (per-system panels) ───────────────────────────

def _plot_percentiles_panel(
    results: dict,
    sys1_m: dict,
    sys2_m: dict,
    prefix: str,
    panel_title: str,
    filename: str,
) -> None:
    ps = [5, 25, 50, 75, 95]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, metrics, times, label in [
        (axes[0], sys1_m, results["sys1_snapshot_times"], results["sys1_label"]),
        (axes[1], sys2_m, results["sys2_snapshot_times"], results["sys2_label"]),
    ]:
        min_v = [metrics[t]["min_radius"] * 1e6 for t in times]
        max_v = [metrics[t]["max_radius"] * 1e6 for t in times]
        ax.plot(times, min_v, "--", color="gray", alpha=0.5, label="min")
        ax.plot(times, max_v, "--", color="gray", alpha=0.5, label="max")
        for p in ps:
            vals = [metrics[t][f"{prefix}{p}"] * 1e6 for t in times]
            ax.plot(times, vals, "o-", linewidth=1.5, label=f"{prefix.upper()}{p}")
        ax.set_xlabel("Время, сек")
        ax.set_ylabel("Радиус, мкм")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(panel_title, fontsize=14)
    fig.tight_layout()
    _save_fig(fig, os.path.join(PLOTS_DIR, filename))


def plot_percentiles(results: dict, sys1_m: dict, sys2_m: dict) -> None:
    _plot_percentiles_panel(results, sys1_m, sys2_m, "p", "Процентили радиуса", "radius_percentiles.png")
    _plot_percentiles_panel(results, sys1_m, sys2_m, "vp", "Объёмные процентили радиуса", "volume_percentiles.png")


# ── Distribution grids (per-system, own snapshot_times) ───────

def _per_system_subplot_grid(
    radii_dict: dict[float, np.ndarray],
    times: list[float],
    plot_func,
    suptitle: str,
    filename: str,
    label: str,
) -> None:
    n = len(times)
    ncols = min(n, 4)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, t in enumerate(times):
        ax = axes_flat[idx]
        r = radii_dict[t]
        plot_func(ax, r, t)

    for idx in range(len(times), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"{suptitle} — {label}", fontsize=14)
    fig.tight_layout()
    _save_fig(fig, os.path.join(PLOTS_DIR, filename))


def _plot_hist_cell(ax: plt.Axes, r: np.ndarray, t: float) -> None:
    r_um = r * 1e6
    if len(r_um) > 1:
        bins = np.linspace(r_um.min(), r_um.max(), 31)
        ax.hist(r_um, bins=bins, density=True, alpha=0.7, color="steelblue")
    ax.set_title(f"t = {t:.0f} сек")
    ax.set_xlabel("Радиус, мкм")
    ax.set_ylabel("Плотность")
    ax.grid(True, alpha=0.3)


def _plot_vhist_cell(ax: plt.Axes, r: np.ndarray, t: float) -> None:
    if len(r) > 1:
        bins_m = np.linspace(r.min(), r.max(), 31)
        c, h = volume_weighted_histogram(r, bins_m)
        ax.plot(c * 1e6, h, linewidth=1.5, color="steelblue")
    ax.set_title(f"t = {t:.0f} сек")
    ax.set_xlabel("Радиус, мкм")
    ax.set_ylabel("dV/dr (норм.)")
    ax.grid(True, alpha=0.3)


def _plot_kde_cell(ax: plt.Axes, r: np.ndarray, t: float) -> None:
    r_um = r * 1e6
    if len(r_um) > 1:
        x = np.linspace(r_um.min() * 0.95, r_um.max() * 1.05, 200)
        kde = gaussian_kde(r_um)
        ax.plot(x, kde(x), linewidth=1.5, color="steelblue")
    ax.set_title(f"t = {t:.0f} сек")
    ax.set_xlabel("Радиус, мкм")
    ax.set_ylabel("Плотность (KDE)")
    ax.grid(True, alpha=0.3)


def plot_distributions(results: dict) -> None:
    for sys_key, times_key, suffix, label in [
        ("sys1_radii", "sys1_snapshot_times", "sys1", results["sys1_label"]),
        ("sys2_radii", "sys2_snapshot_times", "sys2", results["sys2_label"]),
    ]:
        radii = results[sys_key]
        times = results[times_key]
        _per_system_subplot_grid(radii, times, _plot_hist_cell,
                                 "Распределение по радиусам", f"radii_distribution_{suffix}.png", label)
        _per_system_subplot_grid(radii, times, _plot_vhist_cell,
                                 "Объёмно-взвешенное распределение dV/dr", f"volume_histograms_{suffix}.png", label)
        _per_system_subplot_grid(radii, times, _plot_kde_cell,
                                 "KDE распределение", f"radii_kde_{suffix}.png", label)


# ── ECDF tails per-system ─────────────────────────────────────

def plot_ecdf_per_system(results: dict, cfg: dict) -> None:
    r_threshold = cfg["convergence"]["r_threshold_um"] * 1e-6
    n_points = cfg["convergence"]["ecdf_r_star_points"]

    for sys_key, times_key, suffix, label in [
        ("sys1_radii", "sys1_snapshot_times", "sys1", results["sys1_label"]),
        ("sys2_radii", "sys2_snapshot_times", "sys2", results["sys2_label"]),
    ]:
        radii = results[sys_key]
        times = results[times_key]
        n = len(times)
        ncols = min(n, 4)
        nrows = max(1, (n + ncols - 1) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, t in enumerate(times):
            ax = axes_flat[idx]
            r = radii[t]
            tail = r[r > r_threshold]
            if len(tail) > 0:
                r_eval = np.linspace(r_threshold, np.max(tail), n_points)
                f = ecdf_tail(r, r_threshold, r_eval)
                ax.plot(r_eval * 1e6, f, "o-", linewidth=1.5, color="steelblue")
            ax.set_title(f"t = {t:.0f} сек")
            ax.set_xlabel("r*, мкм")
            ax.set_ylabel("F_tail(r*)")
            ax.grid(True, alpha=0.3)

        for idx in range(n, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(f"ECDF хвоста (r > {r_threshold*1e6:.1f} мкм) — {label}", fontsize=14)
        fig.tight_layout()
        _save_fig(fig, os.path.join(PLOTS_DIR, f"ecdf_tails_{suffix}.png"))


# ── Cross-time convergence heatmap ────────────────────────────

def plot_convergence_heatmap(results: dict, cfg: dict) -> None:
    r_threshold = cfg["convergence"]["r_threshold_um"] * 1e-6
    epsilon = cfg["convergence"]["epsilon"]
    n_points = cfg["convergence"]["ecdf_r_star_points"]

    t1_list = results["sys1_snapshot_times"]
    t2_list = results["sys2_snapshot_times"]

    diff_matrix, best_t1, best_t2 = cross_time_convergence(
        results["sys1_radii"], results["sys2_radii"],
        t1_list, t2_list,
        r_threshold, epsilon, n_points,
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        diff_matrix, origin="lower", aspect="auto",
        extent=[t2_list[0], t2_list[-1], t1_list[0], t1_list[-1]],
        cmap="RdYlGn_r", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax, label="max |F_tail_1 − F_tail_2|")

    ax.contour(
        t2_list, t1_list, diff_matrix,
        levels=[epsilon], colors="white", linewidths=2, linestyles="--",
    )

    if best_t1 is not None and best_t2 is not None:
        ax.plot(best_t2, best_t1, "w*", markersize=15, markeredgecolor="black")
        delta_t = abs(best_t1 - best_t2)
        ax.set_title(
            f"Кросс-временная сходимость хвостов\n"
            f"t₁ = {best_t1:.0f} сек, t₂ = {best_t2:.0f} сек, Δt = {delta_t:.0f} сек",
            fontsize=13,
        )
        print(f"\n  Сходимость: t₁={best_t1:.0f}, t₂={best_t2:.0f}, Δt = |t₁ − t₂| = {delta_t:.0f} сек")
    else:
        ax.set_title("Кросс-временная сходимость хвостов (НЕ сошлись)", fontsize=13)
        print(f"\n  Хвосты НЕ сошлись (min max_diff = {np.nanmin(diff_matrix):.4f})")

    ax.set_xlabel(f"Время {results['sys2_label']}, сек")
    ax.set_ylabel(f"Время {results['sys1_label']}, сек")
    ax.grid(False)
    fig.tight_layout()
    _save_fig(fig, os.path.join(PLOTS_DIR, "convergence_heatmap.png"))


# ── Step plots (per plot_interval) ────────────────────────────

def plot_step_plots(results: dict, cfg: dict) -> None:
    plot_interval = cfg["simulation"].get("plot_interval", 10.0)
    r_threshold = cfg["convergence"]["r_threshold_um"] * 1e-6
    n_points = cfg["convergence"]["ecdf_r_star_points"]

    for sys_key, times_key, suffix, label, color in [
        ("sys1_radii", "sys1_snapshot_times", "sys1", results["sys1_label"], SYS1_COLOR),
        ("sys2_radii", "sys2_snapshot_times", "sys2", results["sys2_label"], SYS2_COLOR),
    ]:
        radii = results[sys_key]
        times = results[times_key]
        plot_times = [t for t in times
                      if abs(t % plot_interval) < 0.01
                      or abs(t % plot_interval - plot_interval) < 0.01]
        if not plot_times:
            plot_times = times

        for t in plot_times:
            step_dir = os.path.join(PLOTS_DIR, f"{suffix}_step_t{t:.0f}")
            r = radii[t]
            r_um = r * 1e6

            fig, ax = plt.subplots(figsize=(8, 5))
            if len(r_um) > 1:
                bins = np.linspace(r_um.min(), r_um.max(), 31)
                ax.hist(r_um, bins=bins, alpha=0.7, color=color, density=True)
            ax.set_xlabel("Радиус, мкм")
            ax.set_ylabel("Плотность")
            ax.set_title(f"{label}, t = {t:.0f} сек")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _save_fig(fig, os.path.join(step_dir, "radius_histogram.png"))

            fig, ax = plt.subplots(figsize=(8, 5))
            if len(r) > 1:
                bins_m = np.linspace(r.min(), r.max(), 31)
                c, h = volume_weighted_histogram(r, bins_m)
                ax.plot(c * 1e6, h, linewidth=1.5, color=color)
            ax.set_xlabel("Радиус, мкм")
            ax.set_ylabel("dV/dr (норм.)")
            ax.set_title(f"{label}, t = {t:.0f} сек")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _save_fig(fig, os.path.join(step_dir, "volume_histogram.png"))

            fig, ax = plt.subplots(figsize=(8, 5))
            tail = r[r > r_threshold]
            if len(tail) > 0:
                r_eval = np.linspace(r_threshold, np.max(tail), n_points)
                f = ecdf_tail(r, r_threshold, r_eval)
                ax.plot(r_eval * 1e6, f, "o-", linewidth=1.5, color=color)
            ax.set_xlabel("r*, мкм")
            ax.set_ylabel("F_tail(r*)")
            ax.set_title(f"{label}, t = {t:.0f} сек")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _save_fig(fig, os.path.join(step_dir, "ecdf_tail.png"))

            fig, ax = plt.subplots(figsize=(8, 5))
            if len(r_um) > 1:
                x_grid = np.linspace(r_um.min() * 0.95, r_um.max() * 1.05, 200)
                kde = gaussian_kde(r_um)
                ax.plot(x_grid, kde(x_grid), linewidth=1.5, color=color)
            ax.set_xlabel("Радиус, мкм")
            ax.set_ylabel("Плотность (KDE)")
            ax.set_title(f"{label}, t = {t:.0f} сек")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _save_fig(fig, os.path.join(step_dir, "radius_kde.png"))


# ── Entry point ───────────────────────────────────────────────

def plot_all(cfg: dict) -> None:
    print("\nЗагрузка результатов...")
    results = load_results()
    print(f"  {results['sys1_label']}: N₀={results['N1']}, t_stop={results['sys1_t_stop']:.0f}, расчёт={results['sys1_elapsed']:.1f} сек")
    print(f"  {results['sys2_label']}: N₀={results['N2']}, t_stop={results['sys2_t_stop']:.0f}, расчёт={results['sys2_elapsed']:.1f} сек")

    sys1_m = _compute_metrics(results["sys1_radii"], results["sys1_snapshot_times"])
    sys2_m = _compute_metrics(results["sys2_radii"], results["sys2_snapshot_times"])

    print("\nСводные графики (time-series overlay):")
    plot_all_timeseries(results, sys1_m, sys2_m)

    print("\nПроцентили:")
    plot_percentiles(results, sys1_m, sys2_m)

    print("\nРаспределения (per-system):")
    plot_distributions(results)

    print("\nECDF хвоста (per-system):")
    plot_ecdf_per_system(results, cfg)

    print("\nКросс-временная сходимость:")
    plot_convergence_heatmap(results, cfg)

    print("\nПошаговые графики:")
    plot_step_plots(results, cfg)

    print("\nВсе графики построены!")


def main() -> None:
    cfg_path = os.path.join(SCRIPT_DIR, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    sys1_path = os.path.join(RESULTS_DIR, "sys1_snapshots.npz")
    sys2_path = os.path.join(RESULTS_DIR, "sys2_snapshots.npz")
    for p in (sys1_path, sys2_path):
        if not os.path.isfile(p):
            print(f"Файл не найден: {p}")
            print("Сначала запустите: python -m dual_ensemble.run_system1 и python -m dual_ensemble.run_system2")
            sys.exit(1)

    plot_all(cfg)


if __name__ == "__main__":
    main()
