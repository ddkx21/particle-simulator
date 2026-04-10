#!/usr/bin/env python3
"""
Визуализация результатов бенчмарка periodic vs open, direct vs tree.

Читает .npz файлы из results/boundary_benchmark/ и строит графики.
Может работать с неполными данными (если часть симуляций ещё не завершена).

Запуск: python benchmarks/plot_boundary_benchmark.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Пути ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "boundary_benchmark")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Настройки графиков ────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})

# Цвета и стили для 4 конфигураций
STYLE = {
    ("direct", "open"):     {"color": "#2196F3", "ls": "-",  "label": "Direct + Open"},
    ("direct", "periodic"): {"color": "#F44336", "ls": "-",  "label": "Direct + Periodic"},
    ("tree",   "open"):     {"color": "#2196F3", "ls": "--", "label": "Tree + Open"},
    ("tree",   "periodic"): {"color": "#F44336", "ls": "--", "label": "Tree + Periodic"},
}

CONFIGS = [
    ("direct", "open"),
    ("direct", "periodic"),
    ("tree",   "open"),
    ("tree",   "periodic"),
]

DT_VALUES = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]
HISTOGRAM_TIMES = [0, 20, 40, 60, 80, 100]


# ============================================================
#  Загрузка данных
# ============================================================

def load_result(method, boundary, dt):
    """Загружает .npz файл, возвращает dict или None."""
    fname = os.path.join(RESULTS_DIR, f"{method}_{boundary}_dt{dt}.npz")
    if not os.path.exists(fname):
        return None
    data = np.load(fname, allow_pickle=True)
    return dict(data)


def load_all():
    """Загружает все доступные результаты."""
    results = {}
    for dt in DT_VALUES:
        for method, boundary in CONFIGS:
            data = load_result(method, boundary, dt)
            if data is not None:
                results[(method, boundary, dt)] = data
    return results


# ============================================================
#  Вспомогательные функции
# ============================================================

def savefig(name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


# ============================================================
#  Графики для каждого dt
# ============================================================

def _get_time_series(data, light_key, sparse_key):
    """Возвращает (times, values) — предпочитает light_* если доступно."""
    if light_key in data and "light_times" in data:
        return data["light_times"], data[light_key]
    return data["snapshot_times"], data[sparse_key]


def plot_droplet_count_vs_time(results, dt):
    """График 1: N(t) для 4 конфигураций."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, boundary in CONFIGS:
        key = (method, boundary, dt)
        if key not in results:
            continue
        data = results[key]
        t_arr, counts = _get_time_series(
            data, "light_droplet_counts", "droplet_counts"
        )

        s = STYLE[(method, boundary)]
        mean = counts.mean(axis=0).astype(float)
        std = counts.std(axis=0).astype(float)

        ax.plot(t_arr, mean, color=s["color"], ls=s["ls"], label=s["label"])
        ax.fill_between(t_arr, mean - std, mean + std,
                        color=s["color"], alpha=0.12)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Количество капель")
    ax.set_title(f"Количество капель N(t), dt={dt}")
    ax.legend()
    savefig(f"droplet_count_dt{dt}")


def plot_mean_radius_vs_time(results, dt):
    """График 2: <r>(t) для 4 конфигураций."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, boundary in CONFIGS:
        key = (method, boundary, dt)
        if key not in results:
            continue
        data = results[key]
        t_arr, mr = _get_time_series(data, "light_mean_radii", "mean_radii")

        s = STYLE[(method, boundary)]
        mean = mr.mean(axis=0) * 1e6
        std = mr.std(axis=0) * 1e6

        ax.plot(t_arr, mean, color=s["color"], ls=s["ls"], label=s["label"])
        ax.fill_between(t_arr, mean - std, mean + std,
                        color=s["color"], alpha=0.12)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Средний радиус, мкм")
    ax.set_title(f"Средний радиус <r>(t), dt={dt}")
    ax.legend()
    savefig(f"mean_radius_dt{dt}")


def plot_median_radius_vs_time(results, dt):
    """График 3: Медианный радиус(t) для 4 конфигураций."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, boundary in CONFIGS:
        key = (method, boundary, dt)
        if key not in results:
            continue
        data = results[key]
        t_arr, mr = _get_time_series(data, "light_median_radii", "median_radii")

        s = STYLE[(method, boundary)]
        mean = mr.mean(axis=0) * 1e6
        std = mr.std(axis=0) * 1e6

        ax.plot(t_arr, mean, color=s["color"], ls=s["ls"], label=s["label"])
        ax.fill_between(t_arr, mean - std, mean + std,
                        color=s["color"], alpha=0.12)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Медианный радиус, мкм")
    ax.set_title(f"Медианный радиус(t), dt={dt}")
    ax.legend()
    savefig(f"median_radius_dt{dt}")


def plot_kde_distributions(results, dt):
    """График 4: KDE распределений радиусов (2x3 subplots) для 6 моментов времени."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, t_val in enumerate(HISTOGRAM_TIMES):
        ax = axes_flat[idx]

        for method, boundary in CONFIGS:
            key = (method, boundary, dt)
            if key not in results:
                continue
            data = results[key]
            num_runs = int(data["num_runs"])

            # Собираем все радиусы из всех запусков
            all_r = []
            for i in range(num_runs):
                rkey = f"radii_at_t{t_val}_run{i}"
                if rkey in data:
                    all_r.append(data[rkey] * 1e6)

            if not all_r:
                continue
            all_r = np.concatenate(all_r)
            if len(all_r) < 2:
                continue

            s = STYLE[(method, boundary)]
            kde = gaussian_kde(all_r)
            r_grid = np.linspace(all_r.min(), all_r.max(), 300)
            ax.plot(r_grid, kde(r_grid), color=s["color"], ls=s["ls"],
                    label=s["label"], linewidth=1.3)

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title(f"t = {t_val} сек")
        ax.set_xlabel("Радиус, мкм")
        ax.set_ylabel("Плотность")
        if idx == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Распределение по радиусам (KDE), dt={dt}", fontsize=14)
    savefig(f"kde_distributions_dt{dt}")


def plot_coalescence_rate(results, dt):
    """График 5: dN/dt — скорость коалесценции."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, boundary in CONFIGS:
        key = (method, boundary, dt)
        if key not in results:
            continue
        data = results[key]
        t_arr, counts = _get_time_series(
            data, "light_droplet_counts", "droplet_counts"
        )
        counts_f = counts.astype(float)
        mean_counts = counts_f.mean(axis=0)

        # Численная производная (центральная разность)
        dt_metric = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 1.0
        dN_dt = np.gradient(mean_counts, dt_metric)

        s = STYLE[(method, boundary)]
        ax.plot(t_arr, -dN_dt, color=s["color"], ls=s["ls"], label=s["label"])

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("-dN/dt, капель/сек")
    ax.set_title(f"Скорость коалесценции, dt={dt}")
    ax.legend()
    savefig(f"coalescence_rate_dt{dt}")


def plot_volume_conservation(results, dt):
    """График 6: V(t)/V(0) — проверка сохранения объёма."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, boundary in CONFIGS:
        key = (method, boundary, dt)
        if key not in results:
            continue
        data = results[key]

        if "light_volume_ratio" not in data:
            continue

        t_arr = data["light_times"]
        vr = data["light_volume_ratio"]  # (num_runs, num_points)

        s = STYLE[(method, boundary)]
        mean = vr.mean(axis=0)
        std = vr.std(axis=0)

        ax.plot(t_arr, mean, color=s["color"], ls=s["ls"], label=s["label"])
        ax.fill_between(t_arr, mean - std, mean + std,
                        color=s["color"], alpha=0.12)

    ax.set_xlim(left=0)
    ax.axhline(y=1.0, color="black", ls=":", alpha=0.5, label="V(0)")
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("V(t) / V(0)")
    ax.set_title(f"Сохранение объёма, dt={dt}")
    ax.legend()
    savefig(f"volume_conservation_dt{dt}")


def plot_cumulative_collisions(results, dt):
    """График 7: Кумулятивное число столкновений."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, boundary in CONFIGS:
        key = (method, boundary, dt)
        if key not in results:
            continue
        data = results[key]

        if "light_cumulative_collisions" not in data:
            continue

        t_arr = data["light_times"]
        cc = data["light_cumulative_collisions"].astype(float)

        s = STYLE[(method, boundary)]
        mean = cc.mean(axis=0)
        std = cc.std(axis=0)

        ax.plot(t_arr, mean, color=s["color"], ls=s["ls"], label=s["label"])
        ax.fill_between(t_arr, mean - std, mean + std,
                        color=s["color"], alpha=0.12)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Кумулятивное число столкновений")
    ax.set_title(f"Кумулятивные столкновения, dt={dt}")
    ax.legend()
    savefig(f"cumulative_collisions_dt{dt}")


def plot_polydispersity(results, dt):
    """График 8: Коэффициент полидисперсности std(r)/mean(r)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, boundary in CONFIGS:
        key = (method, boundary, dt)
        if key not in results:
            continue
        data = results[key]

        if "light_polydispersity" not in data:
            continue

        t_arr = data["light_times"]
        pd = data["light_polydispersity"]

        s = STYLE[(method, boundary)]
        mean = pd.mean(axis=0)
        std = pd.std(axis=0)

        ax.plot(t_arr, mean, color=s["color"], ls=s["ls"], label=s["label"])
        ax.fill_between(t_arr, mean - std, mean + std,
                        color=s["color"], alpha=0.12)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("std(r) / mean(r)")
    ax.set_title(f"Коэффициент полидисперсности, dt={dt}")
    ax.legend()
    savefig(f"polydispersity_dt{dt}")


def plot_max_radius(results, dt):
    """График 9: Максимальный радиус r_max(t)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, boundary in CONFIGS:
        key = (method, boundary, dt)
        if key not in results:
            continue
        data = results[key]

        if "light_max_radii" not in data:
            continue

        t_arr = data["light_times"]
        mr = data["light_max_radii"]

        s = STYLE[(method, boundary)]
        mean = mr.mean(axis=0) * 1e6
        std = mr.std(axis=0) * 1e6

        ax.plot(t_arr, mean, color=s["color"], ls=s["ls"], label=s["label"])
        ax.fill_between(t_arr, mean - std, mean + std,
                        color=s["color"], alpha=0.12)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("r_max, мкм")
    ax.set_title(f"Максимальный радиус r_max(t), dt={dt}")
    ax.legend()
    savefig(f"max_radius_dt{dt}")


# ============================================================
#  Сводные графики по всем dt
# ============================================================

def plot_convergence_droplet_count(results):
    """График 5: N(t_stop) vs dt для 4 методов."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, boundary in CONFIGS:
        dts = []
        means = []
        stds = []

        for dt in DT_VALUES:
            key = (method, boundary, dt)
            if key not in results:
                continue
            data = results[key]
            final_counts = data["droplet_counts"][:, -1].astype(float)
            dts.append(dt)
            means.append(final_counts.mean())
            stds.append(final_counts.std())

        if not dts:
            continue

        s = STYLE[(method, boundary)]
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(dts, means, yerr=stds, color=s["color"], ls=s["ls"],
                     marker="o", label=s["label"], capsize=3)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("dt")
    ax.set_ylabel(f"N капель при t={int(results[next(iter(results))]['snapshot_times'][-1])}")
    ax.set_xscale("log")
    ax.set_title("Сходимость: количество капель vs dt")
    ax.legend()
    savefig("convergence_droplet_count")


def plot_convergence_mean_radius(results):
    """График 6: <r>(t_stop) vs dt для 4 методов."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, boundary in CONFIGS:
        dts = []
        means = []
        stds = []

        for dt in DT_VALUES:
            key = (method, boundary, dt)
            if key not in results:
                continue
            data = results[key]
            final_mr = data["mean_radii"][:, -1] * 1e6
            dts.append(dt)
            means.append(final_mr.mean())
            stds.append(final_mr.std())

        if not dts:
            continue

        s = STYLE[(method, boundary)]
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(dts, means, yerr=stds, color=s["color"], ls=s["ls"],
                     marker="o", label=s["label"], capsize=3)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("dt")
    ax.set_ylabel("Средний радиус при t_stop, мкм")
    ax.set_xscale("log")
    ax.set_title("Сходимость: средний радиус vs dt")
    ax.legend()
    savefig("convergence_mean_radius")


def plot_timing_comparison(results):
    """График 7: Время расчёта — grouped bar chart по dt и методу."""
    fig, ax = plt.subplots(figsize=(14, 6))

    n_configs = len(CONFIGS)
    width = 0.8 / n_configs
    dt_labels = [str(dt) for dt in DT_VALUES]
    x = np.arange(len(DT_VALUES))

    for i, (method, boundary) in enumerate(CONFIGS):
        means = []
        stds = []
        available = []

        for dt in DT_VALUES:
            key = (method, boundary, dt)
            if key in results:
                times = results[key]["elapsed_times"]
                means.append(times.mean())
                stds.append(times.std())
                available.append(True)
            else:
                means.append(0)
                stds.append(0)
                available.append(False)

        s = STYLE[(method, boundary)]
        offsets = x + (i - n_configs / 2 + 0.5) * width
        bars = ax.bar(offsets, means, width, yerr=stds, label=s["label"],
                       color=s["color"], alpha=0.7 if "-" in s["ls"] else 0.4,
                       capsize=2)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("dt")
    ax.set_ylabel("Время одного запуска, сек")
    ax.set_title("Сравнение времени расчёта")
    ax.set_xticks(x)
    ax.set_xticklabels(dt_labels)
    ax.legend()
    savefig("timing_comparison")


def plot_heatmap_difference(results):
    """График 8: |periodic - open| разность N капель для direct и tree."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, method in enumerate(["direct", "tree"]):
        ax = axes[ax_idx]

        # Собираем матрицу: строки = dt, столбцы = snapshot_times
        snapshot_times = None
        diff_matrix = []
        dt_available = []

        for dt in DT_VALUES:
            key_open = (method, "open", dt)
            key_per = (method, "periodic", dt)

            if key_open not in results or key_per not in results:
                continue

            data_open = results[key_open]
            data_per = results[key_per]

            t_open, counts_open = _get_time_series(
                data_open, "light_droplet_counts", "droplet_counts"
            )
            t_per, counts_per = _get_time_series(
                data_per, "light_droplet_counts", "droplet_counts"
            )

            if snapshot_times is None:
                snapshot_times = t_open

            mean_open = counts_open.mean(axis=0).astype(float)
            mean_per = counts_per.mean(axis=0).astype(float)

            # Относительная разность в %
            denom = np.maximum(mean_open, 1.0)
            diff = np.abs(mean_per - mean_open) / denom * 100
            diff_matrix.append(diff)
            dt_available.append(dt)

        if not diff_matrix or snapshot_times is None:
            ax.text(0.5, 0.5, "Нет данных", transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_title(f"{method}: |periodic - open|")
            continue

        diff_matrix = np.array(diff_matrix)
        im = ax.imshow(diff_matrix, aspect='auto', cmap='YlOrRd',
                        origin='lower')

        ax.set_yticks(range(len(dt_available)))
        ax.set_yticklabels([str(dt) for dt in dt_available])
        ax.set_xticks(range(len(snapshot_times)))
        ax.set_xticklabels([f"{int(t)}" for t in snapshot_times], fontsize=8)
        ax.set_xlabel("Время, сек")
        ax.set_ylabel("dt")
        ax.set_title(f"{method.capitalize()}: |periodic - open|, %")

        fig.colorbar(im, ax=ax, shrink=0.8, label="Относительная разность, %")

    fig.suptitle("Разность N капель: periodic vs open", fontsize=14)
    savefig("heatmap_periodic_vs_open")


# ============================================================
#  Главная функция
# ============================================================

def main():
    print("Загрузка результатов...")
    results = load_all()

    if not results:
        print("Нет данных в", RESULTS_DIR)
        print("Сначала запустите: python benchmarks/run_boundary_benchmark.py")
        sys.exit(1)

    print(f"Загружено {len(results)} конфигураций")

    # Доступные dt
    available_dts = sorted(set(dt for _, _, dt in results.keys()))
    print(f"Доступные dt: {available_dts}")

    # Графики для каждого dt
    for dt in available_dts:
        print(f"\nГрафики для dt={dt}:")
        plot_droplet_count_vs_time(results, dt)
        plot_mean_radius_vs_time(results, dt)
        plot_median_radius_vs_time(results, dt)
        plot_kde_distributions(results, dt)
        plot_coalescence_rate(results, dt)
        plot_volume_conservation(results, dt)
        plot_cumulative_collisions(results, dt)
        plot_polydispersity(results, dt)
        plot_max_radius(results, dt)

    # Сводные графики
    print("\nСводные графики:")
    plot_convergence_droplet_count(results)
    plot_convergence_mean_radius(results)
    plot_timing_comparison(results)
    plot_heatmap_difference(results)

    print(f"\nВсе графики сохранены в {PLOTS_DIR}")


if __name__ == "__main__":
    main()
