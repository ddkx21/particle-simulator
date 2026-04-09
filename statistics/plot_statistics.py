"""
Построение графиков статистики: direct vs tree.

Читает results/statistics_direct.npz и results/statistics_tree.npz,
строит графики в plots/.

Запуск: python statistics/plot_statistics.py
"""

import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data(filename):
    """Загрузка npz и извлечение структурированных данных."""
    path = os.path.join(RESULTS_DIR, filename)
    data = np.load(path)

    snapshot_times = data["snapshot_times"]
    elapsed_times = data["elapsed_times"]
    droplet_counts = data["droplet_counts"]
    median_radii = data["median_radii"]

    num_runs = elapsed_times.shape[0]

    # Извлекаем histogram_times из ключей вида radii_at_t{T}_run{i}
    hist_times = set()
    for key in data.keys():
        m = re.match(r"radii_at_t(\d+)_run\d+", key)
        if m:
            hist_times.add(int(m.group(1)))
    histogram_times = sorted(hist_times)

    # Собираем радиусы: {t: [array_run0, array_run1, ...]}
    radii_by_time = {}
    for t in histogram_times:
        runs = []
        for i in range(num_runs):
            key = f"radii_at_t{t}_run{i}"
            if key in data:
                runs.append(data[key])
        radii_by_time[t] = runs

    return {
        "snapshot_times": snapshot_times,
        "elapsed_times": elapsed_times,
        "droplet_counts": droplet_counts,
        "median_radii": median_radii,
        "num_runs": num_runs,
        "histogram_times": histogram_times,
        "radii_by_time": radii_by_time,
    }


def plot_droplet_count(direct, tree):
    """График N(t) с mean +/- std и полупрозрачными реализациями."""
    t_arr = direct["snapshot_times"]
    num_runs = direct["num_runs"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(num_runs):
        ax.plot(t_arr, direct["droplet_counts"][i], color="tab:blue", alpha=0.1, linewidth=0.5)
        ax.plot(t_arr, tree["droplet_counts"][i], color="tab:red", alpha=0.1, linewidth=0.5)

    d_mean = direct["droplet_counts"].mean(axis=0)
    d_std = direct["droplet_counts"].std(axis=0)
    t_mean = tree["droplet_counts"].mean(axis=0)
    t_std = tree["droplet_counts"].std(axis=0)

    ax.plot(t_arr, d_mean, color="tab:blue", linewidth=2, label="Direct (mean)")
    ax.fill_between(t_arr, d_mean - d_std, d_mean + d_std, color="tab:blue", alpha=0.2)
    ax.plot(t_arr, t_mean, color="tab:red", linewidth=2, label="Tree (mean)")
    ax.fill_between(t_arr, t_mean - t_std, t_mean + t_std, color="tab:red", alpha=0.2)

    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Число капель N(t)")
    ax.set_title(f"Эволюция числа капель ({num_runs} реализаций)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "droplet_count.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Сохранён: {path}")


def plot_median_radius(direct, tree):
    """График медианного радиуса vs t."""
    t_arr = direct["snapshot_times"]
    num_runs = direct["num_runs"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(num_runs):
        ax.plot(t_arr, direct["median_radii"][i] * 1e6, color="tab:blue", alpha=0.1, linewidth=0.5)
        ax.plot(t_arr, tree["median_radii"][i] * 1e6, color="tab:red", alpha=0.1, linewidth=0.5)

    d_mean = direct["median_radii"].mean(axis=0) * 1e6
    d_std = direct["median_radii"].std(axis=0) * 1e6
    t_mean = tree["median_radii"].mean(axis=0) * 1e6
    t_std = tree["median_radii"].std(axis=0) * 1e6

    ax.plot(t_arr, d_mean, color="tab:blue", linewidth=2, label="Direct (mean)")
    ax.fill_between(t_arr, d_mean - d_std, d_mean + d_std, color="tab:blue", alpha=0.2)
    ax.plot(t_arr, t_mean, color="tab:red", linewidth=2, label="Tree (mean)")
    ax.fill_between(t_arr, t_mean - t_std, t_mean + t_std, color="tab:red", alpha=0.2)

    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Медианный радиус, мкм")
    ax.set_title(f"Эволюция медианного радиуса ({num_runs} реализаций)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "median_radius.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Сохранён: {path}")


def plot_timing(direct, tree):
    """Boxplot времени расчёта."""
    direct_times = direct["elapsed_times"]
    tree_times = tree["elapsed_times"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [direct_times, tree_times],
        tick_labels=["Direct", "Tree"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("tab:blue")
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("tab:red")
    bp["boxes"][1].set_alpha(0.5)

    ax.set_ylabel("Время расчёта, сек")
    ax.set_title("Время расчёта: Direct vs Tree")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "timing.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Сохранён: {path}")

    print(f"\nDirect: {direct_times.mean():.1f} ± {direct_times.std():.1f} сек  "
          f"(min={direct_times.min():.1f}, max={direct_times.max():.1f})")
    print(f"Tree:   {tree_times.mean():.1f} ± {tree_times.std():.1f} сек  "
          f"(min={tree_times.min():.1f}, max={tree_times.max():.1f})")
    if tree_times.mean() > 0:
        speedup = direct_times.mean() / tree_times.mean()
        print(f"Ускорение tree/direct: {speedup:.2f}x")


def _compute_averaged_histograms(radii_list, bins):
    """Усреднённые гистограммы по реализациям."""
    hists = np.array([
        np.histogram(r, bins=bins, density=True)[0] for r in radii_list
    ])
    return hists.mean(axis=0), hists.std(axis=0)


def plot_radii_distribution(direct, tree):
    """Сводный 2x3 subplots распределения радиусов."""
    histogram_times = direct["histogram_times"]
    num_runs = direct["num_runs"]

    n = len(histogram_times)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, t in enumerate(histogram_times):
        ax = axes_flat[idx]

        direct_radii = [r * 1e6 for r in direct["radii_by_time"][t]]
        tree_radii = [r * 1e6 for r in tree["radii_by_time"][t]]

        all_radii = np.concatenate(direct_radii + tree_radii)
        bins = np.linspace(all_radii.min(), all_radii.max(), 31)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        d_mean, d_std = _compute_averaged_histograms(direct_radii, bins)
        t_mean, t_std = _compute_averaged_histograms(tree_radii, bins)

        ax.plot(bin_centers, d_mean, color="tab:blue", linewidth=1.5, label="Direct")
        ax.fill_between(bin_centers, d_mean - d_std, d_mean + d_std, color="tab:blue", alpha=0.2)
        ax.plot(bin_centers, t_mean, color="tab:red", linewidth=1.5, label="Tree")
        ax.fill_between(bin_centers, t_mean - t_std, t_mean + t_std, color="tab:red", alpha=0.2)

        ax.set_title(f"t = {t} сек")
        ax.set_xlabel("Радиус, мкм")
        ax.set_ylabel("Плотность")
        if idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Скрываем лишние субплоты, если histogram_times < 6
    for idx in range(len(histogram_times), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Распределение по радиусам ({num_runs} реализаций)", fontsize=14)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "radii_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Сохранён: {path}")


def plot_radii_histograms_individual(direct, tree):
    """Отдельные гистограммы для каждого histogram_time."""
    histogram_times = direct["histogram_times"]
    num_runs = direct["num_runs"]

    for t in histogram_times:
        direct_radii = [r * 1e6 for r in direct["radii_by_time"][t]]
        tree_radii = [r * 1e6 for r in tree["radii_by_time"][t]]

        all_radii = np.concatenate(direct_radii + tree_radii)
        bins = np.linspace(all_radii.min(), all_radii.max(), 31)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]

        d_mean, d_std = _compute_averaged_histograms(direct_radii, bins)
        t_mean, t_std = _compute_averaged_histograms(tree_radii, bins)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(bin_centers - bin_width * 0.2, d_mean, width=bin_width * 0.4,
               color="tab:blue", alpha=0.7, label="Direct (mean)")
        ax.errorbar(bin_centers - bin_width * 0.2, d_mean, yerr=d_std,
                     fmt="none", ecolor="tab:blue", alpha=0.5, capsize=2)

        ax.bar(bin_centers + bin_width * 0.2, t_mean, width=bin_width * 0.4,
               color="tab:red", alpha=0.7, label="Tree (mean)")
        ax.errorbar(bin_centers + bin_width * 0.2, t_mean, yerr=t_std,
                     fmt="none", ecolor="tab:red", alpha=0.5, capsize=2)

        ax.set_xlabel("Радиус, мкм")
        ax.set_ylabel("Плотность")
        ax.set_title(f"Гистограмма радиусов, t = {t} сек ({num_runs} реализаций)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        path = os.path.join(PLOTS_DIR, f"radii_histogram_t{t}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Сохранён: {path}")


def plot_radii_kde_individual(direct, tree):
    """Отдельные KDE-графики для каждого histogram_time."""
    histogram_times = direct["histogram_times"]
    num_runs = direct["num_runs"]

    for t in histogram_times:
        direct_radii = [r * 1e6 for r in direct["radii_by_time"][t]]
        tree_radii = [r * 1e6 for r in tree["radii_by_time"][t]]

        # Объединённые данные всех реализаций для KDE
        direct_all = np.concatenate(direct_radii)
        tree_all = np.concatenate(tree_radii)
        all_radii = np.concatenate([direct_all, tree_all])

        x_grid = np.linspace(all_radii.min() * 0.95, all_radii.max() * 1.05, 200)

        fig, ax = plt.subplots(figsize=(8, 5))

        # KDE по объединённым данным
        kde_direct = gaussian_kde(direct_all)
        kde_tree = gaussian_kde(tree_all)

        y_direct = kde_direct(x_grid)
        y_tree = kde_tree(x_grid)

        ax.plot(x_grid, y_direct, color="tab:blue", linewidth=2, label="Direct")
        ax.plot(x_grid, y_tree, color="tab:red", linewidth=2, label="Tree")

        # Полоса ±std по отдельным реализациям (KDE каждой реализации)
        if num_runs > 1:
            direct_kdes = np.array([gaussian_kde(r)(x_grid) for r in direct_radii])
            tree_kdes = np.array([gaussian_kde(r)(x_grid) for r in tree_radii])

            d_mean_kde = direct_kdes.mean(axis=0)
            d_std_kde = direct_kdes.std(axis=0)
            t_mean_kde = tree_kdes.mean(axis=0)
            t_std_kde = tree_kdes.std(axis=0)

            ax.fill_between(x_grid, d_mean_kde - d_std_kde, d_mean_kde + d_std_kde,
                            color="tab:blue", alpha=0.15)
            ax.fill_between(x_grid, t_mean_kde - t_std_kde, t_mean_kde + t_std_kde,
                            color="tab:red", alpha=0.15)

        ax.set_xlabel("Радиус, мкм")
        ax.set_ylabel("Плотность")
        ax.set_title(f"KDE распределения радиусов, t = {t} сек ({num_runs} реализаций)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = os.path.join(PLOTS_DIR, f"radii_kde_t{t}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Сохранён: {path}")


def main():
    print("Загрузка данных...")
    direct = load_data("statistics_direct.npz")
    tree = load_data("statistics_tree.npz")

    print(f"Direct: {direct['num_runs']} реализаций, "
          f"histogram_times={direct['histogram_times']}")
    print(f"Tree:   {tree['num_runs']} реализаций, "
          f"histogram_times={tree['histogram_times']}")
    print()

    plot_droplet_count(direct, tree)
    plot_median_radius(direct, tree)
    plot_timing(direct, tree)
    plot_radii_distribution(direct, tree)
    plot_radii_histograms_individual(direct, tree)
    plot_radii_kde_individual(direct, tree)

    print("\nВсе графики построены!")


if __name__ == "__main__":
    main()
