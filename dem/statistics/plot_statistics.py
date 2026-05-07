"""
Построение графиков статистики: direct и/или tree.

Читает results/statistics_direct.npz и/или results/statistics_tree.npz,
строит графики в plots/.

Запуск:
  python statistics/plot_statistics.py                # авто-определение
  python statistics/plot_statistics.py --direct        # только direct
  python statistics/plot_statistics.py --tree           # только tree
  python statistics/plot_statistics.py --direct --tree  # сравнение
"""

import argparse
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

# Цвета для каждого метода
METHOD_COLORS = {"Direct": "tab:blue", "Tree": "tab:red"}


def load_data(filename: str) -> dict:
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


def _plot_path(base_name: str, datasets: list[tuple[str, dict, str]]) -> str:
    """Путь к файлу графика: с суффиксом метода для одиночного режима."""
    if len(datasets) == 1:
        return os.path.join(PLOTS_DIR, f"{base_name}_{datasets[0][0].lower()}.png")
    return os.path.join(PLOTS_DIR, f"{base_name}.png")


def _total_runs(datasets: list[tuple[str, dict, str]]) -> int:
    return sum(d["num_runs"] for _, d, _ in datasets)


def plot_droplet_count(datasets: list[tuple[str, dict, str]]) -> None:
    """График N(t) с mean +/- std и полупрозрачными реализациями."""
    t_arr = datasets[0][1]["snapshot_times"]
    total_runs = _total_runs(datasets)

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data, color in datasets:
        for i in range(data["num_runs"]):
            ax.plot(t_arr, data["droplet_counts"][i],
                    color=color, alpha=0.1, linewidth=0.5)

        mean = data["droplet_counts"].mean(axis=0)
        std = data["droplet_counts"].std(axis=0)

        ax.plot(t_arr, mean, color=color, linewidth=2, label=f"{label} (mean)")
        ax.fill_between(t_arr, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Число капель N(t)")
    ax.set_title(f"Эволюция числа капель ({total_runs} реализаций)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = _plot_path("droplet_count", datasets)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Сохранён: {path}")


def plot_median_radius(datasets: list[tuple[str, dict, str]]) -> None:
    """График медианного радиуса vs t."""
    t_arr = datasets[0][1]["snapshot_times"]
    total_runs = _total_runs(datasets)

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data, color in datasets:
        for i in range(data["num_runs"]):
            ax.plot(t_arr, data["median_radii"][i] * 1e6,
                    color=color, alpha=0.1, linewidth=0.5)

        mean = data["median_radii"].mean(axis=0) * 1e6
        std = data["median_radii"].std(axis=0) * 1e6

        ax.plot(t_arr, mean, color=color, linewidth=2, label=f"{label} (mean)")
        ax.fill_between(t_arr, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Медианный радиус, мкм")
    ax.set_title(f"Эволюция медианного радиуса ({total_runs} реализаций)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = _plot_path("median_radius", datasets)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Сохранён: {path}")


def plot_timing(datasets: list[tuple[str, dict, str]]) -> None:
    """Boxplot времени расчёта."""
    labels = [label for label, _, _ in datasets]
    times_list = [data["elapsed_times"] for _, data, _ in datasets]
    colors = [color for _, _, color in datasets]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(times_list, tick_labels=labels, patch_artist=True)

    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.5)

    title = "Время расчёта: " + " vs ".join(labels)
    ax.set_ylabel("Время расчёта, сек")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = _plot_path("timing", datasets)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Сохранён: {path}")

    for label, data, _ in datasets:
        t = data["elapsed_times"]
        print(f"\n{label}: {t.mean():.1f} ± {t.std():.1f} сек  "
              f"(min={t.min():.1f}, max={t.max():.1f})")

    if len(datasets) == 2:
        t0 = datasets[0][1]["elapsed_times"].mean()
        t1 = datasets[1][1]["elapsed_times"].mean()
        if t1 > 0:
            print(f"Ускорение {datasets[1][0]}/{datasets[0][0]}: {t0 / t1:.2f}x")


def _compute_averaged_histograms(
    radii_list: list[np.ndarray], bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Усреднённые гистограммы по реализациям."""
    hists = np.array([
        np.histogram(r, bins=bins, density=True)[0] for r in radii_list
    ])
    return hists.mean(axis=0), hists.std(axis=0)


def plot_radii_distribution(datasets: list[tuple[str, dict, str]]) -> None:
    """Сводный 2x3 subplots распределения радиусов."""
    histogram_times = datasets[0][1]["histogram_times"]
    total_runs = _total_runs(datasets)

    n = len(histogram_times)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, t in enumerate(histogram_times):
        ax = axes_flat[idx]

        # Собираем все радиусы для определения общих bins
        all_radii_parts = []
        for _, data, _ in datasets:
            all_radii_parts.extend([r * 1e6 for r in data["radii_by_time"][t]])

        all_radii = np.concatenate(all_radii_parts)
        bins = np.linspace(all_radii.min(), all_radii.max(), 31)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        for label, data, color in datasets:
            radii = [r * 1e6 for r in data["radii_by_time"][t]]
            mean, std = _compute_averaged_histograms(radii, bins)
            ax.plot(bin_centers, mean, color=color, linewidth=1.5, label=label)
            ax.fill_between(bin_centers, mean - std, mean + std,
                            color=color, alpha=0.2)

        ax.set_title(f"t = {t} сек")
        ax.set_xlabel("Радиус, мкм")
        ax.set_ylabel("Плотность")
        if idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(histogram_times), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Распределение по радиусам ({total_runs} реализаций)", fontsize=14)
    fig.tight_layout()

    path = _plot_path("radii_distribution", datasets)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Сохранён: {path}")


def plot_radii_histograms_individual(
    datasets: list[tuple[str, dict, str]]
) -> None:
    """Отдельные гистограммы для каждого histogram_time."""
    histogram_times = datasets[0][1]["histogram_times"]
    total_runs = _total_runs(datasets)
    n_datasets = len(datasets)

    for t in histogram_times:
        # Общие bins
        all_radii_parts = []
        for _, data, _ in datasets:
            all_radii_parts.extend([r * 1e6 for r in data["radii_by_time"][t]])

        all_radii = np.concatenate(all_radii_parts)
        bins = np.linspace(all_radii.min(), all_radii.max(), 31)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]

        fig, ax = plt.subplots(figsize=(8, 5))

        if n_datasets == 1:
            label, data, color = datasets[0]
            radii = [r * 1e6 for r in data["radii_by_time"][t]]
            mean, std = _compute_averaged_histograms(radii, bins)
            ax.bar(bin_centers, mean, width=bin_width * 0.8,
                   color=color, alpha=0.7, label=f"{label} (mean)")
            ax.errorbar(bin_centers, mean, yerr=std,
                        fmt="none", ecolor=color, alpha=0.5, capsize=2)
        else:
            for i, (label, data, color) in enumerate(datasets):
                offset = (i - 0.5) * bin_width * 0.4
                radii = [r * 1e6 for r in data["radii_by_time"][t]]
                mean, std = _compute_averaged_histograms(radii, bins)
                ax.bar(bin_centers + offset, mean, width=bin_width * 0.4,
                       color=color, alpha=0.7, label=f"{label} (mean)")
                ax.errorbar(bin_centers + offset, mean, yerr=std,
                            fmt="none", ecolor=color, alpha=0.5, capsize=2)

        ax.set_xlabel("Радиус, мкм")
        ax.set_ylabel("Плотность")
        ax.set_title(f"Гистограмма радиусов, t = {t} сек ({total_runs} реализаций)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        path = _plot_path(f"radii_histogram_t{t}", datasets)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Сохранён: {path}")


def plot_radii_kde_individual(
    datasets: list[tuple[str, dict, str]]
) -> None:
    """Отдельные KDE-графики для каждого histogram_time."""
    histogram_times = datasets[0][1]["histogram_times"]
    total_runs = _total_runs(datasets)

    for t in histogram_times:
        # Общая сетка x
        all_radii_parts = []
        for _, data, _ in datasets:
            all_radii_parts.extend([r * 1e6 for r in data["radii_by_time"][t]])
        all_radii = np.concatenate(all_radii_parts)
        x_grid = np.linspace(all_radii.min() * 0.95, all_radii.max() * 1.05, 200)

        fig, ax = plt.subplots(figsize=(8, 5))

        for label, data, color in datasets:
            radii_um = [r * 1e6 for r in data["radii_by_time"][t]]
            combined = np.concatenate(radii_um)

            kde = gaussian_kde(combined)
            y = kde(x_grid)
            ax.plot(x_grid, y, color=color, linewidth=2, label=label)

            if data["num_runs"] > 1:
                kdes = np.array([gaussian_kde(r)(x_grid) for r in radii_um])
                mean_kde = kdes.mean(axis=0)
                std_kde = kdes.std(axis=0)
                ax.fill_between(x_grid, mean_kde - std_kde, mean_kde + std_kde,
                                color=color, alpha=0.15)

        ax.set_xlabel("Радиус, мкм")
        ax.set_ylabel("Плотность")
        ax.set_title(f"KDE распределения радиусов, t = {t} сек "
                     f"({total_runs} реализаций)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = _plot_path(f"radii_kde_t{t}", datasets)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Сохранён: {path}")


def _build_datasets(
    methods: list[str],
) -> list[tuple[str, dict, str]]:
    """Загрузка данных и формирование списка datasets."""
    filemap = {
        "direct": ("Direct", "statistics_direct.npz"),
        "tree": ("Tree", "statistics_tree.npz"),
    }
    datasets = []
    for method in methods:
        label, filename = filemap[method]
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Файл не найден: {path}\n"
                f"Сначала запустите: python statistics/run_{method}.py"
            )
        data = load_data(filename)
        color = METHOD_COLORS[label]
        datasets.append((label, data, color))
        print(f"{label}: {data['num_runs']} реализаций, "
              f"histogram_times={data['histogram_times']}")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построение графиков статистики симуляций."
    )
    parser.add_argument(
        "--direct", action="store_true",
        help="Включить данные direct-метода",
    )
    parser.add_argument(
        "--tree", action="store_true",
        help="Включить данные tree-метода",
    )
    args = parser.parse_args()

    # Если ни один флаг не указан — автодетект
    if not args.direct and not args.tree:
        methods = []
        for method, filename in [("direct", "statistics_direct.npz"),
                                 ("tree", "statistics_tree.npz")]:
            if os.path.isfile(os.path.join(RESULTS_DIR, filename)):
                methods.append(method)
        if not methods:
            parser.error(
                "Не найдено ни одного файла результатов в "
                f"{RESULTS_DIR}/.\n"
                "Сначала запустите run_direct.py и/или run_tree.py."
            )
    else:
        methods = []
        if args.direct:
            methods.append("direct")
        if args.tree:
            methods.append("tree")

    print("Загрузка данных...")
    datasets = _build_datasets(methods)
    print()

    plot_droplet_count(datasets)
    plot_median_radius(datasets)
    plot_timing(datasets)
    plot_radii_distribution(datasets)
    plot_radii_histograms_individual(datasets)
    plot_radii_kde_individual(datasets)

    print("\nВсе графики построены!")


if __name__ == "__main__":
    main()
