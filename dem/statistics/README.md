# Statistics Pipeline

Сбор статистики и построение графиков для сравнения Direct vs Tree методов расчёта сил.

## Структура

```
statistics/
├── config.json            # Все параметры симуляции (единый конфиг)
├── run_direct.py          # Запуск direct-метода → results/statistics_direct.npz
├── run_tree.py            # Запуск tree-метода  → results/statistics_tree.npz
├── plot_statistics.py     # Построение графиков из .npz файлов
├── results/               # Результаты симуляций (.npz)
│   ├── statistics_direct.npz
│   └── statistics_tree.npz
└── plots/                 # Графики (.png)
    ├── droplet_count.png
    ├── median_radius.png
    ├── timing.png
    ├── radii_distribution.png
    ├── radii_histogram_t{T}.png   (×6)
    └── radii_kde_t{T}.png         (×6)
```

## Быстрый старт

```bash
# 1. Запустить симуляции (можно по отдельности)
python statistics/run_direct.py
python statistics/run_tree.py

# 2. Построить графики
python statistics/plot_statistics.py                # авто-определение (строит по доступным .npz)
python statistics/plot_statistics.py --direct        # только direct-метод
python statistics/plot_statistics.py --tree           # только tree-метод
python statistics/plot_statistics.py --direct --tree  # сравнение обоих методов
```

## Конфигурация

Все параметры задаются в `config.json`:

| Секция                         | Параметр         | Описание                                     |
| ------------------------------ | ---------------- | -------------------------------------------- |
| `simulation.N`                 | 100              | Начальное число капель                       |
| `simulation.dt`                | 0.04             | Шаг по времени                               |
| `simulation.t_stop`            | 100              | Конечное время симуляции (сек)               |
| `simulation.save_interval`     | 250              | Интервал сохранения (шагов)                  |
| `simulation.num_runs`          | 2                | Число реализаций                             |
| `snapshot_times`               | [0,10,...,100]   | Моменты для графиков N(t), median_r(t)       |
| `histogram_times`              | [0,20,...,100]   | Моменты для гистограмм и KDE                 |
| `physics.radii_range`          | [2.5e-6, 7.5e-6] | Диапазон начальных радиусов (м)              |
| `physics.water_volume_content` | 0.02             | Объёмная доля воды                           |
| `physics.eps_oil`              | 2.85             | Диэлектрическая проницаемость масла          |
| `physics.eta_oil`              | 0.065            | Вязкость масла (Па·с)                        |
| `physics.eta_water`            | 0.001            | Вязкость воды (Па·с)                         |
| `physics.rho_water`            | 1000             | Плотность воды (кг/м³)                       |
| `physics.rho_oil`              | 900              | Плотность масла (кг/м³)                      |
| `physics.E`                    | 3e5              | Напряжённость электрического поля (В/м)      |
| `physics.boundary_mode`        | "open"           | Граничные условия: `"open"` или `"periodic"` |
| `tree.theta`                   | 0.25             | Параметр точности tree-метода                |
| `tree.mpl`                     | 16               | Максимальное число частиц в листе octree     |
| `taichi.arch`                  | "cpu"            | Архитектура Taichi (`"cpu"` или `"gpu"`)     |
| `taichi.cpu_max_num_threads`   | 16               | Число потоков CPU                            |

## Графики

### Основные (сравнение direct vs tree)

- **droplet_count.png** — эволюция числа капель N(t): mean ± std, отдельные реализации полупрозрачно
- **median_radius.png** — эволюция медианного радиуса (мкм)
- **timing.png** — boxplot времени расчёта

### Распределения радиусов

- **radii_distribution.png** — сводный 2×3 subplots для всех `histogram_times`
- **radii_histogram_t{T}.png** — отдельная bar-гистограмма для момента t=T (усреднённая по реализациям, errorbar ± std)
- **radii_kde_t{T}.png** — непрерывная KDE-оценка плотности для t=T (gaussian_kde по объединённым данным, заливка ± std по реализациям)

### Одиночный режим

При построении графиков только для одного метода (`--direct` или `--tree`) имена файлов содержат суффикс метода:

- **droplet_count_direct.png** / **droplet_count_tree.png**
- **median_radius_direct.png** / **median_radius_tree.png**
- **timing_direct.png** / **timing_tree.png**
- **radii_distribution_direct.png** / **radii_distribution_tree.png**
- **radii_histogram_t{T}\_direct.png** / **radii_histogram_t{T}\_tree.png**
- **radii_kde_t{T}\_direct.png** / **radii_kde_t{T}\_tree.png**

## Формат .npz

Каждый файл `statistics_{method}.npz` содержит:

| Ключ                   | Shape                           | Описание                                |
| ---------------------- | ------------------------------- | --------------------------------------- |
| `elapsed_times`        | (num_runs,)                     | Время расчёта каждой реализации         |
| `droplet_counts`       | (num_runs, len(snapshot_times)) | Число капель                            |
| `median_radii`         | (num_runs, len(snapshot_times)) | Медианный радиус                        |
| `snapshot_times`       | (len(snapshot_times),)          | Моменты снапшотов                       |
| `radii_at_t{T}_run{i}` | (n_drops,)                      | Радиусы капель в момент T, реализация i |

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
