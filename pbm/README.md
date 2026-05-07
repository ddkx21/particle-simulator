# PBM — Population Balance Model

Модуль для решения уравнений популяционного баланса (агрегация капель)
в связке с DEM-симулятором электростатической коалесценции.

## Что делает PBM

PBM отслеживает **распределение капель по объёмам** N(v, t) — сколько капель
каждого размера существует в каждый момент времени. Вместо моделирования
каждой капли по отдельности (как DEM), PBM решает систему ОДУ:

```
dN[i]/dt = Birth[i] - Death[i]
```

- **Birth** — появление новых капель в бине i за счёт слияния более мелких
- **Death** — исчезновение капель из бина i, когда они сливаются с другими

Скорость слияния определяется **ядром столкновений** Q(v1, v2).

---

## Структура модуля

```
pbm/
  __init__.py
  volume_grid.py              # Сетка по объёмам (бинирование)
  redistribution.py           # Методы перераспределения (fixed-pivot, cell-average)
  pbm_solver.py               # ODE-солвер PBM
  coupling.py                 # Связка DEM → PBM
  kernels/
    analytical_kernel.py      # Аналитическое ядро Q(v1,v2) из электростатики
    dem_extracted_kernel.py   # Извлечение ядра из DEM-данных
  collision_frequency/
    direct_collision_freq.py  # O(N^2) подсчёт коллизий (brute-force)
    tree_collision_freq.py    # O(N log N) подсчёт через октодерево
```

---

## Быстрый старт

### 1. Standalone PBM (без DEM)

Самый простой сценарий — решить PBM с аналитическим ядром:

```python
import numpy as np
from pbm import VolumeGrid, PBMSolver
from pbm.kernels import AnalyticalElectrostaticKernel

# 1. Сетка по объёмам (из диапазона радиусов)
grid = VolumeGrid.from_radii_range(
    r_min=2.5e-6,   # м
    r_max=7.5e-6,   # м
    n_bins=50,
    spacing="geometric",
)

# 2. Начальное распределение
radii = np.random.uniform(2.5e-6, 7.5e-6, 100_000)
N0 = grid.histogram(radii)  # конвертирует радиусы → объёмы → гистограмма

# 3. Ядро столкновений
kernel = AnalyticalElectrostaticKernel(
    eps0=8.85e-12,    # электрическая постоянная, Ф/м
    eps_oil=2.85,     # относительная диэлектрическая проницаемость масла
    E=3e5,            # напряжённость электрического поля, В/м
    eta_oil=0.065,    # вязкость масла, Па·с
)
Q = kernel.build_matrix(grid)  # матрица (n_bins × n_bins)

# 4. Солвер
solver = PBMSolver(
    grid, Q,
    method="cell_average",  # или "fixed_pivot"
    integrator="BDF",       # жёсткий метод (рекомендуется)
    scale_factor=len(radii),
)

# 5. Решение
result = solver.solve(
    N0,
    t_span=(0, 50),
    t_eval=np.linspace(0, 50, 501),
    rtol=1e-5,
    atol=1e-5,
)

# result содержит:
#   "t"            — массив времён
#   "N"            — массив (n_times, n_bins), распределение в каждый момент
#   "total_count"  — суммарное число частиц N(t)
#   "total_volume" — суммарный объём V(t)
```

Готовый скрипт: `main_pbm.py`.

### 2. DEM + PBM (связанный запуск)

PBM работает параллельно с DEM-симуляцией, получая данные о столкновениях:

```python
from pbm import VolumeGrid, PBMSolver
from pbm.kernels import AnalyticalElectrostaticKernel
from pbm.coupling import DEMPBMCoupling

# Настройка PBM (как выше)
grid = VolumeGrid.from_radii_range(2.5e-6, 22.5e-6, 50, "geometric")
kernel = AnalyticalElectrostaticKernel(eps0, eps_oil, E, eta_oil)
Q = kernel.build_matrix(grid)
pbm_solver = PBMSolver(grid, Q, method="cell_average", scale_factor=num_particles)

# Создание связки
coupling = DEMPBMCoupling(
    grid=grid,
    pbm_solver=pbm_solver,
    domain_volume=box_size**3,
    coupling_interval=1.0,  # синхронизация каждую 1 секунду симуляционного времени
)
coupling.initialize_from_dem(initial_radii)

# Передаём coupling в солвер
solver = EulerDropletSolver(
    force_calculator=force_calculator,
    solution=solution,
    post_processor=post_processor,
    collision_detector=collision_detector,
    pbm_coupling=coupling,              # ← вот здесь
)
solver.solve(dt, t_stop)

# После симуляции — сравнение DEM и PBM
centers, pbm_N = coupling.get_pbm_distribution()
dem_kernel = coupling.get_dem_kernel()  # извлечённое ядро из DEM

# История: coupling.history_t, coupling.history_dem_N, coupling.history_pbm_N
```

Готовый скрипт: `main_coupled.py`.

### 3. Tree-ускоренный подсчёт коллизий

Подсчёт коллизий между объёмными классами через октодерево вместо O(N^2):

```python
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

from pbm import VolumeGrid
from pbm.collision_frequency import TreeCollisionFrequency, DirectCollisionFrequency

grid = VolumeGrid.from_radii_range(2.5e-6, 7.5e-6, 20, "geometric")

# Дерево уже построено в force_calculator
octree = force_calculator.octree
octree.build(positions, radii, L, periodic)

# Tree-ускоренный подсчёт
tree_cf = TreeCollisionFrequency(octree, grid, max_particles=len(radii))
collision_matrix = tree_cf.compute(positions, radii)
# collision_matrix[i, j] = число коллизий между бинами i и j

# Для сравнения/валидации — brute-force
direct_cf = DirectCollisionFrequency(grid)
collision_matrix_ref = direct_cf.compute(positions, radii, L=L, periodic=periodic)
```

---

## Описание компонентов

### VolumeGrid — сетка по объёмам

Дискретизирует пространство объёмов капель на бины.

| Параметр  | Описание |
|-----------|----------|
| `v_min`   | Минимальный объём (м^3) |
| `v_max`   | Максимальный объём (м^3). Должен быть достаточно большим для агрегатов |
| `n_bins`  | Число бинов (обычно 30–50) |
| `spacing` | `"geometric"` (рекомендуется), `"logarithmic"` или `"linear"` |

```python
# Из объёмов напрямую
grid = VolumeGrid(v_min=1e-16, v_max=1e-13, n_bins=50)

# Из диапазона радиусов (удобнее)
grid = VolumeGrid.from_radii_range(r_min=2.5e-6, r_max=20e-6, n_bins=50)

# Атрибуты
grid.edges    # (n_bins+1,) — границы бинов
grid.centers  # (n_bins,) — представительные объёмы
grid.widths   # (n_bins,) — ширины бинов

# Методы
grid.histogram(radii)       # радиусы → гистограмма по объёмным бинам
grid.bin_index(volume)      # объём → индекс бина
grid.bin_indices(volumes)   # векторизованная версия
```

**Важно:** `v_max` должен быть в 3–10 раз больше максимального начального объёма,
чтобы вместить агрегаты. Если агрегат вылетает за `v_max`, он теряется.

### AnalyticalElectrostaticKernel — аналитическое ядро

Формула для электростатической коалесценции капель:

```
Q(v1, v2) = (4/√3) * (ε₀ ε E²) / η * (v1^(2/3) * v2^(2/3)) / (v1^(1/3) + v2^(1/3))
```

| Параметр   | Значение по умолчанию | Описание |
|------------|----------------------|----------|
| `eps0`     | 8.85e-12 Ф/м        | Электрическая постоянная |
| `eps_oil`  | 2.85                 | Диэлектрическая проницаемость масла |
| `E`        | 3e5 В/м              | Напряжённость электрического поля |
| `eta_oil`  | 0.065 Па·с           | Динамическая вязкость масла |

```python
kernel = AnalyticalElectrostaticKernel()

# Значение для пары объёмов
q = kernel.evaluate(v1, v2)

# Матрица для всех пар центров сетки
Q = kernel.build_matrix(grid)  # (n_bins, n_bins), симметричная
```

### DEMExtractedKernel — ядро из DEM

Извлекает ядро из статистики DEM-столкновений:

```
K[i,j] = collision_count[i,j] / (T * n_avg[i] * n_avg[j] * dV[i] * dV[j])
```

```python
from pbm.kernels import DEMExtractedKernel

# Из DEM-симуляции (используется через coupling)
dem_kernel = DEMExtractedKernel(grid, domain_volume=box_size**3)
dem_kernel.record_collision(r_i, r_j)       # при каждом столкновении
dem_kernel.update_concentrations(radii, dt)  # на каждом шаге
K = dem_kernel.finalize()                    # → матрица (n_bins, n_bins)

# Из файла (формат .npz с ключами 'volumes' и 'freq')
K = DEMExtractedKernel.load_from_file("fr_err.npz", grid)
```

### PBMSolver — солвер

| Параметр        | Описание |
|-----------------|----------|
| `grid`          | VolumeGrid |
| `kernel_matrix` | Q (n_bins × n_bins), предвычисленное ядро |
| `method`        | `"cell_average"` (рекомендуется) или `"fixed_pivot"` |
| `integrator`    | Метод scipy: `"BDF"` (жёсткий, рекомендуется), `"RK45"`, `"Radau"` |
| `scale_factor`  | Множитель правой части ОДУ (обычно = total_particles) |

**Cell-Average vs Fixed-Pivot:**
- `cell_average` — лучше сохраняет число и объём, точнее для грубых сеток
- `fixed_pivot` — проще, быстрее предвычисление, подходит для мелких сеток

```python
solver = PBMSolver(grid, Q, method="cell_average", integrator="BDF", scale_factor=N_total)
result = solver.solve(N0, t_span=(0, 50), t_eval=t_eval)

# Обновление ядра (например, из DEM)
solver.update_kernel(K_new)
```

### DEMPBMCoupling — связка DEM → PBM

Односторонняя связка: PBM получает данные из DEM, но не влияет на DEM.

| Параметр            | Описание |
|---------------------|----------|
| `grid`              | VolumeGrid |
| `pbm_solver`        | PBMSolver |
| `domain_volume`     | Объём расчётной области (м^3) |
| `coupling_interval` | Интервал синхронизации (секунды симуляции) |

**Что происходит при связке:**
1. Каждый DEM-шаг: `coupling.step(t, radii, dt)` обновляет концентрации
2. При столкновении: `coupling.on_collision(pairs, radii)` регистрирует событие
3. Каждые `coupling_interval` секунд: PBM продвигается на этот интервал
4. История сохраняется в `coupling.history_t/dem_N/pbm_N` для анализа

### TreeCollisionFrequency — ускоренный подсчёт коллизий

Использует октодерево для подсчёта коллизий за ~O(N log N) вместо O(N^2).

**Принцип работы:**
- Для каждой частицы i обходит дерево сверху вниз
- Если минимальное расстояние до bounding box узла > r_i + max_radius[node] →
  весь узел пропускается (коллизий точно нет)
- Иначе: спускается к детям (внутренний узел) или проверяет пары (лист)

**Требования:**
- Октодерево должно быть уже построено (`octree.build(...)`)
- Taichi инициализирован до создания `TreeCollisionFrequency`

---

## Типичные сценарии использования

### Предсказание эволюции распределения

Есть начальное распределение капель → предсказать, как оно изменится за время T:

```python
grid = VolumeGrid.from_radii_range(r_min, r_max * 5, n_bins=50)
kernel = AnalyticalElectrostaticKernel(eps0, eps_oil, E, eta_oil)
Q = kernel.build_matrix(grid)
solver = PBMSolver(grid, Q, scale_factor=total_particles)
result = solver.solve(N0, (0, T))
```

### Валидация DEM аналитикой

Запустить DEM и PBM параллельно, сравнить распределения:

```python
coupling = DEMPBMCoupling(grid, pbm_solver, domain_volume)
# ... после симуляции
for t, dem, pbm in zip(coupling.history_t, coupling.history_dem_N, coupling.history_pbm_N):
    plt.plot(grid.centers, dem, label="DEM")
    plt.plot(grid.centers, pbm, label="PBM")
```

### Извлечение ядра из DEM

Запустить DEM, собрать статистику коллизий, получить ядро K(v1, v2):

```python
coupling = DEMPBMCoupling(grid, pbm_solver, domain_volume)
# ... после симуляции
K_dem = coupling.get_dem_kernel()  # (n_bins, n_bins)
# Сравнить с аналитическим:
K_analytical = kernel.build_matrix(grid)
```

### Использование предвычисленного ядра

Загрузить ядро из файла (например, усреднённое по нескольким DEM-запускам):

```python
from pbm.kernels import DEMExtractedKernel
K = DEMExtractedKernel.load_from_file("kernel_data.npz", grid)
solver = PBMSolver(grid, K, method="cell_average", scale_factor=N_total)
```

---

## Физический смысл параметров

### scale_factor

В стандартном уравнении Смолуховского, если N — абсолютные числа частиц,
а Q — ядро с размерностью м^3/с, то `dN/dt = (Birth - Death) * scale_factor`
где `scale_factor` зависит от нормировки. В референсном коде (`pbm_CA.py`)
используется `scale_factor = total_particles`.

### v_max (верхняя граница сетки)

Агрегаты, объём которых превышает `v_max`, выпадают из расчёта.
Рекомендуется `v_max >= 3 * v_max_initial` для коротких симуляций
и `v_max >= 10 * v_max_initial` для длинных.

### coupling_interval

Интервал синхронизации DEM и PBM. Компромисс между:
- **Малый интервал** (0.1 с): частая синхронизация, больше overhead
- **Большой интервал** (5 с): меньше overhead, но PBM может отстать от DEM

Рекомендуется: 1–2 секунды симуляционного времени (25–50 DEM-шагов при dt=0.04).

---

## Entry points

| Скрипт           | Описание |
|------------------|----------|
| `main_pbm.py`    | Standalone PBM с аналитическим ядром |
| `main_coupled.py`| DEM + PBM с октодеревом и связкой |

Запуск:

```bash
# Standalone PBM (не требует Taichi)
python main_pbm.py

# DEM + PBM (требует Taichi и начальные данные)
.venv/bin/python main_coupled.py
```

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../LICENSE) — см. корневой [README](../README.md).
