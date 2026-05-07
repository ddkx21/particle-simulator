# Particle Simulator — DEM + PBM для коалесценции заряженных капель

[![CI](https://github.com/ddkx21/particle-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/ddkx21/particle-simulator/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checker: mypy](https://img.shields.io/badge/type%20checker-mypy-blue.svg)](https://mypy-lang.org/)

> Гибридный симулятор движения и коалесценции водяных капель в масле под
> внешним электрическим полем. Сочетает **DEM** (Discrete Element Method) для
> трекинга отдельных капель и **PBM** (Population Balance Model) для
> распределения по объёмам, с поддержкой периодических граничных условий.

<p align="center">
  <img src=".github/assets/gifs/evolution.gif"
       alt="Эволюция системы капель: коалесценция под действием электрического поля"
       width="640">
</p>

<p align="center">
  <em>Эволюция ансамбля капель в масле под действием внешнего электрического
  поля.</em>
</p>

---

## Содержание

- [Возможности](#возможности)
- [Физическая модель](#физическая-модель)
- [Архитектура](#архитектура)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Точки входа](#точки-входа)
- [Тестирование и качество кода](#тестирование-и-качество-кода)
- [Структура репозитория](#структура-репозитория)
- [Документация](#документация)
- [Лицензия](#лицензия)

## Возможности

- **DEM-ядро** на [Taichi](https://www.taichi-lang.org/) — параллельный CPU/GPU
  расчёт сил и интегрирование Эйлера.
- Два метода вычисления сил:
  - **Direct** O(N²) — точное попарное суммирование.
  - **Barnes-Hut** O(N log N) на плоском октодереве с многопоточным
    послойным построением (level-by-level).
- **Spatial-hash** обнаружение столкновений за O(N).
- **Периодические граничные условия** (MIC + псевдо-периодическая поправка
  поля скоростей из COMSOL).
- **PBM** с двумя методами перераспределения: *fixed-pivot* (Kumar–Ramkrishna)
  и *cell-average* (Kumar 2006).
- **DEM ↔ PBM coupling** — извлечение ядра столкновений из DEM, параллельный
  расчёт.
- Бенчмарки, статистика, сравнения.

---

## Физическая модель

Водяные капли в масле под действием внешнего электрического поля **E**
приобретают индуцированный дипольный момент ∝ R³. Капли взаимодействуют
через:

1. **Диполь-дипольные силы** (электростатика, анизотропны вдоль E):

   ```
   F_ij = m_const · (rᵢ³ rⱼ³ / |rᵢⱼ|⁷) · g(r̂ᵢⱼ),
   m_const = 12π · ε₀ · ε_oil · E²
   ```

2. **Конвективный перенос** через тензор Озеена (стоксовский поток).

Полное описание формул, угловой зависимости и численных схем —
[ALGORITHM.md](ALGORITHM.md).

---
## Установка

### Требования

- **Python 3.12** (строго; CI собирается только на 3.12).
- Linux/macOS/Windows.
- Опционально: CUDA-совместимая видеокарта для Taichi GPU-бэкенда.

### Через pip

```bash
git clone https://github.com/ddkx21/particle-simulator.git
cd particle-simulator

python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e ".[dev]"             # runtime + dev (pytest, ruff, mypy, black)
```

### Через requirements.txt 

```bash
pip install -r requirements.txt                        # runtime
pip install -r requirements.txt -r requirements-dev.txt # + dev tooling
```

---

## Быстрый старт

```bash
# 1) Сгенерировать начальное состояние и запустить direct DEM
python main.py

# 2) То же, но с октодеревом (быстрее при N ≳ 5000)
python main_tree.py

# 3) Standalone PBM (без DEM)
python main_pbm.py

# 4) Связанный DEM + PBM
python main_coupled.py

# 5) Сравнение PBM vs DEM
python compare_pbm_dem.py
```

Результаты складываются в `results/` (см. `.gitignore` — содержимое
не трекается, только `.gitkeep`).

---

## Точки входа

| Скрипт                  | Назначение                                                           |
|-------------------------|----------------------------------------------------------------------|
| `main.py`               | Direct DEM (O(N²)) с периодикой и COMSOL-поправкой                  |
| `main_tree.py`          | Tree DEM (Barnes-Hut O(N log N))                                    |
| `main_multi.py`         | Запуск нескольких независимых ансамблей                             |
| `main_pbm.py`           | Standalone PBM с аналитическим/файловым ядром                       |
| `main_coupled.py`       | DEM + PBM, обмен столкновениями каждые `coupling_dt`                |
| `main_pbm.py`           | Только PBM с предвычисленным ядром                                  |
| `compare_pbm_dem.py`    | Сравнительные графики результатов DEM и PBM                         |

---

## Тестирование и качество кода

Проект использует следующий tooling (все включены в `pip install -e ".[dev]"`):

| Инструмент | Назначение                            | Конфиг           |
|------------|---------------------------------------|------------------|
| `pytest`   | unit/integration-тесты + coverage     | `pyproject.toml` |
| `ruff`     | линтер (E/F/W/I/B/UP правила)         | `pyproject.toml` |
| `black`    | автоформаттер (line-length 100)       | `pyproject.toml` |
| `mypy`     | статическая проверка типов (warn-only)| `pyproject.toml` |

### Локальный прогон

```bash
# Все тесты
pytest tests/

# С покрытием
pytest tests/ --cov=dem --cov=pbm --cov-report=term-missing

# Проверка линтером и форматом (read-only)
ruff check dem/ pbm/ tests/
black --check dem/ pbm/ tests/
mypy dem/ pbm/

# Авто-исправление линт-ошибок и форматирование (перед коммитом)
ruff check --fix dem/ pbm/ tests/
black dem/ pbm/ tests/
```

### CI

GitHub Actions (`.github/workflows/ci.yml`) на каждый push/PR в `master`/`main`
прогоняет три job'а:

1. **lint** — ruff + black --check + mypy (warn-only).
2. **test** — pytest с покрытием, артефакты coverage.xml + junit.
3. **build** — `python -m build` для sdist + wheel (зависит от lint и test).

Подробнее по каждой группе тестов — [`tests/README.md`](tests/README.md).

---

## Структура репозитория

```
particle_simulator_git/
├── dem/                         # DEM-ядро (см. dem/README.md)
│   ├── collision_detector/      # spatial hash детектор столкновений
│   ├── force_calculator/        # direct O(N²) расчёт сил
│   ├── octree/                  # flat Barnes-Hut octree
│   ├── particle_generator/      # генерация начальных условий
│   ├── particle_state/          # снапшот состояния системы
│   ├── periodic_correction/     # COMSOL lattice correction
│   ├── post_processor/          # визуализация
│   ├── solution/                # хранение траекторий
│   ├── solver/                  # Эйлер-интегратор
│   ├── statistics/              # сбор метрик direct vs tree
│   └── utils/                   # вспомогательные функции
├── pbm/                         # Population Balance Model (см. pbm/README.md)
│   ├── kernels/                 # ядра столкновений
│   ├── collision_frequency/     # подсчёт частот коллизий
│   ├── pbm_solver.py
│   ├── volume_grid.py
│   ├── redistribution.py
│   └── coupling.py              # DEM ↔ PBM интерфейс
├── benchmarks/                  # производительность (см. benchmarks/README.md)
├── convergence_test/            # тест сходимости периодики
├── dual_ensemble/               # запуск двух систем для сравнения
├── studies/                     # одиночные исследования (Эйлер, анимации)
├── tests/                       # pytest-сьюты
├── docs/                        # дополнительная документация
├── results/                     # результаты симуляций (gitignored)
├── main*.py                     # точки входа
├── ALGORITHM.md                 # подробный алгоритм и формулы
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── .github/
    ├── assets/                  # картинки и GIF для README
    │   ├── gifs/
    │   └── images/
    └── workflows/ci.yml
```

---

## Документация

- [ALGORITHM.md](ALGORITHM.md) — полные формулы и численные схемы.
- [dem/README.md](dem/README.md) — устройство DEM-ядра.
- [pbm/README.md](pbm/README.md) — Population Balance Model.
- [CONTRIBUTING.md](CONTRIBUTING.md) — как вносить вклад.
- README в каждом подмодуле описывают локальную ответственность.

---

## Автор

**Dadakhodjaev Rustam Bakhtiyorovich**

- Email: <rustam.dadakhodjaev@gmail.com>
- Учебная почта (СПбГУ): <st094266@student.spbu.ru>
- Telegram: [@ddkx21](https://t.me/ddkx21)
- GitHub: [ddkx21](https://github.com/ddkx21)

По любым вопросам, баг-репортам и предложениям — пишите на любой из
указанных контактов или открывайте
[issue в репозитории](https://github.com/ddkx21/particle-simulator/issues).

---

## Лицензия

Распространяется по лицензии **MIT** — см. [LICENSE](LICENSE).

Copyright © 2026 Dadakhodjaev Rustam Bakhtiyorovich
