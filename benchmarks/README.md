# `benchmarks/` — измерение производительности

Сценарии для оценки скорости и точности расчётных методов DEM.

## Бенчмарки

### `test.py` — Direct vs Tree

Сравнивает `DirectDropletForceCalculator` и `TreeDropletForceCalculator`:

- время одного шага в зависимости от N, `theta`, `mpl`,
- относительная погрешность скоростей (миграция + конвекция),
- результаты пишутся в `benchmarks/results/`.

```bash
python benchmarks/test.py
```

### `run_boundary_benchmark.py` — periodic vs open

Сравнивает 4 конфигурации (direct/tree × open/periodic) при разных `dt`:

```bash
python benchmarks/run_boundary_benchmark.py
```

Результаты — в `results/boundary_benchmark/`.

### `analyze_benchmark.py` — анализ результатов

Строит 6 графиков по CSV-данным:

1. Парето-фронт **speedup vs mean_err**.
2. `mean_err vs theta` при фиксированном `mpl`.
3. `mean_err vs mpl` при фиксированном `theta`.
4. `speedup vs N` при фиксированных `theta`, `mpl`.
5. Тепловая карта ошибки в плоскости `(theta, mpl)`.
6. Эффект дипольной поправки к стоклету.

```bash
python benchmarks/analyze_benchmark.py
```

### `plot_boundary_benchmark.py` / `plot_results.py`

Готовые визуализации для соответствующих CSV-файлов.

## Структура

```
benchmarks/
├── run_boundary_benchmark.py     # запуск periodic/open
├── test.py                       # запуск direct/tree
├── analyze_benchmark.py          # анализ + графики
├── plot_boundary_benchmark.py    # графики граничных условий
├── plot_results.py               # графики direct/tree
├── data.txt                      # справочные параметры
└── plots/                        # генерируется (gitignored, .gitkeep)
```

## Зависимость от pytest-benchmark

```bash
pytest benchmarks/ --benchmark-only
```

Используется опционально — для проверок CI можно отключать `--benchmark-skip`.

## Связанные модули

- [`../dem/force_calculator/`](../dem/force_calculator/),
  [`../dem/octree/`](../dem/octree/) — измеряемые компоненты.
- [`../convergence_test/`](../convergence_test/) — сходимостные тесты.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../LICENSE) — см. корневой [README](../README.md).
