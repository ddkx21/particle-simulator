# `studies/` — отдельные исследования

Скрипты, не входящие в основной пайплайн, но используемые для
аналитических исследований и проверок.

## Содержимое

### `convergence_study.py`

Анализ сходимости по временному шагу `dt`. Запускает симуляцию с
одинаковыми начальными условиями для нескольких `dt`, сравнивает
результаты, строит график "ошибка vs dt".

```bash
python studies/convergence_study.py
```

### `euler_method_error.py`

Исследование статистической ошибки метода Эйлера. Для каждого `dt` из
списка прогоняет N независимых запусков, усредняет, оценивает
дисперсию.

```bash
python studies/euler_method_error.py
```

Результаты → `results/euler_error/`.

### `animate_evolution.py`

Анимирует эволюцию системы из `.npz`-файла (альтернатива
`DropletPostProcessor`). Использует `matplotlib.animation`.

```bash
python studies/animate_evolution.py results/run.npz
```

### `postproc_compare.py`

Сравнительный пост-процессинг двух симуляций (направление сходства,
diff траекторий).

```bash
python studies/postproc_compare.py results/run_a.npz results/run_b.npz
```

## Структура

```
studies/
├── convergence_study.py      # сходимость по dt
├── euler_method_error.py     # статистическая ошибка Эйлера
├── animate_evolution.py      # анимация из .npz
└── postproc_compare.py       # сравнение двух запусков
```

## Соглашение

Скрипты в `studies/` — одноразовые/исследовательские. Они не покрываются
тестами и не попадают в линтер/форматтер (см. `pyproject.toml`,
`extend-exclude`). Стабилизированные после исследования утилиты следует
переносить в соответствующий пакет (`dem.*` или `pbm.*`).

## Связанные модули

- [`../benchmarks/`](../benchmarks/) — производительность.
- [`../convergence_test/`](../convergence_test/) — сходимость периодики.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../LICENSE) — см. корневой [README](../README.md).
