# `dual_ensemble/` — параллельный запуск двух систем

Сравнение двух независимых ансамблей с разными параметрами (например,
мелкие vs крупные капли) — для контроля статистических выводов.

## Концепция

Запускаются две параллельные системы (`system1`, `system2`) с одной
конфигурацией физики, но разными начальными распределениями радиусов.
После расчёта обе обрабатываются единым модулем метрик и графиков.

## Использование

```bash
# Конфиг — config.json в этой директории
# Свежий запуск двух систем
python -m dual_ensemble.run_system1
python -m dual_ensemble.run_system2

# Продолжить с checkpoint
python -m dual_ensemble.run_system1 --continue

# Переопределить t_stop
python -m dual_ensemble.run_system1 --t-stop 300

# После запуска — графики сравнения
python -m dual_ensemble.plot_dual_ensemble
```

## Структура

```
dual_ensemble/
├── config.json              # параметры обеих систем
├── run_common.py            # общая логика запуска (CLI, чекпойнты)
├── run_system1.py           # система 1 (мелкие капли)
├── run_system2.py           # система 2 (крупные капли)
├── metrics.py               # расчёт метрик (median radius, std, и т.д.)
├── plot_dual_ensemble.py    # сравнительные графики
└── results/                 # выходные .npz и графики (gitignored)
```

## `config.json`

Содержит:
- параметры обеих систем (`N`, `radii_range`, `volume_content`),
- общие физические константы (`E`, `eps_oil`, `eta_oil`, ...),
- настройки солвера (`dt`, `t_stop`, `boundary_mode`),
- частоту чекпойнтов.

## Чекпойнты

Каждые `N` шагов состояние пишется в `results/system{1,2}_checkpoint.npz`.
Флаг `--continue` подхватывает последний чекпойнт.

## Связанные модули

- [`../dem/`](../dem/) — DEM-ядро, используется напрямую.
- [`../studies/postproc_compare.py`](../studies/postproc_compare.py) —
  альтернативный сравнительный пост-процесс.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../LICENSE) — см. корневой [README](../README.md).
