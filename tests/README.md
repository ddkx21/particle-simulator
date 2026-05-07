# `tests/` — pytest-сьюты

Покрытие модулей DEM и PBM. Запуск из корня репозитория:

```bash
pytest tests/                                        # все тесты
pytest tests/ --cov=dem --cov=pbm                    # с покрытием
pytest tests/ -m "not slow"                          # быстрые тесты
pytest tests/octree/                                 # один модуль
pytest tests/ -k "test_force"                        # по имени
```

## Структура

```
tests/
├── conftest.py                  # общие фикстуры, инициализация Taichi
├── collision_detector/          # тесты spatial-hash детектора
├── force_calculator/            # direct O(N²) расчёт сил
├── octree/                      # построение и обход дерева
├── particle_generator/          # генерация без пересечений
├── particle_state/              # сериализация xlsx/npz
├── pbm/                         # PBM (volume_grid, redistribution, kernels)
├── periodic_correction/         # COMSOL lattice
├── solution/                    # хранение траекторий
└── solver/                      # Эйлер + слияния + периодика
```

## Маркеры

Определены в [`../pyproject.toml`](../pyproject.toml):

| Маркер | Назначение                        | Запуск                   |
|--------|-----------------------------------|--------------------------|
| `slow` | долгие конвергентные тесты        | `pytest -m slow`         |

## Покрытие

Цель — **80 %+** для `dem/` и `pbm/`. Текущее измеряется в CI и
сохраняется как артефакт `coverage-py3.12`.

```bash
pytest --cov=dem --cov=pbm --cov-report=html
xdg-open htmlcov/index.html
```

## Соглашения

- Файлы тестов: `test_*.py`.
- Функции тестов: `test_*` или классы `Test*`.
- Фикстуры с тяжёлой инициализацией Taichi — в `conftest.py` со скоупом
  `module` или `session`.

## CI

Все тесты прогоняются в [`../.github/workflows/ci.yml`](../.github/workflows/ci.yml)
на Python 3.12. Артефакты — `coverage.xml`, `junit-3.12.xml`.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../LICENSE) — см. корневой [README](../README.md).
