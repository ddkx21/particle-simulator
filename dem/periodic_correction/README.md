# `dem.periodic_correction` — псевдо-периодическая поправка из COMSOL

Реализует **истинно-периодическое** стоксовское поле через сумму:

```
u_periodic(r) = u_stokeslet(r)  +  u_correction(r)
                  ↑                       ↑
            свободное пространство    поправка от
                                     периодических образов
```

Поправка `u_correction` предвычислена в COMSOL Multiphysics на нерегулярной
сетке в четверть-ячейке `[0, L/2]² × [-L/2, L/2]`. Класс зеркалирует
данные в полную ячейку `[-L/2, L/2]³` и пересэмплирует на регулярную сетку
для быстрой трилинейной интерполяции в Taichi.

## Зачем нужно

Прямой суммой стоклетов от периодических образов сходится только условно
(гармонический ряд). Использование частичных сумм даёт неправильный отклик.
COMSOL-поправка — единственный способ получить корректную периодическую
функцию Грина без сложной Эвальд-суммации.

Корректность подтверждается тестом сходимости —
[`../../convergence_test/README.md`](../../convergence_test/README.md).

## Интерфейс

```python
from dem.periodic_correction import COMSOLLatticeCorrection

# Из встроенной директории data/
correction = COMSOLLatticeCorrection.load_default()

# Из произвольной директории
correction = COMSOLLatticeCorrection.from_data_dir("path/to/data/")

# Подключение к force calculator
force_calculator.load_periodic_correction(correction, L_sim=1e-3)
```

## Данные

```
data/
├── U_lattice.txt          # точки (x, y, z, u, v, w) поправки
└── comsol_params.json     # размер ячейки, разрешение сетки, метаданные
```

## Файлы

| Файл                              | Назначение                                |
|-----------------------------------|-------------------------------------------|
| `comsol_lattice_correction.py`    | Загрузка, зеркалирование, ресэмплинг      |
| `data/U_lattice.txt`              | Сырые данные COMSOL                       |
| `data/comsol_params.json`         | Параметры расчёта                         |

## Ограничения

- Требует `boundary_mode = "periodic"` в `ForceCalculator`.
- Один набор данных — для конкретного отношения L_comsol/r_droplet;
  при экстремально других масштабах необходимо пересчитать в COMSOL.

## Связанные документы

- [`../../convergence_test/README.md`](../../convergence_test/README.md) —
  валидация сходимости.
- [`../../docs/periodic_correction_plan/`](../../docs/periodic_correction_plan/) —
  внутренние заметки и план.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
