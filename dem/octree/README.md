# `dem.octree` — Flat Barnes-Hut octree (O(N log N))

Плоское октодерево с послойным построением для параллельных вычислений
на Taichi. Используется для расчёта диполь-дипольных сил и стоксовского
поля при N ≳ 5000.

## Стратегия построения

Дерево строится **уровень за уровнем** — максимальный параллелизм без
локов и гонок:

1. **assign_octants** (parallel) — каждая частица определяет свой октант
   в текущем узле.
2. **subdivide_and_alloc** (serial по активным узлам) — узлы с
   `count > mpl` разбиваются на 8 дочерних.
3. **scatter** (parallel) — частицы перемещаются в дочерние листы.

Повторяется до тех пор, пока во всех листьях `count ≤ mpl` или достигнут
`max_depth`.

## Агрегация R³ (bottom-up)

Для аппроксимации Барнса-Хатта в каждом узле хранятся:

- `R3_sum = Σ rⱼ³`
- `R3_cx, R3_cy, R3_cz = Σ posⱼ · rⱼ³`

Центр масс: `center = (R3_cx, R3_cy, R3_cz) / R3_sum`.
"Эффективный радиус": `R3_eff = R3_sum^(1/3)`.

## Обход дерева (stackless)

Для каждой частицы параллельно:

1. **Лист** → точное попарное взаимодействие со всеми частицами листа.
2. **Внутренний узел** → проверка критерия открытия:
   - `s² < θ² · D²` ⇒ аппроксимировать (одна "эффективная капля").
   - иначе ⇒ раскрыть и спуститься в дочерние узлы.

Параметр `theta` (по умолчанию 0.5):
- `theta = 0` — direct (всегда раскрывать);
- `theta = 1` — грубая аппроксимация.

## Файлы

| Файл                  | Назначение                                       |
|-----------------------|--------------------------------------------------|
| `octree_node.py`      | Структура узла (`count`, `R3_sum`, дочерние)    |
| `flat_tree.py`        | Построение и обход (без расчёта сил)            |
| `force_tree.py`       | Tree-расчёт диполь-дипольных сил + стоклета     |
| `tree_stats.py`       | Статистика по дереву (глубина, occupancy)       |
| `tree_visualizer.py`  | Визуализация структуры дерева                   |

## Интерфейс

```python
from dem.octree.force_tree import TreeDropletForceCalculator

fc = TreeDropletForceCalculator(
    num_particles=10_000,
    eps_oil=2.85, eta_oil=0.065, eta_water=0.001,
    rho_water=1000, rho_oil=900, E=3e5,
    L=1e-3, boundary_mode="periodic",
    theta=0.5, mpl=8, max_depth=20,
)
forces, velocities = fc.compute_forces_and_velocities(positions, radii)
```

## Тесты

`tests/octree/` — корректность построения, сравнение с direct-методом,
edge-cases (один узел, пустые ветки, max_depth).

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
