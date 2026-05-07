# `dem.solver` — интеграция уравнений движения

Интегратор Эйлера для уравнений движения капель в стоксовском режиме
(низкие числа Рейнольдса).

## Уравнение движения

В стоксовском пределе инерция пренебрежимо мала:

```
v_i = F_i / γ_i + u_oseen(r_i),    где
γ_i = 6πη_oil · r_i · (2η_oil + 3η_water) / (η_oil + η_water)
```

Интегрирование явным Эйлером:

```
r_i(t + dt) = r_i(t) + v_i · dt
```

При `boundary_mode="periodic"` после шага позиции оборачиваются в
`[0, L)³`.

## Цикл солвера

```
while t < t_stop:
    forces, velocities = force_calculator.compute(positions, radii)
    positions += velocities * dt
    if collision_detector:
        pairs = collision_detector.detect(positions, radii)
        merge_colliding_droplets(pairs)        # объединение по объёму
    if pbm_coupling:
        pbm_coupling.transfer(...)             # обмен с PBM
    if step % save_interval == 0:
        solution.append(snapshot(positions, radii, t))
    t += dt
```

## Интерфейс

```python
from dem.solver import EulerDropletSolver

solver = EulerDropletSolver(
    force_calculator=fc,
    solution=sol,
    post_processor=pp,
    collision_detector=cd,        # опционально
    save_interval=1,              # каждый K-й шаг сохраняется
    pbm_coupling=None,            # опционально DEMPBMCoupling
)
solver.solve(dt=0.04, t_stop=100)
```

## Файлы

| Файл                           | Назначение                              |
|--------------------------------|-----------------------------------------|
| `solver_base.py`               | Абстрактный интерфейс `Solver`          |
| `euler_droplet_solver.py`      | Эйлер-интегратор + слияние при коллизии |

## Слияние капель

При обнаружении пересечения двух капель `i, j` они заменяются на одну
каплю с:
- объёмом `V = Vᵢ + Vⱼ` ⇒ радиус `r = ∛(rᵢ³ + rⱼ³)`,
- центром масс по объёму,
- индексом `min(i, j)` (вторая капля помечается удалённой).

## Тесты

`tests/solver/` — конвергенция Эйлера по dt, сохранение объёма при
слияниях, корректность периодического оборачивания.

## Связанные модули

- [`../force_calculator/`](../force_calculator/) — вычисление сил.
- [`../collision_detector/`](../collision_detector/) — детектор столкновений.
- [`../../pbm/coupling.py`](../../pbm/coupling.py) — связка с PBM.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
