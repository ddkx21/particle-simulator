# `dem.particle_generator` — генерация начальных условий

Создаёт случайное (равномерное) распределение капель внутри куба,
с заданным минимальным расстоянием для предотвращения пересечений.

## Алгоритм UniformDropletGenerator

1. Размер коробки определяется снаружи (см. формулу в `main.py`):
   ```
   L = ∛( π · N · (rₘᵢₙ + rₘₐₓ) · (rₘᵢₙ² + rₘₐₓ²) / (3 · φ_water) )
   ```
2. Радиусы — равномерно из `radii_range`.
3. Позиции — отбраковка с проверкой минимального расстояния
   (rejection sampling) на Taichi для скорости.
4. Если за разумное число попыток не удаётся разместить N частиц —
   генератор уменьшает требование `minimum_distance` или возбуждает
   ошибку.

## Интерфейс

```python
from dem.particle_generator import UniformDropletGenerator

gen = UniformDropletGenerator(
    coord_range=(0, 1e-3),
    radii_range=(2.5e-6, 7.5e-6),
    num_particles=1000,
    minimum_distance=1e-6,
)
state = gen.generate()        # → DropletState
gen.print()                   # текстовый отчёт
gen.plot()                    # 3D-визуализация (matplotlib)
```

## Файлы

| Файл                                | Назначение                       |
|-------------------------------------|----------------------------------|
| `particle_generator_base.py`        | Абстрактный интерфейс            |
| `uniform_droplet_generator.py`      | Равномерная генерация на Taichi |

## Сохранение

Сгенерированное состояние сохраняется через
`DropletState.export_to_xlsx(path)` для последующего повторного
использования (детерминированные эксперименты).

## Тесты

`tests/particle_generator/` — отсутствие пересечений, диапазоны радиусов,
покрытие всего объёма.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
