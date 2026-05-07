# `dem.force_calculator` — direct O(N²) расчёт диполь-дипольных сил

Точное попарное суммирование сил между всеми каплями. Используется как
эталон для tree-метода и для малых N (≲ 2000).

## Физика

Сила на каплю *i* от капли *j*:

```
F_ij = m_const · (rᵢ³ rⱼ³ / |rᵢⱼ|⁷) · g(r̂ᵢⱼ),  где
m_const = 12π · ε₀ · ε_oil · E²
```

Угловая зависимость анизотропна вдоль оси электрического поля **z**:
капли вдоль E притягиваются, перпендикулярные — отталкиваются.

Подробно: [`../../ALGORITHM.md#22-угловая-зависимость-grij`](../../ALGORITHM.md).

## Граничные условия

- `"periodic"` — minimum image convention (MIC). Опционально подключается
  COMSOL-поправка через `load_periodic_correction()`.
- `"open"` — открытая коробка, без оборачивания.

## Интерфейс

```python
from dem.force_calculator import DirectDropletForceCalculator

fc = DirectDropletForceCalculator(
    num_particles=1000,
    eps_oil=2.85, eta_oil=0.065, eta_water=0.001,
    rho_water=1000, rho_oil=900, E=3e5,
    L=1e-3, boundary_mode="periodic",
    correction_grid_resolution=0,
)

# Опционально: подключить COMSOL-поправку
from dem.periodic_correction import COMSOLLatticeCorrection
fc.load_periodic_correction(COMSOLLatticeCorrection.load_default(), L_sim=1e-3)

forces, velocities = fc.compute_forces_and_velocities(positions, radii)
```

## Файлы

| Файл                                   | Назначение                          |
|----------------------------------------|-------------------------------------|
| `force_calculator_base.py`             | Базовый абстрактный класс           |
| `direct_droplet_force_calculator.py`   | Реализация O(N²) на Taichi          |

## Связанные модули

- [`../octree/`](../octree/) — Barnes-Hut O(N log N) альтернатива.
- [`../periodic_correction/`](../periodic_correction/) — поправка к стоклету.

## Тесты

`tests/force_calculator/` — корректность сил, симметрия, периодика.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
