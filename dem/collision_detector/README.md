# `dem.collision_detector` — детектор столкновений O(N)

Реализует обнаружение пересекающихся капель (|rᵢⱼ| < rᵢ + rⱼ) через
**spatial hashing**.

## Алгоритм

1. **Assign cells** — каждая частица получает hash-bucket по координате.
2. **Prefix sum** — стартовые индексы каждого bucket'а.
3. **Scatter** — частицы раскладываются в отсортированный массив.
4. **Detect** — для каждой частицы проверяются 27 соседних cells
   (3×3×3 в 3D).

Размер hash-таблицы — ближайшая степень двойки ≥ 2N (load factor ≈ 50 %).
Реализовано на Taichi → параллелится на CPU/GPU.

## Интерфейс

```python
from dem.collision_detector import SpatialHashCollisionDetector

detector = SpatialHashCollisionDetector(
    num_particles=10_000,
    L=1e-3,
    boundary_mode="periodic",   # или "open"
)

pairs = detector.detect(positions, radii)   # ndarray (M, 2) индексов
```

## Файлы

| Файл                                  | Назначение                          |
|---------------------------------------|-------------------------------------|
| `collision_detector_base.py`          | Абстрактный интерфейс `CollisionDetector` |
| `spatial_hash_collision_detector.py`  | Реализация на Taichi               |

## Тесты

`tests/collision_detector/` — корректность парных пересечений,
edge-cases на границах периодики.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
