# `pbm.collision_frequency` — подсчёт частот коллизий по бинам

Извлекает **матрицу коллизий** `C[i, j]` — сколько раз за интервал времени
капли из bin `i` столкнулись с каплями из bin `j`. Используется для
построения [DEMExtractedKernel](../kernels/README.md).

## Реализации

| Класс                          | Сложность   | Когда использовать          |
|--------------------------------|-------------|------------------------------|
| `DirectCollisionFrequency`     | O(N²)       | малые N, эталон              |
| `TreeCollisionFrequency`       | O(N log N)  | большие N, привязан к octree |

Tree-вариант обходит то же `FlatOctree`, что и силовой расчёт, и отсекает
далёкие поддеревья по критерию `min_dist > rᵢ + max_radius_of_subtree`.

## Интерфейс

### Direct

```python
from pbm import VolumeGrid
from pbm.collision_frequency import DirectCollisionFrequency

grid = VolumeGrid(v_min=1e-18, v_max=1e-12, n_bins=64, geometric=True)
counter = DirectCollisionFrequency(grid)

C = counter.compute(positions, radii, L=1e-3, periodic=True)
# C[i, j] — количество коллизий между bin i и bin j
```

### Tree

```python
from pbm.collision_frequency import TreeCollisionFrequency

counter = TreeCollisionFrequency(flat_tree=octree, grid=grid,
                                 max_particles=10_000)
C = counter.compute(positions, radii)
```

## Файлы

| Файл                          | Назначение                              |
|-------------------------------|-----------------------------------------|
| `direct_collision_freq.py`    | Brute-force O(N²) подсчёт              |
| `tree_collision_freq.py`      | Octree-ускоренный O(N log N) подсчёт   |

## Конверсия в ядро

```
Q[i, j] = C[i, j] / (Δt · n[i] · n[j] / V_d)
```

где `n[i]` — количество капель в bin `i`, `V_d` — объём домена.
Эта конверсия делается в `DEMExtractedKernel`.

## Тесты

`tests/pbm/` — корректность матрицы (симметрия, неотрицательность),
эквивалентность direct и tree методов.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
