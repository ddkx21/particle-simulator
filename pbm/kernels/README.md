# `pbm.kernels` — ядра столкновений Q(v₁, v₂)

Ядро столкновений `Q(v₁, v₂)` — это **скорость слияния** капель
объёма `v₁` и `v₂` в единице полного объёма системы. Размерность —
`[м³ / с]`.

## Доступные ядра

### `AnalyticalElectrostaticKernel`

Аналитическая аппроксимация для диполь-дипольной коалесценции в
осцилляторном электрическом поле:

```
Q(v₁, v₂) = C · (r₁ + r₂)² · |r₁³ + r₂³| / (rᵢⱼ_min)^p
```

С физическими константами:

- `eps_oil`, `E` — диэлектрическая проницаемость и поле,
- `eta_oil` — вязкость,
- C, p — параметры формы.

```python
from pbm import VolumeGrid
from pbm.kernels import AnalyticalElectrostaticKernel

grid = VolumeGrid(v_min=1e-18, v_max=1e-12, n_bins=64, geometric=True)
kernel = AnalyticalElectrostaticKernel(
    grid=grid,
    eps0=8.85e-12, eps_oil=2.85, eta_oil=0.065, E=3e5,
)
Q = kernel.matrix()                # ndarray (n_bins, n_bins)
```

### `DEMExtractedKernel`

Загружает ядро из частотной матрицы, накопленной в DEM-симуляции.
Ключевой инструмент для калибровки PBM из реальной MD-статистики.

```python
from pbm.kernels import DEMExtractedKernel

kernel = DEMExtractedKernel(
    grid=grid,
    collision_matrix=collision_matrix,   # (n_bins, n_bins) от DEM
    domain_volume=L**3,
    elapsed_time=t_elapsed,
)
Q = kernel.matrix()
```

## Файлы

| Файл                          | Назначение                              |
|-------------------------------|-----------------------------------------|
| `analytical_kernel.py`        | Аналитическое электростатическое ядро   |
| `dem_extracted_kernel.py`     | Ядро из DEM-статистики коллизий         |

## Связанные модули

- [`../volume_grid.py`](../volume_grid.py) — определение бинов.
- [`../collision_frequency/`](../collision_frequency/) — подсчёт коллизий
  в DEM для извлечения ядра.
- [`../coupling.py`](../coupling.py) — обмен ядром между DEM и PBM.

## Соглашения

- `Q` — симметричная матрица: `Q[i, j] = Q[j, i]`.
- `Q[i, j]` — для пары `(bin_i, bin_j)`, в абсолютных частотах
  поделённых на полный объём `V_d`.
- Диагональ `Q[i, i]` соответствует слиянию двух капель одного бина —
  множитель 0.5 берётся в солвере.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
