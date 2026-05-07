# `dem.solution` — хранение траекторий

Контейнер, накапливающий состояния системы во времени. Используется
солвером для записи кадров, пост-процессором — для отрисовки.

## DropletSolution

```python
from dem.solution import DropletSolution

solution = DropletSolution(
    initial_droplet_state=state0,
    real_time_visualization=False,
    update_interval=0.05,        # сек, для real-time
    length=10,                   # сколько кадров держать в памяти
)

# Внутри солвера:
solution.append(new_state)

# После симуляции:
solution.save_chain_to_file("results/run.npz", precision="float32")

# Загрузка
solution = DropletSolution(initial_droplet_state=None,
                           filename="results/run.npz")
```

## Формат `.npz`

```
positions  : (T, N, 3)  float32/64
radii      : (T, N)     float32/64
times      : (T,)       float64
```

`precision="float32"` уменьшает размер файла вдвое, без потери для
визуализации.

## Файлы

| Файл                       | Назначение                              |
|----------------------------|-----------------------------------------|
| `solution_base.py`         | Абстрактный интерфейс `Solution`        |
| `droplet_solution.py`      | Реализация для капель                   |

## Тесты

`tests/solution/` — round-trip сохранение/загрузка, append, обрезка
по `length`.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
