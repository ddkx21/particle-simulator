# `dem.particle_state` — снапшот состояния системы капель

Иммутабельный (по соглашению) контейнер: координаты, радиусы и время
для одного момента симуляции.

## DropletState

```python
from dem.particle_state import DropletState

# Создание из массивов
state = DropletState(
    positions=np.array([[x1, y1, z1], ...]),
    radii=np.array([r1, r2, ...]),
    time=0.0,
)

# Загрузка из файла
state = DropletState(filename="results/initial_N1000.xlsx")
# либо
state = DropletState(filename="results/snapshot.npz")

# Сохранение
state.export_to_xlsx("path.xlsx")
state.save_to_file("path.npz")
```

### Поддерживаемые форматы

- `.xlsx` (через `openpyxl`/`pandas`) — человекочитаемый, удобен для
  ручного редактирования начальных условий.
- `.npz` — компактный бинарный формат NumPy.

## Файлы

| Файл                         | Назначение                              |
|------------------------------|-----------------------------------------|
| `particle_state_base.py`     | Абстрактный интерфейс `ParticleState`   |
| `droplet_state.py`           | Реализация для капель                   |

## Соглашение

Состояние не мутируется напрямую — солвер создаёт новый `DropletState`
для каждого сохраняемого шага и кладёт в `DropletSolution`.

## Тесты

`tests/particle_state/` — round-trip xlsx/npz, корректность размерностей.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
