# `dem.utils` — вспомогательные функции

Общие утилиты, используемые несколькими модулями DEM.

## Содержимое

- **MIC (minimum image convention)** — оборачивание разности координат
  `Δr` в `[-L/2, L/2)` для периодических ГУ.
- **Расстояния** — векторизованные расчёты с учётом периодики.
- **Преобразования** — конверсии "позиции ↔ Taichi-поля".

## Использование

```python
from dem.utils import mic_displacement, periodic_distance

dr = mic_displacement(r_i, r_j, L)
d  = periodic_distance(r_i, r_j, L)
```

## Файлы

| Файл        | Назначение                                                   |
|-------------|--------------------------------------------------------------|
| `utils.py`  | Векторизованные NumPy-утилиты + Taichi-обёртки              |

## Соглашение

Утилиты не зависят от других модулей DEM (только от `numpy`/`taichi`),
чтобы исключить циклические импорты.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
