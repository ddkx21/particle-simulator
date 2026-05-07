# `dem.post_processor` — визуализация и пост-обработка

3D-рендеринг траекторий, анимация эволюции системы, экспорт кадров.

## Возможности

- Реал-тайм отрисовка во время `solver.solve()` (флаг
  `real_time_visualization=True` в `DropletSolution`).
- Покадровая анимация после симуляции (через `matplotlib.animation`).
- Отрисовка периодического бокса.
- Сохранение последнего кадра в PNG / последовательности в GIF.

## Интерфейс

```python
from dem.post_processor import DropletPostProcessor

pp = DropletPostProcessor(solution=solution, box_size=1e-3)
pp.plot_final_state()                         # последний кадр
pp.animate(filename="evolution.gif", fps=30)  # анимация
```

## Файлы

| Файл                          | Назначение                              |
|-------------------------------|-----------------------------------------|
| `post_processor_base.py`      | Абстрактный интерфейс `PostProcessor`   |
| `droplet_post_processor.py`   | matplotlib-реализация для капель        |

## Производительность

3D matplotlib медленный — для больших N включайте только финальный кадр.
Тяжёлые анимации лучше делать через [`../../studies/animate_evolution.py`](../../studies/animate_evolution.py)
или экспорт в [VTK / ParaView].

## Связанные модули

- [`../solution/`](../solution/) — источник траекторий.
- [`../../studies/animate_evolution.py`](../../studies/animate_evolution.py) —
  альтернативный аниматор.

---

## Автор и контакты

**Dadakhodjaev Rustam Bakhtiyorovich**
[rustam.dadakhodjaev@gmail.com](mailto:rustam.dadakhodjaev@gmail.com) ·
[st094266@student.spbu.ru](mailto:st094266@student.spbu.ru) ·
Telegram: [@ddkx21](https://t.me/ddkx21)

Лицензия: [MIT](../../LICENSE) — см. корневой [README](../../README.md).
