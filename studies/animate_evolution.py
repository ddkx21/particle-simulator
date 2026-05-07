import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dem.solution import DropletSolution
from dem.particle_state import DropletState

"""
Скрипт для создания GIF анимации эволюции системы капель из NPZ файлов.
"""

# === Настройки ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_filename = os.path.join(PROJECT_ROOT, "results/results_N10000_vol0.02_dt0.04_0_d.npz")
output_filename = os.path.join(PROJECT_ROOT, "results/evolution.gif")
num_frames = 100       # Количество кадров в анимации
fps = 10               # Кадров в секунду
dpi = 200              # Разрешение

# === Загрузка данных ===
print(f"Загрузка решения из {results_filename}...")
solution = DropletSolution.load_chain_from_file(results_filename)

# Определяем временной диапазон всей цепочки
first_sol = solution.get_first_solution()
t_start = first_sol.times[0][0]

# Находим конец цепочки
current = first_sol
while current._next is not None:
    current = current._next
t_end = current.times[current.current_step][0]

print(f"Временной диапазон: [{t_start:.2f}, {t_end:.2f}]")

# Генерируем временные точки для кадров
frame_times = np.linspace(t_start, t_end, num_frames)

# Определяем box_size из начального состояния
state0 = solution.get_state(t_start)
box_size = np.max(np.abs(state0.positions))

# Определяем глобальный диапазон радиусов для единой цветовой шкалы
r_min = state0.radii.min()
r_max = state0.radii.max()
for t in frame_times[1:]:
    try:
        st = solution.get_state(t)
        r_min = min(r_min, st.radii.min())
        r_max = max(r_max, st.radii.max())
    except ValueError:
        pass

# === Создание анимации ===
fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

# Цветовая карта
norm = plt.Normalize(vmin=r_min, vmax=r_max)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.6, label='Радиус')

def update(frame_idx):
    ax.cla()
    t = frame_times[frame_idx]

    try:
        state = solution.get_state(t)
    except ValueError:
        return

    positions = state.positions
    radii = state.radii
    n = len(radii)

    # Размер маркеров пропорционален радиусу (масштабируем для видимости)
    sizes = (radii / r_max) ** 2 * 2500
    colors = plt.cm.viridis(norm(radii))

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               s=sizes, c=colors, alpha=0.7, edgecolors='none')

    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f't = {t:.2f}  |  N = {n}')

    if frame_idx % 10 == 0:
        print(f"  Кадр {frame_idx + 1}/{num_frames}")

print("Создание анимации...")
anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 // fps)

print(f"Сохранение в {output_filename}...")
anim.save(output_filename, writer=PillowWriter(fps=fps), dpi=dpi)
plt.close()
print("Готово!")
