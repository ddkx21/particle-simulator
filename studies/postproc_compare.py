import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from datetime import datetime
from force_calculator import *
from particle_generator import *
from particle_state import *
from post_processor import *
from solution import *
from solver import *
import taichi as ti
ti.init(arch=ti.cpu, cpu_max_num_threads=6)  # Инициализация Taichi с использованием CPU
# ti.init(arch=ti.gpu)  # Инициализация Taichi с использованием GPU

print('\n'*3)

"""
Скрипт для сравнения результатов двух разных решений
"""

results_filename_1 = "/home/ddkx/Work/particle_simulator_v20_clean/results/results_N1000_vol0.02_dt0.04_0_d1.npz"
results_filename_2 = "/home/ddkx/Work/particle_simulator_v20_clean/results/results_N1000_vol0.02_dt0.04_0_d2.npz"

solution_1 = DropletSolution.load_chain_from_file(results_filename_1)
solution_2 = DropletSolution.load_chain_from_file(results_filename_2)

# Вычисляем box_size
xs = solution_1.get_state(0).positions[:, 0]
ys = solution_1.get_state(0).positions[:, 1]
zs = solution_1.get_state(0).positions[:, 2]
box_size = np.max([np.max(xs), np.max(ys), np.max(zs)])

post_processor_1 = DropletPostProcessor(solution_1, box_size=box_size)  
post_processor_2 = DropletPostProcessor(solution_2, box_size=box_size) 


time = 100
post_processor_1.plot_state(time, plt_show=False)
post_processor_2.plot_state(time)

post_processor_1.plot_number_of_droplets(time_step=1, plt_show=False)
post_processor_2.plot_number_of_droplets(time_step=1)

#post_processor_1.plot_radius_evolution(plt_show=False)
#post_processor_2.plot_radius_evolution()

#post_processor_1.plot_radius_histogram(time, plt_show=False)
#post_processor_1.plot_radius_histogram(time)

#post_processor_1.plot_volume_histogram(time,num_bins=1000, plt_show=False)
#post_processor_1.plot_volume_histogram(time,num_bins=1000)

#post_processor_1.plot_radius_ratio_statistics(window_size=10, overlap=0.5, ratio_intervals=[2, 8], plt_show=False)
#post_processor_2.plot_radius_ratio_statistics(window_size=10, overlap=0.5, ratio_intervals=[2, 8])

#post_processor_1.plot_angle_statistics(window_size=10, overlap=0.5, angle_intervals=[5, 10, 15], plt_show=False)
#post_processor_2.plot_angle_statistics(window_size=10, overlap=0.5, angle_intervals=[5, 10, 15])

#post_processor_1.plot_angle_histogram(plt_show=False)
#post_processor_2.plot_angle_histogram()

#post_processor_1.plot_volume_histogram(time, plt_show=False)
#post_processor_2.plot_volume_histogram(time, plt_show=False)

#post_processor_1.compare(solution_2, time_step=1)


# ======================================================
# Графики ошибок
# ======================================================
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde

analysis_time = time  # Используем тот же момент времени

state_1 = solution_1.get_state(analysis_time)
state_2 = solution_2.get_state(analysis_time)

# --- 2a. Ошибка volume distribution (KDE кривые) ---
volumes_1 = (4/3) * np.pi * state_1.radii**3
volumes_2 = (4/3) * np.pi * state_2.radii**3

# KDE для обоих распределений
kde_1 = gaussian_kde(volumes_1)
kde_2 = gaussian_kde(volumes_2)

all_volumes = np.concatenate([volumes_1, volumes_2])
v_grid = np.linspace(all_volumes.min(), all_volumes.max(), 500)
pdf_1 = kde_1(v_grid)
pdf_2 = kde_2(v_grid)

abs_error = np.abs(pdf_1 - pdf_2)
mean_pdf = 0.5 * (pdf_1 + pdf_2)
rel_error = np.where(mean_pdf > 1e-10, abs_error / mean_pdf * 100, 0)

fig_vol, (ax_vol1, ax_vol2) = plt.subplots(1, 2, figsize=(14, 5))

# Наложенные KDE кривые
ax_vol1.plot(v_grid, pdf_1, label='Решение 1', linewidth=2)
ax_vol1.plot(v_grid, pdf_2, label='Решение 2', linewidth=2, linestyle='--')
ax_vol1.fill_between(v_grid, pdf_1, alpha=0.2)
ax_vol1.fill_between(v_grid, pdf_2, alpha=0.2)
ax_vol1.set_xlabel('Объём капли')
ax_vol1.set_ylabel('Плотность вероятности')
ax_vol1.set_title(f'Volume distribution (t={analysis_time})')
ax_vol1.legend()

# Ошибка
ax_vol2.plot(v_grid, rel_error, color='red', linewidth=2)
ax_vol2.fill_between(v_grid, rel_error, alpha=0.2, color='red')
ax_vol2.set_xlabel('Объём капли')
ax_vol2.set_ylabel('Относительная ошибка, %')
ax_vol2.set_title('Ошибка volume distribution')

fig_vol.tight_layout()
fig_vol.savefig('compare_volume_distribution.png', dpi=300, bbox_inches='tight')

# --- 2b. % ошибки позиции по размеру капель ---
# Сопоставляем капли по KDTree (ближайшие соседи)
tree_2 = KDTree(state_2.positions)
distances, indices = tree_2.query(state_1.positions)

# % ошибки позиции
pos_error_pct = distances / box_size * 100

fig_pos, (ax_pos1, ax_pos2) = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: радиус vs % ошибки
ax_pos1.scatter(state_1.radii, pos_error_pct, s=5, alpha=0.3)
ax_pos1.set_xlabel('Радиус капли')
ax_pos1.set_ylabel('Ошибка позиции, % от box_size')
ax_pos1.set_title(f'Ошибка позиции по размеру (t={analysis_time})')

# Binned средняя ошибка
n_rbins = 20
radii_sorted_idx = np.argsort(state_1.radii)
radii_sorted = state_1.radii[radii_sorted_idx]
error_sorted = pos_error_pct[radii_sorted_idx]

r_bin_edges = np.linspace(radii_sorted.min(), radii_sorted.max(), n_rbins + 1)
r_bin_centers = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])
mean_errors = np.zeros(n_rbins)
for i in range(n_rbins):
    mask = (radii_sorted >= r_bin_edges[i]) & (radii_sorted < r_bin_edges[i+1])
    if mask.any():
        mean_errors[i] = error_sorted[mask].mean()

ax_pos2.bar(r_bin_centers, mean_errors, width=np.diff(r_bin_edges)[0], color='steelblue', alpha=0.8)
ax_pos2.set_xlabel('Радиус капли')
ax_pos2.set_ylabel('Средняя ошибка позиции, %')
ax_pos2.set_title('Средняя ошибка по диапазонам радиусов')

fig_pos.tight_layout()
fig_pos.savefig('compare_position_error.png', dpi=300, bbox_inches='tight')

# ======================================================
# Графики разности
# ======================================================

# --- 3a. Гистограмма углов столкновений (разность) ---

#def extract_collision_angles(solution):
#    """Извлекает углы столкновений из цепочки решений."""
#    all_angles = []
#    current_solution = solution.get_first_solution()
#    while current_solution is not None:
#        if current_solution.is_collision:
#            for pair in current_solution.collided_droplets:
#                if current_solution.trajectories.shape[0] >= 2:
#                    r1 = current_solution.radii[pair[0]]
#                    r2 = current_solution.radii[pair[1]]
#                    collision_distance = r1 + r2
#                    pos1_last = current_solution.trajectories[-1, pair[0], :]
#                    pos2_last = current_solution.trajectories[-1, pair[1], :]
#                    pos1_prev = current_solution.trajectories[-2, pair[0], :]
#                    pos2_prev = current_solution.trajectories[-2, pair[1], :]
#                    delta1 = pos1_last - pos1_prev
#                    delta2 = pos2_last - pos2_prev
#                    current_distance = np.linalg.norm(pos2_last - pos1_last)
#                    prev_distance = np.linalg.norm(pos2_prev - pos1_prev)
#                    denom = current_distance - prev_distance
#                    if abs(denom) > 1e-30:
#                        t_contact = (current_distance - collision_distance) / denom
#                        pos1_contact = pos1_prev + t_contact * delta1
#                        pos2_contact = pos2_prev + t_contact * delta2
#                        vector = pos2_contact - pos1_contact
#                    else:
#                        vector = pos2_last - pos1_last
#                else:
#                    pos1 = current_solution.trajectories[-1, pair[0], :]
#                    pos2 = current_solution.trajectories[-1, pair[1], :]
#                    vector = pos2 - pos1
#                z_vector = np.array([0, 0, 1])
#                cos_theta = np.dot(vector, z_vector) / (np.linalg.norm(vector) * np.linalg.norm(z_vector))
#                angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
#                if angle > 90:
#                    angle = 180 - angle
#                all_angles.append(angle)
#        current_solution = current_solution._next
#    return np.array(all_angles)
#
#angles_1 = extract_collision_angles(solution_1)
#angles_2 = extract_collision_angles(solution_2)
#
#num_angle_bins = 29
#angle_range = (0, 90)
#bin_edges_a = np.linspace(angle_range[0], angle_range[1], num_angle_bins + 1)
#hist_a1, _ = np.histogram(angles_1, bins=bin_edges_a)
#hist_a2, _ = np.histogram(angles_2, bins=bin_edges_a)
#diff_a = hist_a1 - hist_a2
#width_a = np.diff(bin_edges_a)
#
#fig_ang, (ax_ang1, ax_ang2) = plt.subplots(1, 2, figsize=(14, 5))
#ax_ang1.bar(bin_edges_a[:-1], hist_a1, width=width_a, edgecolor="black", align="edge", alpha=0.6, label='Решение 1')
#ax_ang1.bar(bin_edges_a[:-1], hist_a2, width=width_a, edgecolor="black", align="edge", alpha=0.6, label='Решение 2')
#ax_ang1.set_title('Гистограмма углов столкновений капель')
#ax_ang1.set_xlabel('Угол (градусы)')
#ax_ang1.set_ylabel('Количество столкновений')
#ax_ang1.legend()
#ax_ang1.grid(True)
#
#ax_ang2.bar(bin_edges_a[:-1], diff_a, width=width_a, edgecolor="black", align="edge", color='steelblue')
#ax_ang2.set_title('Разность (Решение 1 − Решение 2)')
#ax_ang2.set_xlabel('Угол (градусы)')
#ax_ang2.set_ylabel('Разность')
#ax_ang2.axhline(0, color='black', linewidth=0.5)
#ax_ang2.grid(True)
#fig_ang.tight_layout()
#fig_ang.savefig('compare_collision_angles.png', dpi=300, bbox_inches='tight')
#
## --- 3b. Количество капель (разность) ---
#
#time_window = [solution_1.get_chain_start_time(), solution_1.get_chain_end_time()]
#times_nd = np.arange(time_window[0], time_window[1], 1)
#nums_1 = np.zeros_like(times_nd)
#nums_2 = np.zeros_like(times_nd)
#for ind, t in enumerate(times_nd):
#    nums_1[ind] = solution_1.get_state(t).radii.shape[0]
#    nums_2[ind] = solution_2.get_state(t).radii.shape[0]

#fig_num, (ax_num1, ax_num2) = plt.subplots(1, 2, figsize=(14, 5))
#ax_num1.plot(times_nd, nums_1, label='Решение 1', linewidth=2)
#ax_num1.plot(times_nd, nums_2, label='Решение 2', linewidth=2, linestyle='--')
#ax_num1.set_title('Количество капель от времени')
#ax_num1.set_xlabel('Время, с')
#ax_num1.set_ylabel('Количество капель')
#ax_num1.legend()
#ax_num1.grid(True)
#
#ax_num2.plot(times_nd, nums_1 - nums_2, color='steelblue', linewidth=2)
#ax_num2.set_title('Разность (Решение 1 − Решение 2)')
#ax_num2.set_xlabel('Время, с')
#ax_num2.set_ylabel('Разность')
#ax_num2.axhline(0, color='black', linewidth=0.5)
#ax_num2.grid(True)
#fig_num.tight_layout()
#fig_num.savefig('compare_droplet_count.png', dpi=300, bbox_inches='tight')

# --- 3c. Распределение радиусов (разность) ---

radii_1 = state_1.radii
radii_2 = state_2.radii

num_r_bins = 20
all_radii = np.concatenate([radii_1, radii_2])
bin_edges_r = np.linspace(all_radii.min(), all_radii.max(), num_r_bins + 1)
hist_r1, _ = np.histogram(radii_1, bins=bin_edges_r)
hist_r2, _ = np.histogram(radii_2, bins=bin_edges_r)
diff_r = hist_r1 - hist_r2
width_r = np.diff(bin_edges_r)

fig_rad, (ax_rad1, ax_rad2) = plt.subplots(1, 2, figsize=(14, 5))
ax_rad1.bar(bin_edges_r[:-1], hist_r1, width=width_r, edgecolor="black", align="edge", alpha=0.6, label='Решение 1')
ax_rad1.bar(bin_edges_r[:-1], hist_r2, width=width_r, edgecolor="black", align="edge", alpha=0.6, label='Решение 2')
ax_rad1.set_title(f'Распределение радиусов в момент времени t={analysis_time}')
ax_rad1.set_xlabel('Радиус')
ax_rad1.set_ylabel('Количество капель')
ax_rad1.legend()
ax_rad1.grid(True)

ax_rad2.bar(bin_edges_r[:-1], diff_r, width=width_r, edgecolor="black", align="edge", color='steelblue')
ax_rad2.set_title('Разность (Решение 1 − Решение 2)')
ax_rad2.set_xlabel('Радиус')
ax_rad2.set_ylabel('Разность')
ax_rad2.axhline(0, color='black', linewidth=0.5)
ax_rad2.grid(True)
fig_rad.tight_layout()
fig_rad.savefig('compare_radius_distribution.png', dpi=300, bbox_inches='tight')

plt.show()
