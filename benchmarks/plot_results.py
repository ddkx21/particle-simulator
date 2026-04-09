#!/usr/bin/env python3
"""
Визуализация результатов сравнения прямого метода и дерева (Barnes-Hut).
Тестировалось на Ryzen 4
Каждый график сохраняется в отдельный файл в папке plots/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Настройки ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(os.path.join(SCRIPT_DIR, "data.txt"))

# Уникальные значения
all_N = sorted(df["N"].unique())
all_theta = sorted(df["theta"].unique())
all_mpl = sorted(df["mpl"].unique())

# Цветовые палитры
cmap_theta = plt.cm.plasma(np.linspace(0.1, 0.95, len(all_theta)))
cmap_N = ["#2196F3", "#FF9800", "#4CAF50"]
markers_N = ["o", "s", "D"]


def savefig(name):
    path = os.path.join(OUTDIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Ускорение vs θ  (лучший mpl для каждой точки), по одному графику на N
# ══════════════════════════════════════════════════════════════════════════════
for n in all_N:
    sub = df[df["N"] == n]
    best = sub.loc[sub.groupby("theta")["speedup"].idxmax()]

    fig, ax = plt.subplots()
    ax.plot(best["theta"], best["speedup"], "o-", color="#E53935", markersize=8)
    for _, row in best.iterrows():
        ax.annotate(f"mpl={int(row['mpl'])}", (row["theta"], row["speedup"]),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=8, ha="center", color="#555")

    ax.set_xlabel("θ (параметр открытия)")
    ax.set_ylabel("Ускорение (direct / tree)")
    ax.set_title(f"Максимальное ускорение дерева vs θ  |  N = {n:,}")
    ax.axhline(1, color="gray", ls="--", lw=1, label="нет ускорения")
    ax.legend()
    savefig(f"01_speedup_vs_theta_N{n}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Ускорение vs θ  — все N на одном графике (лучший mpl)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots()
for i, n in enumerate(all_N):
    sub = df[df["N"] == n]
    best = sub.loc[sub.groupby("theta")["speedup"].idxmax()]
    ax.plot(best["theta"], best["speedup"], marker=markers_N[i],
            color=cmap_N[i], label=f"N = {n:,}")

ax.set_xlabel("θ")
ax.set_ylabel("Ускорение")
ax.set_title("Максимальное ускорение дерева vs θ  (лучший mpl)")
ax.axhline(1, color="gray", ls="--", lw=1)
ax.legend()
savefig("02_speedup_vs_theta_allN")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Ускорение vs mpl  — для каждого N, кривые по θ
# ══════════════════════════════════════════════════════════════════════════════
for n in all_N:
    fig, ax = plt.subplots()
    sub = df[df["N"] == n]
    for j, theta in enumerate(all_theta):
        s = sub[sub["theta"] == theta].sort_values("mpl")
        ax.plot(s["mpl"], s["speedup"], marker="o", markersize=4,
                color=cmap_theta[j], label=f"θ={theta}")

    ax.set_xlabel("mpl (макс. частиц в листе)")
    ax.set_ylabel("Ускорение")
    ax.set_title(f"Ускорение vs mpl  |  N = {n:,}")
    ax.axhline(1, color="gray", ls="--", lw=1)
    ax.legend(fontsize=8, ncol=3)
    savefig(f"03_speedup_vs_mpl_N{n}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Время: прямой метод vs дерево — для каждого N (лучший mpl)
# ══════════════════════════════════════════════════════════════════════════════
for n in all_N:
    sub = df[df["N"] == n]
    best = sub.loc[sub.groupby("theta")["speedup"].idxmax()].sort_values("theta")

    fig, ax = plt.subplots()
    ax.bar(best["theta"] - 0.015, best["direct_ms"] / 1000, width=0.03,
           color="#90CAF9", label="Прямой метод", edgecolor="#1565C0")
    ax.bar(best["theta"] + 0.015, best["tree_ms"] / 1000, width=0.03,
           color="#EF9A9A", label="Дерево", edgecolor="#C62828")

    ax.set_xlabel("θ")
    ax.set_ylabel("Время (с)")
    ax.set_title(f"Время выполнения: прямой метод vs дерево  |  N = {n:,}")
    ax.legend()
    savefig(f"04_time_comparison_N{n}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Средняя относительная ошибка vs θ  (лучший mpl по скорости)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots()
for i, n in enumerate(all_N):
    sub = df[df["N"] == n]
    best = sub.loc[sub.groupby("theta")["speedup"].idxmax()].sort_values("theta")
    ax.plot(best["theta"], best["mean_rel_pct"], marker=markers_N[i],
            color=cmap_N[i], label=f"N = {n:,}")

ax.set_xlabel("θ")
ax.set_ylabel("Средняя отн. ошибка (%)")
ax.set_title("Средняя относительная ошибка vs θ  (лучший mpl по скорости)")
ax.legend()
savefig("05_mean_error_vs_theta")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Максимальная относительная ошибка vs θ
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots()
for i, n in enumerate(all_N):
    sub = df[df["N"] == n]
    best = sub.loc[sub.groupby("theta")["speedup"].idxmax()].sort_values("theta")
    ax.plot(best["theta"], best["max_rel_pct"], marker=markers_N[i],
            color=cmap_N[i], label=f"N = {n:,}")

ax.set_xlabel("θ")
ax.set_ylabel("Макс. отн. ошибка (%)")
ax.set_title("Максимальная относительная ошибка vs θ  (лучший mpl по скорости)")
ax.legend()
savefig("06_max_error_vs_theta")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Компромисс: Ускорение vs Средняя ошибка  (Парето-фронт)
# ══════════════════════════════════════════════════════════════════════════════
for n in all_N:
    fig, ax = plt.subplots()
    sub = df[df["N"] == n]

    sc = ax.scatter(sub["mean_rel_pct"], sub["speedup"],
                    c=sub["theta"], cmap="plasma", s=30, alpha=0.7,
                    edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("θ")

    # Парето-фронт: лучшие точки (больше ускорение, меньше ошибка)
    pareto = []
    sorted_pts = sub.sort_values("mean_rel_pct")
    max_speedup = -1
    for _, row in sorted_pts.iterrows():
        if row["speedup"] > max_speedup:
            pareto.append(row)
            max_speedup = row["speedup"]
    if pareto:
        pareto_df = pd.DataFrame(pareto).sort_values("mean_rel_pct")
        ax.plot(pareto_df["mean_rel_pct"], pareto_df["speedup"],
                "r--", lw=1.5, label="Парето-фронт")

    ax.set_xlabel("Средняя отн. ошибка (%)")
    ax.set_ylabel("Ускорение")
    ax.set_title(f"Ускорение vs Точность  |  N = {n:,}")
    ax.legend()
    savefig(f"07_speedup_vs_accuracy_N{n}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Средняя ошибка vs mpl — для каждого N, кривые по θ
# ══════════════════════════════════════════════════════════════════════════════
for n in all_N:
    fig, ax = plt.subplots()
    sub = df[df["N"] == n]
    for j, theta in enumerate(all_theta):
        s = sub[sub["theta"] == theta].sort_values("mpl")
        ax.plot(s["mpl"], s["mean_rel_pct"], marker="o", markersize=4,
                color=cmap_theta[j], label=f"θ={theta}")

    ax.set_xlabel("mpl (макс. частиц в листе)")
    ax.set_ylabel("Средняя отн. ошибка (%)")
    ax.set_title(f"Средняя ошибка vs mpl  |  N = {n:,}")
    ax.legend(fontsize=8, ncol=3)
    savefig(f"08_mean_error_vs_mpl_N{n}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Время дерева vs θ  — для разных mpl (выборка)
# ══════════════════════════════════════════════════════════════════════════════
selected_mpl = [1, 2, 4, 8, 16, 32]

for n in all_N:
    fig, ax = plt.subplots()
    sub = df[df["N"] == n]
    colors_mpl = plt.cm.viridis(np.linspace(0.15, 0.9, len(selected_mpl)))

    for j, mpl_val in enumerate(selected_mpl):
        s = sub[sub["mpl"] == mpl_val].sort_values("theta")
        ax.plot(s["theta"], s["tree_ms"] / 1000, marker="o", markersize=5,
                color=colors_mpl[j], label=f"mpl={mpl_val}")

    # прямой метод — горизонтальная линия
    direct_val = sub["direct_ms"].iloc[0] / 1000
    ax.axhline(direct_val, color="red", ls="--", lw=1.5,
               label=f"Прямой: {direct_val:.1f} с")

    ax.set_xlabel("θ")
    ax.set_ylabel("Время (с)")
    ax.set_title(f"Время дерева vs θ (разные mpl)  |  N = {n:,}")
    ax.legend(fontsize=9)
    savefig(f"09_tree_time_vs_theta_N{n}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. Оптимальный mpl vs θ  — для каждого N
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots()
for i, n in enumerate(all_N):
    sub = df[df["N"] == n]
    best = sub.loc[sub.groupby("theta")["speedup"].idxmax()].sort_values("theta")
    ax.plot(best["theta"], best["mpl"], marker=markers_N[i],
            color=cmap_N[i], label=f"N = {n:,}")

ax.set_xlabel("θ")
ax.set_ylabel("Оптимальный mpl")
ax.set_title("Оптимальный mpl (макс. ускорение) vs θ")
ax.legend()
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
savefig("10_optimal_mpl_vs_theta")


# ══════════════════════════════════════════════════════════════════════════════
# 11. Масштабируемость по N — время для фикс. θ
# ══════════════════════════════════════════════════════════════════════════════
selected_theta = [0.3, 0.5, 0.7, 1.0]

fig, ax = plt.subplots()
colors_th = ["#1E88E5", "#43A047", "#FB8C00", "#E53935"]

# прямой метод
direct_by_N = df.groupby("N")["direct_ms"].first().sort_index()
ax.plot(direct_by_N.index, direct_by_N.values / 1000, "k--o",
        lw=2.5, markersize=8, label="Прямой (O(N²))")

for j, theta in enumerate(selected_theta):
    sub = df[df["theta"] == theta]
    best = sub.loc[sub.groupby("N")["tree_ms"].idxmin()].sort_values("N")
    ax.plot(best["N"], best["tree_ms"] / 1000, marker="s", markersize=7,
            color=colors_th[j], label=f"Дерево θ={theta}")

ax.set_xlabel("N (число частиц)")
ax.set_ylabel("Время (с)")
ax.set_title("Масштабируемость: прямой метод vs дерево")
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
savefig("11_scalability_N")


# ══════════════════════════════════════════════════════════════════════════════
# 12. Heatmap: ускорение по (θ, mpl) — для каждого N
# ══════════════════════════════════════════════════════════════════════════════
for n in all_N:
    sub = df[df["N"] == n]
    pivot = sub.pivot_table(index="mpl", columns="theta", values="speedup")

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{t}" for t in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(m) for m in pivot.index])

    ax.set_xlabel("θ")
    ax.set_ylabel("mpl")
    ax.set_title(f"Heatmap ускорения (θ × mpl)  |  N = {n:,}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Ускорение")

    # аннотации
    for yi in range(pivot.shape[0]):
        for xi in range(pivot.shape[1]):
            val = pivot.values[yi, xi]
            color = "white" if val > pivot.values.max() * 0.7 else "black"
            ax.text(xi, yi, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color=color)

    savefig(f"12_heatmap_speedup_N{n}")


# ══════════════════════════════════════════════════════════════════════════════
# 13. Heatmap: средняя ошибка по (θ, mpl) — для каждого N
# ══════════════════════════════════════════════════════════════════════════════
for n in all_N:
    sub = df[df["N"] == n]
    pivot = sub.pivot_table(index="mpl", columns="theta", values="mean_rel_pct")

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{t}" for t in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(m) for m in pivot.index])

    ax.set_xlabel("θ")
    ax.set_ylabel("mpl")
    ax.set_title(f"Heatmap средней ошибки (%)  (θ × mpl)  |  N = {n:,}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Средняя отн. ошибка (%)")

    for yi in range(pivot.shape[0]):
        for xi in range(pivot.shape[1]):
            val = pivot.values[yi, xi]
            color = "white" if val > pivot.values.max() * 0.6 else "black"
            ax.text(xi, yi, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color=color)

    savefig(f"13_heatmap_error_N{n}")


print(f"\nГотово! Все графики в папке '{OUTDIR}/'")
