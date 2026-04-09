"""
Анализ и визуализация бенчмарков:
  - benchmark_partial_ss.csv  — БЕЗ дипольной поправки к Стокслету (монополь)
  - benchmark_partial_bez.csv — С дипольной поправкой (монополь + диполь)

Генерирует 6 графиков, отражающих суть улучшения:
  1. Парето-фронт: speedup vs mean_err (ключевой trade-off)
  2. mean_err vs theta при фиксированном mpl (эффект θ на точность)
  3. mean_err vs mpl при фиксированном theta (эффект mpl на точность)
  4. Speedup vs N при фиксированных theta/mpl (масштабируемость)
  5. max_err сравнение (worst-case поведение)
  6. Сводная таблица лучших конфигураций
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Настройки ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Загрузка
df_dip = pd.read_csv(RESULTS_DIR / "benchmark_partial_ss.csv")
df_mono  = pd.read_csv(RESULTS_DIR / "benchmark_partial_bez.csv")

df_mono["method"] = "Монополь"
df_dip["method"]  = "Монополь + Диполь"
df_all = pd.concat([df_mono, df_dip], ignore_index=True)

# Цвета и стили
C_MONO = "#ff1900"   # красный
C_DIP  = "#0398fb"   # синий
ALPHA  = 0.7

# ── Фильтр: только N = 100 000 ────────────────────────────────────────────────
N_TARGET = 100_000
sub_0 = df_dip[df_dip["N"] == N_TARGET].copy()
sub_1 = df_mono[df_mono["N"] == N_TARGET].copy()
 

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})
 
# ── Фигура ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")
 
# Зона качества: err < 5%, speedup > 1
ylim_top = max(sub_0["speedup"].max(), sub_1["speedup"].max()) * 1.08
ax.fill_between(
    [0, 5], 1, ylim_top,
    color="green", alpha=0.15, zorder=0,
    label="_nolegend_"
)
ax.text(0.12, ylim_top * 0.77, "Зона качества",
        color="#0D910D", fontsize=14, va="top")
 
# Опорные линии
ax.axhline(1,  color="gray", ls="--", lw=0.9, zorder=1)
ax.axvline(5,  color="gray", ls=":",  lw=0.9, zorder=1)
ax.text(5.1,  0.15, "err = 5%",     color="gray", fontsize=8)
ax.text(0.11, 1.05, "speedup = 1×", color="gray", fontsize=8)
 
# Точки
ax.scatter(sub_0["mean_rel_pct"], sub_0["speedup"],
           c=C_MONO, marker="o", alpha=0.75, s=35,
           edgecolors="white", linewidths=0.4, zorder=3,
           label="0+1 поправка")
 
ax.scatter(sub_1["mean_rel_pct"], sub_1["speedup"],
           c=C_DIP, marker="s", alpha=0.75, s=35,
           edgecolors="white", linewidths=0.4, zorder=3,
           label="0 поправка")
 
# ── Оси ───────────────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_xlim(left=0.1)
ax.set_ylim(bottom=0, top=ylim_top)
 
ax.set_xlabel("Средняя относительная ошибка, %", fontsize=12)
ax.set_ylabel("Ускорение (раз)",        fontsize=12)
ax.set_title(
    f"Ускорение vs точность   (N = {N_TARGET:,})\n"
    "каждая точка — одна комбинация θ × leaf_size\n",
    fontsize=13, fontweight="bold"
)
 
ax.grid(True, alpha=0.25)
ax.legend(fontsize=10, loc="upper left",
          framealpha=0.85, edgecolor="#cccccc")
 
plt.tight_layout()
 
# ── Сохранение ────────────────────────────────────────────────────────────────
out_path = RESULTS_DIR / "plots" / "pareto_N100k.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Сохранено: {out_path}")
plt.show()


def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Сохранено: {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 1. ПАРЕТО-ФРОНТ: Speedup vs Mean Error (ключевой график)
# ═════════════════════════════════════════════════════════════════════════════
print("\n1. Парето-фронт: speedup vs mean_err")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Парето-фронт: ускорение vs точность\n"
             "(каждая точка — комбинация θ × mpl)")

for idx, N in enumerate(sorted(df_all["N"].unique())):
    ax = axes.flat[idx]
    for method, color, marker in [("Монополь", C_MONO, "o"),
                                   ("Монополь + Диполь", C_DIP, "s")]:
        sub = df_all[(df_all["N"] == N) & (df_all["method"] == method)]
        ax.scatter(sub["mean_rel_pct"], sub["speedup"],
                   c=color, marker=marker, alpha=ALPHA, s=30,
                   label=method, edgecolors="white", linewidth=0.3)

    # Зона "хорошо": err < 5%, speedup > 1
    ax.axhline(1, color="gray", ls="--", lw=0.8, label="speedup = 1×")
    ax.axvline(5, color="gray", ls=":", lw=0.8, label="err = 5%")
    ax.fill_between([0, 5], 1, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 10,
                    alpha=0.05, color="green")

    ax.set_xlabel("Средняя ошибка, %")
    ax.set_ylabel("Ускорение (vs direct)")
    ax.set_title(f"N = {N:,}", fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(left=1e-3)
    if idx == 0:
        ax.legend(fontsize=8, loc="upper right")

fig.tight_layout()
save(fig, "01_pareto_speedup_vs_error")


# ═════════════════════════════════════════════════════════════════════════════
# 2. MEAN ERROR vs THETA при лучшем mpl (эффект θ)
# ═════════════════════════════════════════════════════════════════════════════
print("2. Mean error vs theta")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Средняя ошибка vs θ (Barnes-Hut параметр)\n"
             "Для каждого θ выбран mpl с минимальной ошибкой", fontsize=14, fontweight="bold")

for idx, N in enumerate(sorted(df_all["N"].unique())):
    ax = axes.flat[idx]
    for method, color, marker in [("Монополь", C_MONO, "o-"),
                                   ("Монополь + Диполь", C_DIP, "s-")]:
        sub = df_all[(df_all["N"] == N) & (df_all["method"] == method)]
        # Лучший mpl для каждого theta
        best = sub.loc[sub.groupby("theta")["mean_rel_pct"].idxmin()]
        best = best.sort_values("theta")
        ax.plot(best["theta"], best["mean_rel_pct"], marker,
                color=color, label=method, markersize=6, lw=1.5)

    ax.set_xlabel("θ")
    ax.set_ylabel("Средняя ошибка, %")
    ax.set_title(f"N = {N:,}")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
    if idx == 0:
        ax.legend(fontsize=9)

    # Аннотация: теоретические наклоны
    thetas = np.array([0.1, 0.4])
    if idx == 0:
        ax.plot(thetas, 50 * thetas, "--", color=C_MONO, alpha=0.3, lw=1)
        ax.plot(thetas, 50 * thetas**2, "--", color=C_DIP, alpha=0.3, lw=1)
        ax.text(0.35, 50 * 0.35, "~θ", color=C_MONO, fontsize=8, alpha=0.5)
        ax.text(0.35, 50 * 0.35**2, "~θ²", color=C_DIP, fontsize=8, alpha=0.5)

fig.tight_layout()
save(fig, "02_mean_error_vs_theta")


# ═════════════════════════════════════════════════════════════════════════════
# 3. MEAN ERROR vs MPL при фиксированных theta
# ═════════════════════════════════════════════════════════════════════════════
print("3. Mean error vs mpl")

thetas_show = [0.1, 0.2, 0.3, 0.4]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Средняя ошибка vs mpl (макс. частиц в листе)\n"
             "N = max доступный, разные θ", fontsize=14, fontweight="bold")

N_max = df_all["N"].max()

for idx, theta in enumerate(thetas_show):
    ax = axes.flat[idx]
    for method, color, marker in [("Монополь", C_MONO, "o-"),
                                   ("Монополь + Диполь", C_DIP, "s-")]:
        sub = df_all[(df_all["N"] == N_max) &
                     (df_all["theta"] == theta) &
                     (df_all["method"] == method)]
        sub = sub.sort_values("mpl")
        ax.plot(sub["mpl"], sub["mean_rel_pct"], marker,
                color=color, label=method, markersize=5, lw=1.2)

    ax.set_xlabel("mpl")
    ax.set_ylabel("Средняя ошибка, %")
    ax.set_title(f"θ = {theta}, N = {N_max:,}")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)
    if idx == 0:
        ax.legend(fontsize=9)

fig.tight_layout()
save(fig, "03_mean_error_vs_mpl")


# ═════════════════════════════════════════════════════════════════════════════
# 4. SPEEDUP vs N (масштабируемость)
# ═════════════════════════════════════════════════════════════════════════════
print("4. Speedup vs N")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Масштабируемость: ускорение vs N\n"
             "Лучшая конфигурация (θ, mpl) с mean_err < 5%", fontsize=14, fontweight="bold")

for ax_idx, (method, color, df_src) in enumerate([
    ("Монополь", C_MONO, df_mono),
    ("Монополь + Диполь", C_DIP, df_dip),
]):
    ax = axes[ax_idx]
    good = df_src[df_src["mean_rel_pct"] < 5.0]
    if len(good) == 0:
        good = df_src[df_src["mean_rel_pct"] < 20.0]

    # Лучший speedup для каждого N
    best = good.loc[good.groupby("N")["speedup"].idxmax()]
    best = best.sort_values("N")

    bars = ax.bar(range(len(best)), best["speedup"], color=color, alpha=0.8,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels([f"{n:,}" for n in best["N"]], rotation=30)
    ax.set_xlabel("N (число частиц)")
    ax.set_ylabel("Макс. ускорение (vs direct)")
    ax.set_title(f"{method}\n(при mean_err < 5%)")
    ax.axhline(1, color="gray", ls="--", lw=0.8)

    # Аннотации: theta, mpl для каждого бара
    for i, (_, row) in enumerate(best.iterrows()):
        ax.text(i, row["speedup"] + 0.1, f"θ={row['theta']}\nmpl={int(row['mpl'])}\n"
                f"err={row['mean_rel_pct']:.1f}%",
                ha="center", va="bottom", fontsize=7)

fig.tight_layout()
save(fig, "04_speedup_vs_N")


# ═════════════════════════════════════════════════════════════════════════════
# 5. MAX ERROR: монополь vs диполь (worst-case)
# ═════════════════════════════════════════════════════════════════════════════
print("5. Max error comparison")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Максимальная ошибка (worst-case) vs θ\n"
             "Для каждого θ выбран mpl с минимальной max_err", fontsize=14, fontweight="bold")

for idx, N in enumerate(sorted(df_all["N"].unique())):
    ax = axes.flat[idx]
    for method, color, marker in [("Монополь", C_MONO, "o-"),
                                   ("Монополь + Диполь", C_DIP, "s-")]:
        sub = df_all[(df_all["N"] == N) & (df_all["method"] == method)]
        best = sub.loc[sub.groupby("theta")["max_rel_pct"].idxmin()]
        best = best.sort_values("theta")
        ax.plot(best["theta"], best["max_rel_pct"], marker,
                color=color, label=method, markersize=6, lw=1.5)

    ax.set_xlabel("θ")
    ax.set_ylabel("Макс. ошибка, %")
    ax.set_title(f"N = {N:,}")
    ax.set_yscale("log")
    if idx == 0:
        ax.legend(fontsize=9)

fig.tight_layout()
save(fig, "05_max_error_vs_theta")


# ═════════════════════════════════════════════════════════════════════════════
# 6. ВРЕМЯ ДЕРЕВА vs DIRECT (абсолютные миллисекунды)
# ═════════════════════════════════════════════════════════════════════════════
print("6. Абсолютное время: tree vs direct")

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Время вычисления: прямой метод O(N²) vs дерево O(N log N)\n"
             "Лучшая конфигурация дерева (θ, mpl) с mean_err < 5%",
             fontsize=13, fontweight="bold")

# Direct time (one per N)
for df_src, method, color, marker in [
    (df_mono, "Дерево (монополь)", C_MONO, "o"),
    (df_dip, "Дерево (+ диполь)", C_DIP, "s"),
]:
    good = df_src[df_src["mean_rel_pct"] < 5.0]
    if len(good) == 0:
        good = df_src[df_src["mean_rel_pct"] < 20.0]
    best = good.loc[good.groupby("N")["speedup"].idxmax()]
    best = best.sort_values("N")
    ax.plot(best["N"], best["tree_ms"], f"-{marker}", color=color,
            label=method, markersize=7, lw=1.5)

# Direct line (same for both)
direct = df_mono.groupby("N")["direct_ms"].first().reset_index().sort_values("N")
ax.plot(direct["N"], direct["direct_ms"], "-^", color="#2ecc71",
        label="Прямой метод O(N²)", markersize=7, lw=2)

ax.set_xlabel("N (число частиц)")
ax.set_ylabel("Время, мс")
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=10)
ax.set_title("")

fig.tight_layout()
save(fig, "06_absolute_time_vs_N")


# ═════════════════════════════════════════════════════════════════════════════
# 7. СВОДНАЯ ТАБЛИЦА
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА: лучшие конфигурации (mean_err < 5%)")
print("=" * 80)

for method_name, df_src in [("МОНОПОЛЬ (без поправки)", df_mono),
                              ("МОНОПОЛЬ + ДИПОЛЬ (с поправкой)", df_dip)]:
    print(f"\n--- {method_name} ---")
    good = df_src[df_src["mean_rel_pct"] < 5.0]
    if len(good) == 0:
        print("  Нет конфигураций с mean_err < 5%!")
        good = df_src.nsmallest(5, "mean_rel_pct")
        print("  Топ-5 по точности:")

    best = (good
            .sort_values("speedup", ascending=False)
            .groupby("N")
            .head(3)
            .sort_values(["N", "speedup"], ascending=[True, False]))

    print(f"  {'N':>7} {'θ':>5} {'mpl':>4} {'speedup':>8} {'mean%':>7} {'max%':>8} {'tree_ms':>9}")
    print(f"  {'─'*7} {'─'*5} {'─'*4} {'─'*8} {'─'*7} {'─'*8} {'─'*9}")
    for _, r in best.iterrows():
        print(f"  {int(r['N']):7,} {r['theta']:5.2f} {int(r['mpl']):4d} "
              f"{r['speedup']:7.2f}× {r['mean_rel_pct']:7.2f} {r['max_rel_pct']:8.1f} "
              f"{r['tree_ms']:9.1f}")

# Improvement summary
print("\n\n" + "=" * 80)
print("ВЫИГРЫШ ОТ ДИПОЛЬНОЙ ПОПРАВКИ")
print("=" * 80)

merged = df_mono.merge(df_dip, on=["N", "theta", "mpl"], suffixes=("_mono", "_dip"))
merged["err_ratio"] = merged["mean_rel_pct_mono"] / merged["mean_rel_pct_dip"].clip(lower=1e-10)
merged["max_err_ratio"] = merged["max_rel_pct_mono"] / merged["max_rel_pct_dip"].clip(lower=1e-10)

print(f"\n{'N':>7} {'θ':>5} {'mpl':>4} │ {'err_mono%':>9} {'err_dip%':>9} {'улучш.':>7} │ "
      f"{'max_mono%':>9} {'max_dip%':>9} {'улучш.':>7}")
print(f"{'─'*7} {'─'*5} {'─'*4} │ {'─'*9} {'─'*9} {'─'*7} │ {'─'*9} {'─'*9} {'─'*7}")

# Show a representative subset
for N in sorted(merged["N"].unique()):
    for theta in [0.1, 0.2, 0.3, 0.4]:
        sub = merged[(merged["N"] == N) & (merged["theta"] == theta)]
        if len(sub) == 0:
            continue
        # Pick mpl with best improvement
        best_row = sub.loc[sub["err_ratio"].idxmax()]
        r = best_row
        print(f"{int(r['N']):7,} {r['theta']:5.2f} {int(r['mpl']):4d} │ "
              f"{r['mean_rel_pct_mono']:9.2f} {r['mean_rel_pct_dip']:9.2f} "
              f"{r['err_ratio']:6.1f}× │ "
              f"{r['max_rel_pct_mono']:9.1f} {r['max_rel_pct_dip']:9.1f} "
              f"{r['max_err_ratio']:6.1f}×")

print(f"\nВсе графики сохранены в {OUT_DIR}/")
