"""
Графики сходимости псевдо-периодического метода.

График 1: err_F vs K (линии для каждого N)
График 2: err_V vs K (линии для каждого N)
График 3: err_F(K=K_max) vs N
График 4: scatter F_ref vs F_pseudo для наибольшего N
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_all(results_dir: str = "convergence_test/results"):
    """Строит все 4 графика из сохранённых результатов."""
    csv_path = os.path.join(results_dir, "convergence_results.csv")
    df = pd.DataFrame(pd.read_csv(csv_path))

    N_list = sorted(df['N'].unique())
    K_max = df['K'].max()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── График 1: err_F vs K ──
    ax = axes[0, 0]
    for N in N_list:
        sub = df[df['N'] == N].sort_values('K')
        ax.semilogy(sub['K'], sub['err_F_direct'], 'o-', label=f'N={N}')
    ax.set_xlabel('K (число копий по оси)')
    ax.set_ylabel('err_F (относительная RMSE)')
    ax.set_title('Сходимость ошибки сил')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── График 2: err_V vs K ──
    ax = axes[0, 1]
    for N in N_list:
        sub = df[df['N'] == N].sort_values('K')
        ax.semilogy(sub['K'], sub['err_V_direct'], 's-', label=f'N={N}')
    ax.set_xlabel('K (число копий по оси)')
    ax.set_ylabel('err_V (относительная RMSE)')
    ax.set_title('Сходимость ошибки скоростей')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── График 3: err_F(K=K_max) vs N ──
    ax = axes[1, 0]
    sub = df[df['K'] == K_max].sort_values('N')
    ax.semilogy(sub['N'], sub['err_F_direct'], 'o-', color='tab:blue', label='err_F')
    ax.semilogy(sub['N'], sub['err_V_direct'], 's-', color='tab:orange', label='err_V')
    ax.set_xlabel('N (число частиц)')
    ax.set_ylabel(f'Ошибка при K={K_max}')
    ax.set_title(f'Ошибка vs число частиц (K={K_max})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── График 4: scatter F_ref vs F_pseudo ──
    ax = axes[1, 1]
    N_max = max(N_list)
    f_ref_path = os.path.join(results_dir, f"F_ref_N{N_max}_K{K_max}.npy")
    f_pseudo_path = os.path.join(results_dir, f"F_pseudo_N{N_max}.npy")

    if os.path.exists(f_ref_path) and os.path.exists(f_pseudo_path):
        F_ref = np.load(f_ref_path)
        F_pseudo = np.load(f_pseudo_path)
        # Показываем z-компоненту (основная)
        ax.scatter(F_ref[:, 2], F_pseudo[:, 2], s=3, alpha=0.5, color='tab:blue')
        lim = max(np.abs(F_ref[:, 2]).max(), np.abs(F_pseudo[:, 2]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(f'F_ref_z (K={K_max})')
        ax.set_ylabel('F_pseudo_z')
        ax.set_title(f'Корреляция Fz: N={N_max}, K={K_max}')
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, 'Данные не найдены', ha='center', va='center',
                transform=ax.transAxes)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "convergence_plots.png")
    plt.savefig(out_path, dpi=150)
    print(f"Графики сохранены в {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_all()
