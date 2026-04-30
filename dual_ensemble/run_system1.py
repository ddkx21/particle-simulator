"""
Запуск системы 1 (мелкие капли).

  python -m dual_ensemble.run_system1                  # свежий запуск
  python -m dual_ensemble.run_system1 --continue       # продолжить с checkpoint
  python -m dual_ensemble.run_system1 --t-stop 300     # переопределить t_stop
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dual_ensemble.run_common import (
    RESULTS_DIR,
    compute_system_params,
    load_config,
    load_elapsed,
    load_snapshots,
    make_snapshot_times,
    merge_snapshots,
    run_single_system,
    save_snapshots,
)

SYS_PREFIX = "sys1"
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "sys1_checkpoint.npz")
SNAPSHOTS_FILE = os.path.join(RESULTS_DIR, "sys1_snapshots.npz")


def main() -> None:
    import taichi as ti

    parser = argparse.ArgumentParser(description="Запуск системы 1")
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="Продолжить с последнего checkpoint")
    parser.add_argument("--t-stop", type=float, default=None,
                        help="Переопределить t_stop из конфига")
    args = parser.parse_args()

    cfg = load_config()
    tai = cfg["taichi"]
    ti.init(arch=getattr(ti, tai["arch"]),
            cpu_max_num_threads=tai["cpu_max_num_threads"],
            default_fp=ti.f64)

    sim = cfg["simulation"]
    sys_cfg = cfg["system1"]
    radii_range = np.array(sys_cfg["radii_range"])
    t_stop = args.t_stop if args.t_stop is not None else sys_cfg["t_stop"]

    N1, _, box_size = compute_system_params(cfg)

    snapshot_interval = sim["snapshot_interval"]
    snapshot_times = make_snapshot_times(t_stop, snapshot_interval)

    initial_state_path = None
    if args.continue_run:
        if os.path.isfile(CHECKPOINT_FILE):
            initial_state_path = CHECKPOINT_FILE
            print(f"Продолжение из checkpoint: {CHECKPOINT_FILE}")
        else:
            print(f"Checkpoint не найден ({CHECKPOINT_FILE}), запуск с нуля")

    print(f"\nСистема 1: N={N1}, radii=[{radii_range[0]*1e6:.1f}, {radii_range[1]*1e6:.1f}] мкм")
    print(f"L={box_size*1e6:.1f} мкм, t_stop={t_stop}, continue={args.continue_run}")

    result = run_single_system(
        label=sys_cfg["label"],
        N=N1,
        radii_range=radii_range,
        box_size=box_size,
        boundary_mode=sys_cfg["boundary_mode"],
        t_stop=t_stop,
        snapshot_times=snapshot_times,
        cfg=cfg,
        initial_state_path=initial_state_path,
    )

    result["final_state"].save(CHECKPOINT_FILE)
    print(f"Checkpoint сохранён: {CHECKPOINT_FILE}")

    if args.continue_run and os.path.isfile(SNAPSHOTS_FILE):
        old_radii, old_times, old_acc = load_snapshots(SNAPSHOTS_FILE)
        old_elapsed = load_elapsed(SNAPSHOTS_FILE)
        merged_radii, merged_times, new_acc = merge_snapshots(
            old_radii, old_times, old_acc,
            result["radii_snapshots"], snapshot_times, t_stop,
        )
    else:
        if args.continue_run:
            print(f"ВНИМАНИЕ: файл snapshots не найден ({SNAPSHOTS_FILE}), история сброшена")
        merged_radii = result["radii_snapshots"]
        merged_times = snapshot_times
        new_acc = t_stop
        old_elapsed = 0.0

    total_elapsed = old_elapsed + result["elapsed_time"]

    save_snapshots(
        path=SNAPSHOTS_FILE,
        prefix=SYS_PREFIX,
        radii=merged_radii,
        times=merged_times,
        accumulated_time=new_acc,
        label=sys_cfg["label"],
        radii_range=radii_range,
        box_size=box_size,
        N_initial=result["N_initial"],
        elapsed=total_elapsed,
    )

    if args.continue_run and old_elapsed > 0:
        print(f"Время расчёта: {result['elapsed_time']:.1f} сек (этот сегмент) + "
              f"{old_elapsed:.1f} сек (накоплено) = {total_elapsed:.1f} сек суммарно")

    print(f"\nДля продолжения: python -m dual_ensemble.run_system1 --continue")
    print(f"Для графиков: python -m dual_ensemble.plot_dual_ensemble")


if __name__ == "__main__":
    main()
