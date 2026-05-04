#!/usr/bin/env python3
"""
supplementary_eval_minimal.py
Minimal evaluation plot for Go2 locomotion policy.
Replaces multi-panel graphics with:
  (a) instantaneous linear velocity error over time
  (b) instantaneous yaw error over time

Removed:
  - composite score panel

Designed for simple debugging + reporting.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── helpers ────────────────────────────────────────────────────────────────────

def find(bundle, *keys, required=True):
    for k in keys:
        if k in bundle:
            return np.asarray(bundle[k], dtype=np.float64)
    if required:
        raise KeyError(f"None of {keys} found in rollout.")
    return None


def extract(bundle):
    cmd_lin  = find(bundle, "command_lin_vel_xy", "command_xy", "cmd_lin_vel_xy")
    meas_lin = find(bundle, "measured_lin_vel_xy", "base_lin_vel_xy", "obs_lin_vel_xy")
    cmd_yaw  = find(bundle, "command_yaw_rate", "cmd_yaw_rate", "command_ang_vel_z").ravel()
    meas_yaw = find(bundle, "measured_yaw_rate", "base_yaw_rate", "obs_yaw_rate").ravel()

    ep_id = find(bundle, "episode_id", "episode_ids", required=False)

    N = len(cmd_yaw)
    if ep_id is None:
        ep_id = np.zeros(N, dtype=int)

    ep_id = ep_id.ravel().astype(int)

    if cmd_lin.ndim == 1:
        cmd_lin = cmd_lin.reshape(-1, 2)
    if meas_lin.ndim == 1:
        meas_lin = meas_lin.reshape(-1, 2)

    # instantaneous errors
    lin_err = np.linalg.norm(cmd_lin - meas_lin, axis=1)
    yaw_err = np.abs(cmd_yaw - meas_yaw)

    return dict(
        lin_err=lin_err,
        yaw_err=yaw_err,
        N=N,
    )

# ── plotting ───────────────────────────────────────────────────────────────────

def run_all(bundle, save_path="minimal_eval.png"):
    f = extract(bundle)

    t = np.arange(f["N"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # (a) instantaneous linear velocity error
    axes[0].plot(t, f["lin_err"], linewidth=1.5)
    axes[0].set_ylabel("linear velocity error")
    axes[0].set_title("Instantaneous linear velocity tracking error")
    axes[0].set_xlabel("timestep")

    # (b) instantaneous yaw error
    axes[1].plot(t, f["yaw_err"], linewidth=1.5, color="orange")
    axes[1].set_ylabel("yaw error")
    axes[1].set_title("Instantaneous yaw tracking error")
    axes[1].set_xlabel("timestep")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-npz", type=Path, required=True)
    p.add_argument("--output-png", type=Path, default=Path("minimal_eval.png"))
    args = p.parse_args()

    bundle = dict(np.load(args.rollout_npz))
    run_all(bundle, save_path=args.output_png)
