#!/usr/bin/env python3
"""
supplementary_eval_minimal.py
Minimal evaluation plot for Go2 locomotion policy.
Replaces multi-panel graphics with:
  (a) linear error over time
  (b) yaw error over time
  (c) composite score:
      1 - mean( sqrt( sum_t (direction_tracking_error^2 per episode) ) )

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

    lin_err = np.linalg.norm(cmd_lin - meas_lin, axis=1)
    yaw_err = np.abs(cmd_yaw - meas_yaw)

    return dict(
        lin_err=lin_err,
        yaw_err=yaw_err,
        ep_id=ep_id,
        cmd_lin=cmd_lin,
        N=N,
    )

# ── composite score ────────────────────────────────────────────────────────────

def composite_score(f):
    ep_ids = np.unique(f["ep_id"])
    ep_vals = []

    for e in ep_ids:
        mask = f["ep_id"] == e
        cmd = f["cmd_lin"][mask]

        # direction tracking error per step
        err = np.linalg.norm(cmd, axis=1)

        ep_vals.append(np.sqrt(np.sum(err ** 2)))

    ep_vals = np.array(ep_vals)

    if len(ep_vals) == 0:
        return 0.0

    return 1.0 - float(np.mean(ep_vals))

# ── plotting ───────────────────────────────────────────────────────────────────

def run_all(bundle, save_path="minimal_eval.png"):
    f = extract(bundle)

    t = np.arange(f["N"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # (a) linear error over time
    axes[0].plot(t, f["lin_err"], linewidth=1.5)
    axes[0].set_ylabel("linear error")
    axes[0].set_title("Linear tracking error over time")

    # (b) yaw error over time
    axes[1].plot(t, f["yaw_err"], linewidth=1.5, color="orange")
    axes[1].set_ylabel("yaw error")
    axes[1].set_title("Yaw tracking error over time")

    # (c) composite score
    score = composite_score(f)
    axes[2].bar([0], [score])
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"Composite score: {score:.4f}")
    axes[2].set_xticks([])

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
