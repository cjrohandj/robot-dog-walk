#!/usr/bin/env python3
"""
supplementary_eval_extended.py
Four-panel evaluation plot for Go2 locomotion policy:
  (a) instantaneous linear velocity error over time
  (b) instantaneous yaw error over time
  (c) instantaneous mechanical energy expenditure over time
  (d) per-foot slip speed over time
All panels share the same x-axis and episode boundary markers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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

    # energy: |tau * dq/dt| summed over joints, per timestep
    torques  = find(bundle, "joint_torques", "actuator_force")          # (T, J)
    jvel     = find(bundle, "joint_velocities", "joint_vel", "qvel")    # (T, J)
    energy   = np.sum(np.abs(torques * jvel), axis=1)                   # (T,)

    # foot slip: each foot's XY slip speed; shape (T, n_feet)
    slip     = find(bundle, "foot_slip_speed", "feet_slip_speed")       # (T, F)

    ep_id    = find(bundle, "episode_id", "episode_ids", required=False)
    N        = len(cmd_yaw)

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
        energy=energy,
        slip=slip,
        ep_id=ep_id,
        N=N,
    )


# ── plotting ───────────────────────────────────────────────────────────────────

PANEL_COLORS = {
    "lin":    "#4c72b0",
    "yaw":    "#dd8452",
    "energy": "#55a868",
    "slip":   None,       # per-foot colours defined below
}

FOOT_LABELS = ["FL", "FR", "RL", "RR"]
FOOT_COLORS = ["#c44e52", "#8172b3", "#937860", "#da8bc3"]


def _add_boundaries(ax, boundaries, *, first=False):
    """Draw episode boundary vlines; only add legend label on the first call."""
    for i, b in enumerate(boundaries):
        label = "episode boundary" if (first and i == 0) else None
        ax.axvline(b, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label=label)


def run_all(bundle, save_path="extended_eval.png"):
    f = extract(bundle)
    t = np.arange(f["N"])
    ep_id = f["ep_id"]
    boundaries = np.where(np.diff(ep_id) != 0)[0] + 1

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    # ── (a) linear velocity error ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, f["lin_err"], linewidth=1.5, color=PANEL_COLORS["lin"])
    ax.set_ylabel("‖cmd − meas‖  [m/s]")
    ax.set_title("(a)  Instantaneous linear velocity tracking error")
    _add_boundaries(ax, boundaries, first=True)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper right")

    # ── (b) yaw error ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, f["yaw_err"], linewidth=1.5, color=PANEL_COLORS["yaw"])
    ax.set_ylabel("|cmd − meas|  [rad/s]")
    ax.set_title("(b)  Instantaneous yaw tracking error")
    _add_boundaries(ax, boundaries)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # ── (c) mechanical energy ─────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, f["energy"], linewidth=1.5, color=PANEL_COLORS["energy"])
    ax.set_ylabel("Σ|τ·q̇|  [W]")
    ax.set_title("(c)  Instantaneous mechanical energy expenditure  (Σ |torque × joint vel|)")
    _add_boundaries(ax, boundaries)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # ── (d) foot slip speed ───────────────────────────────────────────────────
    ax = axes[3]
    slip = f["slip"]           # (T, n_feet)
    n_feet = slip.shape[1]
    for fi in range(n_feet):
        label = FOOT_LABELS[fi] if fi < len(FOOT_LABELS) else f"foot {fi}"
        color = FOOT_COLORS[fi] if fi < len(FOOT_COLORS) else None
        ax.plot(t, slip[:, fi], linewidth=1.2, label=label, color=color, alpha=0.85)
    ax.set_ylabel("slip speed  [m/s]")
    ax.set_title("(d)  Per-foot slip speed  (XY velocity of foot in contact)")
    ax.set_xlabel("timestep")
    _add_boundaries(ax, boundaries)
    ax.legend(loc="upper right", ncol=n_feet, fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-npz", type=Path, required=True)
    p.add_argument("--output-png",  type=Path, default=Path("extended_eval.png"))
    args = p.parse_args()

    bundle = dict(np.load(args.rollout_npz))
    run_all(bundle, save_path=args.output_png)
