#!/usr/bin/env python3
"""
supplementary_eval.py
Supplementary evaluation for the Go2 locomotion policy.
Uses seaborn + matplotlib for polished, publication-quality figures.

Colab quick-start
-----------------
    !pip -q install seaborn numpy matplotlib scipy

    # from script:
    !python supplementary_eval.py --rollout-npz rollout.npz

    # or inline:
    import numpy as np
    from supplementary_eval import run_all
    run_all(dict(np.load("rollout.npz")))
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import gaussian_kde

# ── Palette & theme ────────────────────────────────────────────────────────────

BG        = "#0f0f13"
PANEL     = "#17171d"
FG        = "#e8e6e1"
FG_DIM    = "#6b6a66"
ACCENT    = "#4ac0f2"   # sky blue
GOOD      = "#3dd68c"   # mint
BAD       = "#f05454"   # coral
WARN      = "#f5a623"   # amber
GRID      = "#ffffff0f" # near-invisible gridlines

PALETTE_SPEED = [ACCENT, WARN, BAD]
PALETTE_DIR   = sns.color_palette("husl", 8)

DIR_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
N_DIRS     = 8
SPD_EDGES  = [0.0, 0.20, 0.80, np.inf]
SPD_LABELS = ["slow\n(<0.2 m/s)", "medium\n(0.2–0.8)", "fast\n(>0.8)"]


def apply_theme():
    sns.set_theme(style="dark", font_scale=1.0)
    mpl.rcParams.update({
        "figure.facecolor":   BG,
        "axes.facecolor":     PANEL,
        "axes.edgecolor":     FG_DIM,
        "axes.labelcolor":    FG,
        "axes.titlecolor":    FG,
        "axes.titleweight":   "bold",
        "axes.titlesize":     12,
        "axes.labelsize":     10,
        "axes.grid":          True,
        "grid.color":         GRID,
        "grid.linewidth":     0.6,
        "xtick.color":        FG,
        "ytick.color":        FG,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "text.color":         FG,
        "legend.facecolor":   PANEL,
        "legend.edgecolor":   FG_DIM,
        "legend.labelcolor":  FG,
        "legend.fontsize":    9,
        "figure.dpi":         130,
        "savefig.dpi":        180,
        "savefig.facecolor":  BG,
        "font.family":        "DejaVu Sans",
        "lines.linewidth":    1.8,
        "patch.linewidth":    0.5,
    })

# ── Data helpers ───────────────────────────────────────────────────────────────

def find(bundle, *keys, required=True):
    for k in keys:
        if k in bundle:
            return np.asarray(bundle[k], dtype=np.float64)
    if required:
        raise KeyError(f"None of {keys} found in rollout.")
    return None


def angle_bin(vx, vy):
    angles    = np.arctan2(vy, vx)
    bin_width = 2 * np.pi / N_DIRS
    return np.floor((angles + np.pi / N_DIRS) / bin_width).astype(int) % N_DIRS


def speed_bin(speed):
    bins = np.zeros(len(speed), dtype=int)
    for i, (lo, hi) in enumerate(zip(SPD_EDGES[:-1], SPD_EDGES[1:])):
        bins[(speed >= lo) & (speed < hi)] = i
    return bins


def extract(bundle):
    cmd_lin  = find(bundle, "command_lin_vel_xy", "command_xy",    "cmd_lin_vel_xy")
    meas_lin = find(bundle, "measured_lin_vel_xy","base_lin_vel_xy","obs_lin_vel_xy")
    cmd_yaw  = find(bundle, "command_yaw_rate",   "cmd_yaw_rate",  "command_ang_vel_z").ravel()
    meas_yaw = find(bundle, "measured_yaw_rate",  "base_yaw_rate", "obs_yaw_rate").ravel()

    ep_id = find(bundle, "episode_id","episode_ids", required=False)
    fell  = find(bundle, "fell","fall","fall_flag","terminated_by_fall", required=False)
    torq  = find(bundle, "joint_torques","torques","tau", required=False)
    jvel  = find(bundle, "joint_velocities","joint_vel","qvel_joints", required=False)
    slip  = find(bundle, "foot_slip_speed","foot_slip","foot_slip_proxy", required=False)

    N = len(cmd_yaw)
    if ep_id is None: ep_id = np.zeros(N, dtype=int)
    if fell   is None: fell  = np.zeros(N, dtype=bool)
    ep_id = ep_id.ravel().astype(int)
    fell  = fell.ravel().astype(bool)
    if cmd_lin.ndim == 1:  cmd_lin  = cmd_lin.reshape(-1, 2)
    if meas_lin.ndim == 1: meas_lin = meas_lin.reshape(-1, 2)

    cmd_speed  = np.linalg.norm(cmd_lin,  axis=1)
    meas_speed = np.linalg.norm(meas_lin, axis=1)
    lin_err    = np.linalg.norm(cmd_lin - meas_lin, axis=1)
    yaw_err    = np.abs(cmd_yaw - meas_yaw)
    energy     = np.abs(torq * jvel).sum(axis=1) if (torq is not None and jvel is not None) else None
    if slip is not None:
    slip = np.asarray(slip)
    if slip.ndim == 2:
        # aggregate per timestep (choose one)
        slip = slip.mean(axis=1)      # average slip across feet (recommended)
        # slip = slip.max(axis=1)     # OR worst-foot slip
    else:
        slip = slip.ravel()

    return dict(
        cmd_lin=cmd_lin, meas_lin=meas_lin,
        cmd_speed=cmd_speed, meas_speed=meas_speed,
        lin_err=lin_err, yaw_err=yaw_err,
        ep_id=ep_id, fell=fell,
        dir_bin=angle_bin(cmd_lin[:,0], cmd_lin[:,1]),
        spd_bin=speed_bin(cmd_speed),
        energy=energy, slip=slip, N=N,
    )


def bin_stats(values, bins, n_bins):
    means = np.full(n_bins, np.nan)
    stds  = np.full(n_bins, np.nan)
    cnts  = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        idx = bins == b
        cnts[b] = idx.sum()
        if idx.any():
            means[b] = values[idx].mean()
            stds[b]  = values[idx].std()
    return means, stds, cnts

# ── Panel helpers ──────────────────────────────────────────────────────────────

def _spine_style(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(FG_DIM)
    ax.tick_params(colors=FG, which="both")


def _errbar(ax, x, y, yerr, color, width=0.65, alpha=0.85):
    bars = ax.bar(x, y, color=color, width=width, alpha=alpha,
                  edgecolor="none", zorder=3)
    ax.errorbar(x, y, yerr=yerr, fmt="none", color=FG_DIM,
                linewidth=1.2, capsize=4, capthick=1, zorder=4)
    return bars

# ── 1. Direction tracking ──────────────────────────────────────────────────────

def plot_direction_tracking(ax_lin, ax_yaw, f):
    lin_m, lin_s, cnts = bin_stats(f["lin_err"], f["dir_bin"], N_DIRS)
    yaw_m, yaw_s, _    = bin_stats(f["yaw_err"], f["dir_bin"], N_DIRS)

    cmap  = plt.cm.RdYlGn_r
    valid = lin_m[~np.isnan(lin_m)]
    norm  = mpl.colors.Normalize(vmin=valid.min(), vmax=valid.max())

    x = np.arange(N_DIRS)
    colors = [cmap(norm(v)) if not np.isnan(v) else FG_DIM for v in lin_m]

    _errbar(ax_lin, x, np.nan_to_num(lin_m), np.nan_to_num(lin_s), colors)
    for i, (m, n) in enumerate(zip(lin_m, cnts)):
        if not np.isnan(m):
            ax_lin.text(i, 0.002, f"n={n}", ha="center", va="bottom",
                        fontsize=7.5, color=FG_DIM)

    ax_lin.set_xticks(x); ax_lin.set_xticklabels(DIR_LABELS)
    ax_lin.set_ylabel("linear error (m/s)")
    ax_lin.set_title("Tracking error by command direction")
    _spine_style(ax_lin)

    ycolors = [cmap(norm(v)) if not np.isnan(v) else FG_DIM for v in yaw_m]
    _errbar(ax_yaw, x, np.nan_to_num(yaw_m), np.nan_to_num(yaw_s), ycolors)
    ax_yaw.set_xticks(x); ax_yaw.set_xticklabels(DIR_LABELS)
    ax_yaw.set_ylabel("yaw rate error (rad/s)")
    ax_yaw.set_title("Yaw rate error by direction")
    _spine_style(ax_yaw)

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=[ax_lin, ax_yaw], pad=0.02, shrink=0.85, aspect=25)
    cb.set_label("mean error (m/s)", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
    cb.outline.set_edgecolor(FG_DIM)

# ── 2. Speed regime analysis ───────────────────────────────────────────────────

def _kde_violin(ax, data, pos, color, width=0.35):
    """Draw a single KDE violin at position `pos`."""
    if len(data) < 3:
        ax.scatter([pos], [np.mean(data)] if len(data) else [0],
                   color=color, s=40, zorder=5)
        return
    kde  = gaussian_kde(data, bw_method="scott")
    lo, hi = data.min(), data.max()
    ys   = np.linspace(lo, hi, 300)
    dens = kde(ys)
    dens = dens / dens.max() * width
    ax.fill_betweenx(ys, pos - dens, pos + dens, color=color, alpha=0.30)
    ax.plot(pos + dens, ys, color=color, linewidth=1.5, alpha=0.85)
    ax.plot(pos - dens, ys, color=color, linewidth=1.5, alpha=0.85)
    med = np.median(data)
    ax.plot([pos - width * 0.6, pos + width * 0.6], [med, med],
            color="white", linewidth=2.0, zorder=5, solid_capstyle="round")


def plot_magnitude_analysis(ax_lin, ax_yaw, ax_ratio, f):
    for ax, key, ylabel, title in [
        (ax_lin, "lin_err", "linear error (m/s)",  "Linear tracking vs speed"),
        (ax_yaw, "yaw_err", "yaw rate error (rad/s)", "Yaw tracking vs speed"),
    ]:
        for b, (color, lbl) in enumerate(zip(PALETTE_SPEED, SPD_LABELS)):
            vals = f[key][f["spd_bin"] == b]
            if len(vals):
                _kde_violin(ax, vals, b, color)
        ax.set_xticks(range(len(SPD_LABELS)))
        ax.set_xticklabels(SPD_LABELS, fontsize=8.5)
        ax.set_ylabel(ylabel); ax.set_title(title)
        _spine_style(ax)

    # ratio bars
    ratio_m, ratio_s = [], []
    for b in range(len(SPD_LABELS)):
        mask = f["spd_bin"] == b
        nz   = mask & (f["cmd_speed"] > 0.05)
        if nz.any():
            r = f["meas_speed"][nz] / f["cmd_speed"][nz]
            ratio_m.append(r.mean()); ratio_s.append(r.std())
        else:
            ratio_m.append(np.nan); ratio_s.append(0)

    _errbar(ax_ratio, range(len(SPD_LABELS)),
            np.nan_to_num(ratio_m), np.nan_to_num(ratio_s),
            PALETTE_SPEED, width=0.5)
    ax_ratio.axhline(1.0, color=FG, linewidth=1.2, linestyle="--",
                     alpha=0.7, label="ideal (1.0)", zorder=5)
    ax_ratio.set_xticks(range(len(SPD_LABELS)))
    ax_ratio.set_xticklabels(SPD_LABELS, fontsize=8.5)
    ax_ratio.set_ylabel("measured / commanded speed")
    ax_ratio.set_title("Speed tracking ratio")
    ax_ratio.legend()
    _spine_style(ax_ratio)

# ── 3. Stability ───────────────────────────────────────────────────────────────

def plot_stability(ax_tl, ax_surv, ax_bar, f):
    ep_ids  = np.unique(f["ep_id"])
    ep_len  = np.array([(f["ep_id"] == e).sum()        for e in ep_ids])
    ep_fell = np.array([f["fell"][f["ep_id"] == e].any() for e in ep_ids])

    # timeline
    colors = [BAD if fe else GOOD for fe in ep_fell]
    ax_tl.bar(range(len(ep_ids)), ep_len, color=colors,
              alpha=0.82, edgecolor="none", zorder=3)
    for i, (eid, fe) in enumerate(zip(ep_ids, ep_fell)):
        if fe:
            mask = f["ep_id"] == eid
            fs   = int(f["fell"][mask].argmax())
            ax_tl.scatter(i, fs, marker="x", color="white",
                          s=50, linewidths=1.5, zorder=6)
    ax_tl.set_xlabel("episode index"); ax_tl.set_ylabel("steps")
    ax_tl.set_title("Episode length & fall timing")
    ax_tl.legend(handles=[
        mpatches.Patch(color=GOOD, label="survived"),
        mpatches.Patch(color=BAD,  label="fell"),
        Line2D([0],[0], marker="x", color="white", linestyle="", markersize=7,
               label="fall step"),
    ], loc="upper right")
    _spine_style(ax_tl)

    # survival curve
    max_len = int(ep_len.max()) if len(ep_len) else 1
    ts      = np.arange(max_len + 1)
    alive   = np.zeros(max_len + 1)
    for t in ts:
        active = ep_len >= t
        if not active.any():
            alive[t] = 0.0; continue
        not_fallen = []
        for i, eid in enumerate(ep_ids):
            if not active[i]: continue
            mask = f["ep_id"] == eid
            fell_so_far = f["fell"][mask][:t].any() if t > 0 else False
            not_fallen.append(not fell_so_far)
        alive[t] = np.mean(not_fallen)

    ax_surv.plot(ts, alive * 100, color=ACCENT, linewidth=2.2, zorder=4)
    ax_surv.fill_between(ts, alive * 100, alpha=0.15, color=ACCENT, zorder=3)
    ax_surv.axhline(100, color=FG_DIM, linewidth=0.8, linestyle="--")
    ax_surv.set_xlabel("step within episode")
    ax_surv.set_ylabel("% episodes alive")
    ax_surv.set_title("Survival curve")
    ax_surv.set_ylim(0, 108)
    _spine_style(ax_surv)

    # summary bar
    fall_pct = ep_fell.mean() * 100
    surv_pct = 100 - fall_pct
    bars = ax_bar.bar([0, 1], [surv_pct, fall_pct],
                      color=[GOOD, BAD], alpha=0.85, width=0.5, edgecolor="none")
    for bar, v in zip(bars, [surv_pct, fall_pct]):
        ax_bar.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color=FG)
    ax_bar.set_xticks([0, 1]); ax_bar.set_xticklabels(["survived", "fell"])
    ax_bar.set_ylabel("% of episodes")
    ax_bar.set_title(f"Fall rate: {fall_pct:.1f}%")
    ax_bar.set_ylim(0, 118)
    _spine_style(ax_bar)

# ── 4. Polar heatmap ───────────────────────────────────────────────────────────

def plot_polar(ax, f):
    lin_m, _, _ = bin_stats(f["lin_err"], f["dir_bin"], N_DIRS)
    cmap  = plt.cm.RdYlGn_r
    valid = lin_m[~np.isnan(lin_m)]
    norm  = mpl.colors.Normalize(vmin=valid.min(), vmax=valid.max())

    angles    = np.linspace(0, 2*np.pi, N_DIRS, endpoint=False)
    width     = 2*np.pi / N_DIRS * 0.88

    for b, (θ, m) in enumerate(zip(angles, lin_m)):
        if np.isnan(m): continue
        c = cmap(norm(m))
        ax.bar(θ, m, width=width, bottom=0, color=c, alpha=0.85,
               edgecolor=BG, linewidth=1.0)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(np.degrees(angles), DIR_LABELS,
                      fontsize=10, fontweight="bold", color=FG)
    ax.set_facecolor(PANEL)
    ax.spines["polar"].set_color(FG_DIM)
    ax.tick_params(colors=FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.set_title("Tracking error\nby direction", pad=16)
    ax.grid(color=GRID, linewidth=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.12, shrink=0.72, aspect=20)
    cb.set_label("error (m/s)", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
    cb.outline.set_edgecolor(FG_DIM)

# ── 5. Energy & foot-slip ─────────────────────────────────────────────────────

def plot_efficiency(ax_energy, ax_slip, f):
    if f["energy"] is not None:
        em, es, _ = bin_stats(f["energy"], f["dir_bin"], N_DIRS)
        _errbar(ax_energy, range(N_DIRS), np.nan_to_num(em), np.nan_to_num(es),
                [c for c in PALETTE_DIR])
        ax_energy.set_xticks(range(N_DIRS)); ax_energy.set_xticklabels(DIR_LABELS)
        ax_energy.set_ylabel("energy proxy  |τ·q̇| (W)")
        ax_energy.set_title("Energy consumption by direction")
    else:
        ax_energy.text(0.5, 0.5, "torques / joint_vel\nnot in rollout",
                       ha="center", va="center", transform=ax_energy.transAxes,
                       fontsize=11, color=FG_DIM)
        ax_energy.set_title("Energy by direction")
    _spine_style(ax_energy)

    if f["slip"] is not None:
        sm, ss, _ = bin_stats(f["slip"], f["spd_bin"], len(SPD_LABELS))
        _errbar(ax_slip, range(len(SPD_LABELS)), np.nan_to_num(sm), np.nan_to_num(ss),
                PALETTE_SPEED, width=0.5)
        ax_slip.set_xticks(range(len(SPD_LABELS)))
        ax_slip.set_xticklabels(SPD_LABELS, fontsize=8.5)
        ax_slip.set_ylabel("foot slip speed")
        ax_slip.set_title("Foot slip by speed regime")
    else:
        ax_slip.text(0.5, 0.5, "foot_slip\nnot in rollout",
                     ha="center", va="center", transform=ax_slip.transAxes,
                     fontsize=11, color=FG_DIM)
        ax_slip.set_title("Foot slip by speed")
    _spine_style(ax_slip)

# ── 6. Per-episode scatter ────────────────────────────────────────────────────

def plot_episode_scatter(ax, f):
    ep_ids = np.unique(f["ep_id"])
    spds   = np.array([f["cmd_speed"][f["ep_id"] == e].mean() for e in ep_ids])
    errs   = np.array([f["lin_err"][f["ep_id"] == e].mean()   for e in ep_ids])
    fells  = np.array([f["fell"][f["ep_id"] == e].any()       for e in ep_ids])

    surv = ~fells
    sns.scatterplot(x=spds[surv], y=errs[surv], ax=ax, color=GOOD,
                    s=55, alpha=0.80, edgecolor="none", label="survived", zorder=4)
    sns.scatterplot(x=spds[fells], y=errs[fells], ax=ax, color=BAD,
                    s=55, alpha=0.85, edgecolor="none", marker="^",
                    label="fell", zorder=5)

    if len(spds) > 2:
        z  = np.polyfit(spds, errs, 1)
        xs = np.linspace(spds.min(), spds.max(), 200)
        ax.plot(xs, np.polyval(z, xs), color=FG_DIM, linewidth=1.5,
                linestyle="--", alpha=0.7, label="trend")

    ax.set_xlabel("mean command speed (m/s)")
    ax.set_ylabel("mean linear tracking error (m/s)")
    ax.set_title("Per-episode: tracking error vs command speed")
    ax.legend()
    _spine_style(ax)

# ── Text summary ───────────────────────────────────────────────────────────────

def print_summary(f):
    ep_ids  = np.unique(f["ep_id"])
    ep_fell = np.array([f["fell"][f["ep_id"] == e].any() for e in ep_ids])
    print("\n" + "═"*62)
    print("  SUPPLEMENTARY EVALUATION — Go2 locomotion policy")
    print("═"*62)
    print(f"  Steps: {f['N']:,}   Episodes: {len(ep_ids)}   "
          f"Fall rate: {ep_fell.mean()*100:.1f}%")
    print(f"  Linear error : {f['lin_err'].mean():.4f} ± {f['lin_err'].std():.4f} m/s")
    print(f"  Yaw error    : {f['yaw_err'].mean():.4f} ± {f['yaw_err'].std():.4f} rad/s")

    print("\n  By direction:")
    lin_m, _, _ = bin_stats(f["lin_err"], f["dir_bin"], N_DIRS)
    for b, lbl in enumerate(DIR_LABELS):
        n = (f["dir_bin"] == b).sum()
        if np.isnan(lin_m[b]):
            print(f"    {lbl:3s}: no data")
        else:
            print(f"    {lbl:3s}: {lin_m[b]:.4f} m/s  (n={n})")

    print("\n  By speed regime:")
    for b, lbl in enumerate(SPD_LABELS):
        mask = f["spd_bin"] == b
        label = lbl.replace("\n", " ")
        if mask.any():
            print(f"    {label:22s}: {f['lin_err'][mask].mean():.4f} m/s  (n={mask.sum()})")
        else:
            print(f"    {label:22s}: no data")

    valid = lin_m[~np.isnan(lin_m)]
    if len(valid):
        best  = DIR_LABELS[int(np.nanargmin(lin_m))]
        worst = DIR_LABELS[int(np.nanargmax(lin_m))]
        print(f"\n  Best  direction: {best}  ({np.nanmin(lin_m):.4f} m/s)")
        print(f"  Worst direction: {worst}  ({np.nanmax(lin_m):.4f} m/s)")
    print("═"*62 + "\n")

# ── Main figure ────────────────────────────────────────────────────────────────

def run_all(bundle: dict, save_path: str | Path = "supplementary_eval.png"):
    apply_theme()
    f = extract(bundle)
    print_summary(f)

    fig = plt.figure(figsize=(18, 26))
    fig.patch.set_facecolor(BG)

    # Super-title
    fig.text(0.5, 0.992, "Go2 Locomotion — Supplementary Policy Evaluation",
             ha="center", va="top", fontsize=17, fontweight="bold", color=FG)

    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.50, wspace=0.35,
                           top=0.975, bottom=0.035, left=0.07, right=0.96)

    # Row 0: direction tracking (spans 2 cols + 1 col)
    ax_lin = fig.add_subplot(gs[0, :2])
    ax_yaw = fig.add_subplot(gs[0, 2])
    plot_direction_tracking(ax_lin, ax_yaw, f)

    # Row 1: speed regime (3 cols)
    ax_m0 = fig.add_subplot(gs[1, 0])
    ax_m1 = fig.add_subplot(gs[1, 1])
    ax_m2 = fig.add_subplot(gs[1, 2])
    plot_magnitude_analysis(ax_m0, ax_m1, ax_m2, f)

    # Row 2: stability timeline (spans 2) + survival
    ax_tl   = fig.add_subplot(gs[2, :2])
    ax_surv = fig.add_subplot(gs[2, 2])
    # Row 3 col 2: fall summary bar — share with efficiency
    ax_bar  = fig.add_subplot(gs[3, 2])
    plot_stability(ax_tl, ax_surv, ax_bar, f)

    # Row 3: polar + energy + (fall bar already placed)
    ax_pol     = fig.add_subplot(gs[3, 0], projection="polar")
    ax_energy  = fig.add_subplot(gs[3, 1])
    plot_polar(ax_pol, f)
    plot_efficiency(ax_energy, ax_bar, f)   # slip reuses ax_bar slot

    # Row 4: scatter full width
    ax_scat = fig.add_subplot(gs[4, :])
    plot_episode_scatter(ax_scat, f)

    # Watermark
    fig.text(0.97, 0.012, "supplementary_eval.py · seaborn + matplotlib",
             ha="right", va="bottom", fontsize=8, color=FG_DIM, style="italic")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout-npz", type=Path, required=True)
    p.add_argument("--output-png",  type=Path, default=Path("supplementary_eval.png"))
    args = p.parse_args()
    print(f"Loading {args.rollout_npz} ...")
    bundle = dict(np.load(args.rollout_npz))
    print(f"Keys: {list(bundle.keys())}")
    run_all(bundle, save_path=args.output_png)
