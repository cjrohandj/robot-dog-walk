#!/usr/bin/env python3
"""Evaluate a checkpoint on x-only, y-only, and yaw-only command segments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    save_json,
    set_runtime_env,
)
from test_policy import load_policy_with_workaround


ROOT = Path(__file__).resolve().parent
SEGMENT_LABELS = ("x only", "y only", "yaw only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to a PPO checkpoint directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the course config JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "per_direction_eval",
        help="Directory for the plot, rollout bundle, and summary JSON.",
    )
    parser.add_argument(
        "--stage-name",
        choices=["stage_1", "stage_2"],
        default="stage_2",
        help="Which stage config to use when building the eval environment.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=4.0,
        help="Seconds to hold each of the three commands.",
    )
    parser.add_argument(
        "--command-scale",
        type=float,
        default=0.6,
        help="Fraction of the configured safe command max to use.",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Force JAX onto CPU.")
    return parser.parse_args()


def _force_command(state: Any, command: np.ndarray, jax: Any) -> Any:
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = np.int32(10**9)
    return state


def _build_per_direction_commands(config: dict[str, Any], command_scale: float) -> list[np.ndarray]:
    safe_ranges = config["public_eval"]["safe_command_ranges"]
    _, vx_max = map(float, safe_ranges["vx"])
    _, vy_max = map(float, safe_ranges["vy"])
    _, yaw_max = map(float, safe_ranges["yaw"])
    scale = float(command_scale)
    return [
        np.asarray([scale * vx_max, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, scale * vy_max, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, scale * yaw_max], dtype=np.float32),
    ]


def _looks_like_checkpoint(path: Path) -> bool:
    return (path / "ppo_network_config.json").is_file()


def _resolve_checkpoint_dir(path: Path, stage_name: str) -> Path:
    path = path.resolve()
    if _looks_like_checkpoint(path):
        return path

    candidates = [
        path / "best_checkpoint",
        path / stage_name / "best_checkpoint",
        path / stage_name / "checkpoints",
        path / "checkpoints",
    ]

    for candidate in candidates[:2]:
        if _looks_like_checkpoint(candidate):
            return candidate

    for checkpoint_root in candidates[2:]:
        if not checkpoint_root.is_dir():
            continue
        numbered = []
        for child in checkpoint_root.iterdir():
            if not child.is_dir() or not _looks_like_checkpoint(child):
                continue
            try:
                numbered.append((int(child.name), child))
            except ValueError:
                continue
        if numbered:
            numbered.sort(key=lambda item: item[0])
            return numbered[-1][1]

    discovered = sorted(path.rglob("ppo_network_config.json"))
    if discovered:
        preferred = [item for item in discovered if "best_checkpoint" in item.parts]
        return (preferred[0] if preferred else discovered[-1]).parent

    raise FileNotFoundError(
        f"No PPO checkpoint found under {path}. "
        "Pass --checkpoint-dir to a directory that contains ppo_network_config.json, or pass the training "
        "run directory after training has produced checkpoints. Common examples: "
        "artifacts/YOUR_RUN_NAME/best_checkpoint or artifacts/YOUR_RUN_NAME/stage_2/checkpoints/000000123456."
    )


def _plot_rollout(bundle: dict[str, np.ndarray], summary: dict[str, Any], output_png: Path) -> None:
    time = bundle["time_seconds"]
    command_xy = bundle["command_lin_vel_xy"]
    measured_xy = bundle["measured_lin_vel_xy"]
    command_yaw = bundle["command_yaw_rate"]
    measured_yaw = bundle["measured_yaw_rate"]

    lin_err = np.linalg.norm(command_xy - measured_xy, axis=1)
    yaw_err = np.abs(command_yaw - measured_yaw)
    boundaries = summary["command_change_times_seconds"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time, command_xy[:, 0], color="#4c72b0", linestyle="--", label="cmd vx")
    axes[0].plot(time, measured_xy[:, 0], color="#4c72b0", label="measured vx")
    axes[0].set_ylabel("vx [m/s]")

    axes[1].plot(time, command_xy[:, 1], color="#55a868", linestyle="--", label="cmd vy")
    axes[1].plot(time, measured_xy[:, 1], color="#55a868", label="measured vy")
    axes[1].set_ylabel("vy [m/s]")

    axes[2].plot(time, command_yaw, color="#dd8452", linestyle="--", label="cmd yaw")
    axes[2].plot(time, measured_yaw, color="#dd8452", label="measured yaw")
    axes[2].set_ylabel("yaw rate [rad/s]")

    axes[3].plot(time, lin_err, color="#4c72b0", label="linear velocity error")
    axes[3].plot(time, yaw_err, color="#dd8452", label="yaw error")
    axes[3].set_ylabel("error")
    axes[3].set_xlabel("time [s]")

    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    ymax = axes[0].get_ylim()[1]
    for label, start_time in zip(summary["segment_labels"], summary["segment_start_times_seconds"]):
        axes[0].text(start_time + 0.05, ymax, label, va="top", fontsize=9)

    fig.suptitle("Per-direction command tracking test")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Saved {output_png}")


def _segment_metrics(
    command_xy: np.ndarray,
    measured_xy: np.ndarray,
    command_yaw: np.ndarray,
    measured_yaw: np.ndarray,
    segment_ids: np.ndarray,
) -> list[dict[str, float | str]]:
    results = []
    for segment_idx, label in enumerate(SEGMENT_LABELS):
        mask = segment_ids == segment_idx
        if not np.any(mask):
            continue
        results.append(
            {
                "segment": label,
                "mean_linear_velocity_error": float(np.mean(np.linalg.norm(command_xy[mask] - measured_xy[mask], axis=1))),
                "mean_yaw_error": float(np.mean(np.abs(command_yaw[mask] - measured_yaw[mask]))),
                "mean_vx_error": float(np.mean(np.abs(command_xy[mask, 0] - measured_xy[mask, 0]))),
                "mean_vy_error": float(np.mean(np.abs(command_xy[mask, 1] - measured_xy[mask, 1]))),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    config["runtime_overrides"] = {}
    if args.force_cpu:
        config["force_cpu"] = True
        config["runtime_overrides"]["force_cpu"] = True

    force_cpu = bool(config.get("force_cpu")) or bool(config.get("runtime_overrides", {}).get("force_cpu"))
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = lazy_import_stack()
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    jax = stack["jax"]

    env_name = config["environment_name"]
    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, config["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, config, args.stage_name)

    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))
    segment_steps = max(1, int(round(float(args.segment_seconds) / float(env.dt))))
    commands = _build_per_direction_commands(config, args.command_scale)
    total_steps = segment_steps * len(commands)

    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir, args.stage_name)
    policy = load_policy_with_workaround(checkpoint_dir, deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)
    reset_fn = env.reset if force_cpu else jax.jit(env.reset)
    step_fn = env.step if force_cpu else jax.jit(env.step)

    rng = jax.random.PRNGKey(int(config["seed"]) + 789)
    state = reset_fn(rng)
    state = _force_command(state, commands[0], jax)

    time_seconds = []
    segment_ids = []
    command_xy = []
    measured_xy = []
    command_yaw = []
    measured_yaw = []
    fell = []

    done_step = None
    for step_idx in range(total_steps):
        segment_idx = min(len(commands) - 1, step_idx // segment_steps)
        command = commands[segment_idx]
        state = _force_command(state, command, jax)

        rng, act_key = jax.random.split(rng)
        action, _ = policy(state.obs, act_key)
        state = step_fn(state, action)
        state = _force_command(state, command, jax)

        time_seconds.append((step_idx + 1) * float(env.dt))
        segment_ids.append(segment_idx)
        command_xy.append(command[:2])
        measured_xy.append(np.asarray(env.get_local_linvel(state.data)[:2], dtype=np.float32))
        command_yaw.append(command[2])
        measured_yaw.append(np.asarray(env.get_gyro(state.data)[2], dtype=np.float32))

        done = bool(np.asarray(state.done))
        fell.append(done)
        if done and done_step is None:
            done_step = step_idx + 1
            break

    bundle = {
        "time_seconds": np.asarray(time_seconds, dtype=np.float32),
        "segment_id": np.asarray(segment_ids, dtype=np.int32),
        "command_lin_vel_xy": np.asarray(command_xy, dtype=np.float32),
        "measured_lin_vel_xy": np.asarray(measured_xy, dtype=np.float32),
        "command_yaw_rate": np.asarray(command_yaw, dtype=np.float32),
        "measured_yaw_rate": np.asarray(measured_yaw, dtype=np.float32),
        "fell": np.asarray(fell, dtype=bool),
    }

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_request": str(args.checkpoint_dir.resolve()),
        "stage_name": args.stage_name,
        "segment_seconds": float(args.segment_seconds),
        "segment_steps": int(segment_steps),
        "command_scale": float(args.command_scale),
        "commands": [command.tolist() for command in commands],
        "segment_labels": list(SEGMENT_LABELS),
        "segment_start_times_seconds": [idx * segment_steps * float(env.dt) for idx in range(len(commands))],
        "command_change_times_seconds": [idx * segment_steps * float(env.dt) for idx in range(1, len(commands))],
        "num_steps_simulated": int(len(bundle["time_seconds"])),
        "terminated_early": done_step is not None,
        "done_step": done_step,
        "segment_metrics": _segment_metrics(
            bundle["command_lin_vel_xy"],
            bundle["measured_lin_vel_xy"],
            bundle["command_yaw_rate"],
            bundle["measured_yaw_rate"],
            bundle["segment_id"],
        ),
    }

    rollout_npz = output_dir / "per_direction_rollout.npz"
    plot_png = output_dir / "per_direction_tracking.png"
    np.savez(rollout_npz, **bundle)
    summary["rollout_npz"] = str(rollout_npz)
    summary["plot_png"] = str(plot_png)
    save_json(output_dir / "per_direction_summary.json", summary)
    _plot_rollout(bundle, summary, plot_png)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
