#!/usr/bin/env python3
"""Plot training-time tracking, energy, and slip diagnostics from PPO logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METRIC_KEYS = {
    "lin_vel_error": (
        "eval/episode_tracking/lin_vel_error",
        "eval/tracking/lin_vel_error",
        "tracking/lin_vel_error",
    ),
    "yaw_error": (
        "eval/episode_tracking/yaw_error",
        "eval/tracking/yaw_error",
        "tracking/yaw_error",
    ),
    "energy_usage": (
        "eval/episode_tracking/energy_usage",
        "eval/tracking/energy_usage",
        "tracking/energy_usage",
    ),
    "slip_rate": (
        "eval/episode_tracking/slip_rate",
        "eval/tracking/slip_rate",
        "tracking/slip_rate",
    ),
}

PLOT_SPECS = (
    ("lin_vel_error", "Training-time linear velocity tracking error", "mean ||cmd - meas|| [m/s]"),
    ("yaw_error", "Training-time yaw tracking error", "mean |cmd - meas| [rad/s]"),
    ("energy_usage", "Training-time energy usage", "mean sum |joint velocity * torque| [W]"),
    ("slip_rate", "Training-time foot slip rate", "mean contact-foot slip speed [m/s]"),
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_metric(metrics: dict[str, Any], candidates: tuple[str, ...]) -> float | None:
    for key in candidates:
        if key in metrics:
            return float(metrics[key])

    suffixes = tuple(f"/{key}" for key in candidates)
    for key, value in metrics.items():
        if key.endswith(suffixes):
            return float(value)
    return None


def _load_stage_records(stage_dir: Path) -> tuple[list[int], dict[str, list[float]]]:
    progress_path = stage_dir / "progress.json"
    if not progress_path.exists():
        progress_path = stage_dir / "progress_live.json"
    if not progress_path.exists():
        raise FileNotFoundError(f"No progress.json or progress_live.json found in {stage_dir}")

    steps: list[int] = []
    series: dict[str, list[float]] = {name: [] for name in METRIC_KEYS}

    for record in _load_json(progress_path):
        metrics = record.get("metrics", {})
        values = {name: _find_metric(metrics, keys) for name, keys in METRIC_KEYS.items()}
        if any(value is None for value in values.values()):
            continue
        steps.append(int(record["num_steps"]))
        for name, value in values.items():
            series[name].append(float(value))

    if not steps:
        available = sorted(_load_json(progress_path)[-1].get("metrics", {}).keys())
        raise KeyError(
            "No complete tracking diagnostic metrics were found. "
            "This plot requires a run trained after tracking, energy, and slip metrics were added. "
            f"Last record metric keys: {available}"
        )

    return steps, series


def _has_progress_log(path: Path) -> bool:
    return (path / "progress.json").exists() or (path / "progress_live.json").exists()


def _discover_stage_dirs(run_dir: Path) -> list[Path]:
    if _has_progress_log(run_dir):
        return [run_dir]

    stage_dirs = [path for path in (run_dir / "stage_1", run_dir / "stage_2") if _has_progress_log(path)]
    if stage_dirs:
        return stage_dirs

    discovered = sorted({path.parent for path in run_dir.rglob("progress*.json")})
    return discovered


def plot_training_errors(run_dir: Path, output_png: Path) -> None:
    stage_dirs = _discover_stage_dirs(run_dir)
    if not stage_dirs:
        raise FileNotFoundError(
            f"No progress.json or progress_live.json found under {run_dir}. "
            "Run training first, or pass --run-dir to the actual output directory, for example "
            "--run-dir artifacts/YOUR_RUN_NAME. If you trained before adding these metrics, retrain so "
            "the new tracking/energy/slip keys are written."
        )

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=False)
    colors = {"stage_1": "#4c72b0", "stage_2": "#dd8452", run_dir.name: "#4c72b0"}

    for stage_dir in stage_dirs:
        steps, series = _load_stage_records(stage_dir)
        label = stage_dir.name
        color = colors.get(label)
        for ax, (metric_name, _, _) in zip(axes, PLOT_SPECS):
            ax.plot(steps, series[metric_name], marker="o", linewidth=1.8, label=label, color=color)

    for ax, (_, title, ylabel) in zip(axes, PLOT_SPECS):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("environment steps")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Saved {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/run_default"),
        help="Training output directory containing stage progress logs.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Where to save the plot. Defaults to RUN_DIR/training_diagnostics.png.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_png = args.output_png.resolve() if args.output_png else run_dir / "training_diagnostics.png"
    plot_training_errors(run_dir, output_png)


if __name__ == "__main__":
    main()
