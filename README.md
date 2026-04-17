# Go2 Course Homework (Readable Repo for Colab Download)

This repository is the readable homework package that a Google Colab notebook
can `git clone` and run.

The key idea is:
- keep the important source code visible as normal `.py` / `.json` files
- let the Colab notebook install dependencies and clone this repo on demand
- avoid the old large payload cell that hid the real homework code

## Colab usage model

If you already have a cloud notebook, it should download this repo with:

```python
COURSE_REPO_URL = "https://github.com/WeijieLai1024/EEC289A_Robotics-Homework.git"
```

The notebook then needs to:

1. clone pinned versions of `mujoco_playground` and `unitree_mujoco`
2. clone this repository into `/content/go2_course_repo`
3. install `configs/colab_requirements.txt`
4. install the cloned `mujoco_playground` checkout in editable mode
5. copy Go2 mesh assets into `go2_pg_env/xmls/assets/`
6. run `inspect_env.py`, `train.py`, `test_policy.py`, and `public_eval.py`

The included notebook template at `notebooks/go2_public_colab_template.ipynb`
has already been updated to use this public repo URL and the repo-side helper
scripts in this repository.

## Why this repo exists

The original homework notebook worked, but it hid most important files inside a
payload blob. That was fine for distribution, but it made the repository harder
to inspect and maintain. This repo keeps the same main pipeline while exposing
the code students are expected to read and modify.

## File map

```text
configs/course_config.json         Course-level knobs and benchmark settings
configs/colab_requirements.txt     Python dependencies the Colab notebook should install
course_common.py                   Shared utilities used by all scripts

go2_pg_env/
  __init__.py                      Register the local Go2 env
  constants.py                     Names and XML paths
  base.py                          Base Go2 environment wrapper
  joystick.py                      Main task logic: obs, action, reward, reset, step
  randomize.py                     Domain randomization
  xmls/                            MuJoCo XML files

scripts/copy_go2_assets.py         Copy Go2 meshes from unitree_mujoco into this repo layout

train.py                           Two-stage PPO training
test_policy.py                     Restore a checkpoint and render a deterministic demo
generate_public_rollout.py         Generate the standardized public benchmark rollout
public_eval.py                     Score a rollout bundle
inspect_env.py                     Print a compact environment summary
quick_policy_check.py              Tiny sanity-check script
benchmark_specs.py                 Deterministic command scripts
```

## Baseline scope

The default baseline in this repo is intentionally narrow:
- training samples only forward `vx` commands
- the public benchmark still probes forward, lateral, yaw-only, and combined motion

That makes the baseline easy to understand while leaving clear room for
students to extend the command space and measure the effect with the benchmark.

For the homework version, keep `stage_1` as the fixed forward-only baseline.
Students should use `stage_2` to extend command sampling and curriculum in
`go2_pg_env/joystick.py`, without changing the notebook entry points.

## Expected Colab workflow

### 1. Setup and environment inspection

```bash
python scripts/copy_go2_assets.py \
  --unitree-dir /content/unitree_mujoco \
  --course-dir /content/go2_course_repo

python inspect_env.py --stage-name stage_2
```

### 2. Dry-run the config

```bash
python train.py --config configs/colab_runtime_config.json --dry-run
```

### 3. Train the baseline

```bash
python train.py \
  --config configs/colab_runtime_config.json \
  --stage both \
  --output-dir artifacts/run_baseline
```

### 4. Restore a checkpoint and render a demo

```bash
python test_policy.py \
  --config configs/colab_runtime_config.json \
  --checkpoint-dir artifacts/run_baseline/best_checkpoint \
  --stage-name stage_2 \
  --output-dir artifacts/demo_bundle
```

### 5. Generate the public benchmark

```bash
python generate_public_rollout.py \
  --config configs/colab_runtime_config.json \
  --checkpoint-dir artifacts/run_baseline/best_checkpoint \
  --stage-name stage_2 \
  --output-dir artifacts/public_eval_bundle \
  --num-episodes 4 \
  --render-first-episode

python public_eval.py \
  --config configs/colab_runtime_config.json \
  --rollout-npz artifacts/public_eval_bundle/rollout_public_eval.npz \
  --output-json artifacts/public_eval_bundle/public_eval.json
```

## What the Colab runtime still must provide

- a GPU runtime
- internet access to clone public GitHub repositories
- `ffmpeg` installed through `apt-get`
- the pinned `mujoco_playground` and `unitree_mujoco` checkouts
- a fresh asset copy step into `go2_pg_env/xmls/assets/`

This repo intentionally does not vendor the Unitree Go2 mesh assets.

## Student modification boundary

Students should mostly modify:
- `go2_pg_env/joystick.py`
- `go2_pg_env/randomize.py`
- `configs/course_config.json`

Students should usually not modify:
- `public_eval.py`
- checkpoint restore logic
- rollout bundle field names

That keeps the benchmark comparable across submissions.
