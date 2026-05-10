# RSA2C

Code for RSA2C: SHAP-guided RKHS actor-critic experiments on continuous-control tasks.

This repository provides runnable experiment entry points for:

- RSA2C-CME
- RSA2C-KME
- RKHS-AC, the single-critic no-SHAP actor-critic baseline
- Advanced AC, the two-critic no-SHAP actor-critic baseline
- RSA2C-AC, the no-SHAP ablation
- Uniform SHAP
- SAC
- PPO

## Repository Layout

- `envs/continuous_control/`: shared Pendulum, Walker, and Ant implementations for RSA2C, RKHS-AC, Advanced AC, and Uniform SHAP.
- `envs/LQR/`: LQR implementation.
- `baselines/`: shared Stable-Baselines3 SAC and PPO baselines.
- `analysis/`: plotting, return aggregation, and SHAP visualization utilities.
- `instrumentation/`: memory footprint, runtime, FLOPs, and dictionary-size profiling code.
- `scripts/`: shell wrappers for the current repository layout.
- `run_experiment.py`: unified launcher for the main experiments.
- `EXPERIMENTS.md`: experiment matrix and additional analysis notes.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Additional environment notes:

- `BipedalWalker-v3` requires Box2D.
- `Ant-v5` requires MuJoCo support through Gymnasium.
- SAC and PPO use Stable-Baselines3.

## Quick Start

List supported environment/algorithm pairs:

```bash
python run_experiment.py --list
```

Dry-run a command:

```bash
python run_experiment.py Pendulum RSA2C-CME --dry-run --epochs 10 --seed 512
```

Run experiments:

```bash
python run_experiment.py Pendulum RSA2C-CME --epochs 2000 --seed 512
python run_experiment.py Walker RSA2C-KME --epochs 1500 --seed 888
python run_experiment.py Ant SAC --total_episodes 2000 --seed 76872
python run_experiment.py LQR RSA2C-CME --epochs 500 --seed 245
```

Arguments after the environment and algorithm are forwarded to the target script. Use `--cuda-device 0` to set `CUDA_VISIBLE_DEVICES` for the child process.

Equivalent shell wrappers are available under `scripts/`:

```bash
bash scripts/Pendulum/run_RSA2C_CME.sh --epochs 2000 --seed 512
bash scripts/Walker/run_RSA2C_KME.sh --epochs 1500 --seed 888
bash scripts/Ant/run_SAC.sh --total_episodes 2000 --seed 76872
bash scripts/instrumentation/run_Walker_line_profile.sh --epochs 100 --profile_flops
```

## Supported Experiments

| Environment | Algorithms |
| --- | --- |
| Pendulum-v1 | RSA2C-CME, RSA2C-KME, RKHS-AC, Advanced AC, RSA2C-AC, Uniform SHAP, SAC, PPO |
| BipedalWalker-v3 | RSA2C-CME, RSA2C-KME, RKHS-AC, Advanced AC, RSA2C-AC, Uniform SHAP, SAC, PPO |
| Ant-v5 | RSA2C-CME, RSA2C-KME, RKHS-AC, Advanced AC, RSA2C-AC, Uniform SHAP, SAC, PPO |
| LIP-LQR | RSA2C-CME, RSA2C-KME, Advanced AC, RSA2C-AC |

LIP-LQR is included for the appendix-style comparison and does not expose SAC,
PPO, Uniform SHAP, or the separate single-critic RKHS-AC variant.

## Notes

- Pendulum-v1, BipedalWalker-v3, and Ant-v5 use the same continuous-control code; environment differences are passed as launcher/script arguments.
- `RSA2C-CME` and `RSA2C-KME` select different SHAP backends.
- `RSA2C-AC` is kept as the no-SHAP RSA2C launcher name and uses the same two-critic path as `Advanced AC`.
- `RKHS-AC` is the single-critic RKHS actor-critic baseline without SHAP.
- `Advanced AC` is the two-critic actor-critic baseline without SHAP.
- `Uniform SHAP` uses fixed attribution weights.
- `instrumentation/` contains optional scripts for memory, runtime, FLOPs, and dictionary-size analysis.
