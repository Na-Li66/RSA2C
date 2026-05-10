# Experiments

## Main Entry Points

| Environment | File | Launcher algorithms |
| --- | --- | --- |
| Pendulum-v1 / BipedalWalker-v3 / Ant-v5 | `envs/continuous_control/RSA2C.py` | `RSA2C-CME`, `RSA2C-KME`, `Advanced AC`, `RSA2C-AC` |
| Pendulum-v1 / BipedalWalker-v3 / Ant-v5 | `envs/continuous_control/RKHS_AC.py` | `RKHS-AC` |
| Pendulum-v1 / BipedalWalker-v3 / Ant-v5 | `envs/continuous_control/Uniform_SHAP.py` | `Uniform SHAP` |
| LIP-LQR | `envs/LQR/main.py` | `RSA2C-CME`, `RSA2C-KME`, `Advanced AC`, `RSA2C-AC` |
| Pendulum/Walker/Ant | `baselines/SAC_SB3.py` | `SAC` |
| Pendulum/Walker/Ant | `baselines/PPO_SB3.py` | `PPO` |

Use `python run_experiment.py --list` to print the supported matrix. The launcher applies environment-specific argument presets for Pendulum-v1, BipedalWalker-v3, and Ant-v5 while using the same Python implementation.
The separate single-critic RKHS-AC baseline is registered only for Pendulum-v1, BipedalWalker-v3, and Ant-v5.
`Advanced AC` and `RSA2C-AC` both use the two-critic RSA2C code path with SHAP disabled.

## Additional Analysis

| Analysis need | Code |
| --- | --- |
| Return aggregation and plotting | `analysis/plot_returns.py` |
| SHAP heatmap and beeswarm plots | `analysis/plot_shap.py` |
| Runtime, FLOPs, dictionary size | `instrumentation/run_profile.py`, `instrumentation/RSA2C_compute_profile.py`, `instrumentation/RSA2C_line_profile.py` |

## Launch Scripts

The `scripts/` directory mirrors the current code layout:

- `scripts/Pendulum/`: Pendulum-v1 wrappers.
- `scripts/Walker/`: BipedalWalker-v3 wrappers.
- `scripts/Ant/`: Ant-v5 wrappers.
- `scripts/LQR/`: LIP-LQR wrappers.
- `scripts/instrumentation/`: profiling and measurement wrappers.
