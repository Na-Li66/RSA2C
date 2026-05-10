# Continuous-Control Entry Points

Pendulum-v1, BipedalWalker-v3, and Ant-v5 share the same implementation files in
this directory. Environment-specific differences such as horizon, dictionary
size, exploration scale, SHAP schedule, and evaluation cadence are supplied by
`run_experiment.py` and the shell wrappers under `scripts/`.

| File | Algorithms |
| --- | --- |
| `RSA2C.py` | RSA2C-CME, RSA2C-KME, Advanced AC, RSA2C-AC |
| `RKHS_AC.py` | RKHS-AC |
| `Uniform_SHAP.py` | Uniform SHAP |

`RKHS-AC` is the single-critic no-SHAP baseline. `Advanced AC` uses the
two-critic RSA2C architecture with SHAP disabled.
