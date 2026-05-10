# Instrumentation Code

This folder contains measurement-oriented entry points for memory, runtime,
FLOPs, and dictionary-size checks. The launcher uses the same environment
argument pattern as the main training code.

| File | Purpose |
| --- | --- |
| `run_profile.py` | Unified profiling launcher for Pendulum, Walker, and Ant. |
| `RSA2C_compute_profile.py` | RSA2C profiling/runtime/dictionary-size implementation. |
| `RSA2C_line_profile.py` | RSA2C line/profile-stat/FLOPs implementation. |

These are not the default experiment entry points; use `envs/` for the paper-facing algorithms.

Examples:

```bash
python instrumentation/run_profile.py Walker compute --epochs 100
python instrumentation/run_profile.py Ant line --epochs 100
python instrumentation/run_profile.py Pendulum RKHS-AC --epochs 100
python instrumentation/run_profile.py Pendulum "Advanced AC" --epochs 100
```
