#!/usr/bin/env python3
"""Unified launcher for profiling and measurement-oriented entry points."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ENV_IDS = {
    "Pendulum": "Pendulum-v1",
    "Walker": "BipedalWalker-v3",
    "Ant": "Ant-v5",
}

ENV_ALIASES = {
    "pendulum": "Pendulum",
    "pendulum-v1": "Pendulum",
    "walker": "Walker",
    "bipedalwalker": "Walker",
    "bipedalwalker-v3": "Walker",
    "ant": "Ant",
    "ant-v5": "Ant",
}

PROFILE_ALIASES = {
    "rsa2c-cme": "RSA2C-CME",
    "rsa2c-kme": "RSA2C-KME",
    "rkhs-ac": "RKHS-AC",
    "advanced-ac": "Advanced AC",
    "compute": "compute",
    "line": "line",
}


def _norm(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RSA2C profiling helpers.")
    parser.add_argument("env")
    parser.add_argument(
        "profile",
        help=(
            "RSA2C-CME/RSA2C-KME/RKHS-AC/Advanced AC call the main launcher with "
            "measurement-friendly defaults; compute/line call the dedicated RSA2C profiling scripts."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args()

    env_name = ENV_ALIASES.get(_norm(args.env))
    profile_name = PROFILE_ALIASES.get(_norm(args.profile))
    if env_name is None:
        raise SystemExit("Unknown environment. Use Pendulum, Walker, or Ant.")
    if profile_name is None:
        raise SystemExit("Unknown profile. Use RSA2C-CME, RSA2C-KME, RKHS-AC, Advanced AC, compute, or line.")

    if profile_name in {"RSA2C-CME", "RSA2C-KME", "RKHS-AC", "Advanced AC"}:
        command = [sys.executable, str(ROOT / "run_experiment.py"), env_name, profile_name, *forwarded]
    elif profile_name == "compute":
        command = [
            sys.executable,
            str(ROOT / "instrumentation" / "RSA2C_compute_profile.py"),
            "--env",
            ENV_IDS[env_name],
            *forwarded,
        ]
    else:
        command = [
            sys.executable,
            str(ROOT / "instrumentation" / "RSA2C_line_profile.py"),
            "--env",
            ENV_IDS[env_name],
            *forwarded,
        ]

    print(subprocess.list2cmdline(command))
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
