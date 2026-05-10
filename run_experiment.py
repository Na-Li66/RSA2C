#!/usr/bin/env python3
"""Unified launcher for the organized RSA2C experiment code."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent

ENV_IDS = {
    "Pendulum": "Pendulum-v1",
    "Walker": "BipedalWalker-v3",
    "Ant": "Ant-v5",
    "LQR": "LIP-LQR",
}

ENV_DISPLAY = {
    "Pendulum": "Pendulum",
    "Walker": "Walker",
    "Ant": "Ant",
    "LQR": "LQR",
}

ENV_ALIASES = {
    "pendulum-v1": "Pendulum",
    "pendulum": "Pendulum",
    "walker": "Walker",
    "bipedalwalker": "Walker",
    "bipedalwalker-v3": "Walker",
    "ant": "Ant",
    "ant-v5": "Ant",
    "lqr": "LQR",
    "lip-lqr": "LQR",
}

ALG_DISPLAY = {
    "RSA2C-CME": "RSA2C-CME",
    "RSA2C-KME": "RSA2C-KME",
    "RKHS-AC": "RKHS-AC",
    "Advanced AC": "Advanced AC",
    "RSA2C-AC": "RSA2C-AC",
    "Uniform SHAP": "Uniform SHAP",
    "SAC": "SAC",
    "PPO": "PPO",
}

ALG_ALIASES = {
    "rsa2c-cme": "RSA2C-CME",
    "cme": "RSA2C-CME",
    "rsa2c-kme": "RSA2C-KME",
    "kme": "RSA2C-KME",
    "rkhs-ac": "RKHS-AC",
    "rkhs": "RKHS-AC",
    "advanced-ac": "Advanced AC",
    "advanced": "Advanced AC",
    "rsa2c-ac": "RSA2C-AC",
    "woshap": "RSA2C-AC",
    "wo-shap": "RSA2C-AC",
    "noshap": "RSA2C-AC",
    "uniform-shap": "Uniform SHAP",
    "uniform": "Uniform SHAP",
    "sac": "SAC",
    "ppo": "PPO",
}

Entry = Optional[Tuple[str, List[str]]]

RSA2C_SCRIPT = "envs/continuous_control/RSA2C.py"
RKHS_AC_SCRIPT = "envs/continuous_control/RKHS_AC.py"
UNIFORM_SHAP_SCRIPT = "envs/continuous_control/Uniform_SHAP.py"

RSA2C_ENV_ARGS: Dict[str, List[str]] = {
    "Pendulum": [
        "--env", ENV_IDS["Pendulum"],
        "--episodes_per_update", "24",
        "--horizon", "200",
        "--epochs", "2000",
        "--actor_ell", "0.8",
        "--centers_init_a", "128",
        "--max_centers_a", "384",
        "--ald_eps_final_a", "2e-4",
        "--ald_eps_anneal_end_a", "300",
        "--ald_max_add_per_epoch_a", "64",
        "--centers_init_v", "128",
        "--max_centers_v", "384",
        "--ald_eps_final_v", "2e-4",
        "--ald_eps_anneal_end_v", "300",
        "--ald_max_add_per_epoch_v", "64",
        "--ald_add_interval", "1",
        "--ald_prune_interval", "1",
        "--sigma", "0.35",
        "--sigma_anneal_to", "0.25",
        "--sigma_anneal_end", "300",
        "--restart_p", "0.001",
        "--ts_actor_delay", "5",
        "--ts_critic_steps", "2",
        "--lr_actor", "1.0",
        "--pg_clip", "0",
        "--pg_scale", "1.0",
        "--value_mode", "td",
        "--lr_v", "1e-2",
        "--td_lambda", "0.9",
        "--use_adv_reg", "0",
        "--shap_interval", "10",
        "--cme_reg", "1e-3",
        "--shap_kappa", "0.25",
        "--shap_ratio_min", "0.5",
        "--shap_ratio_max", "2.0",
        "--shap_tr_kl", "0.005",
        "--shap_warmup", "5",
        "--shap_topk", "-1",
        "--seed", "5",
        "--eval_interval", "10",
        "--succ_threshold", "-200",
    ],
    "Walker": [
        "--env", ENV_IDS["Walker"],
        "--episodes_per_update", "12",
        "--horizon", "1600",
        "--epochs", "1500",
        "--actor_ell", "1.0",
        "--centers_init_a", "192",
        "--max_centers_a", "3072",
        "--ald_eps_final_a", "7e-4",
        "--ald_eps_anneal_end_a", "600",
        "--ald_max_add_per_epoch_a", "24",
        "--centers_init_v", "192",
        "--max_centers_v", "3072",
        "--ald_eps_final_v", "7e-4",
        "--ald_eps_anneal_end_v", "600",
        "--ald_max_add_per_epoch_v", "24",
        "--ald_add_interval", "3",
        "--ald_prune_interval", "50",
        "--sigma", "0.55",
        "--sigma_anneal_to", "0.30",
        "--sigma_anneal_end", "1200",
        "--ts_actor_delay", "1",
        "--ts_critic_steps", "3",
        "--lr_actor", "0.05",
        "--pg_clip", "0.6",
        "--pg_scale", "2.0",
        "--value_mode", "mc_ridge",
        "--lr_v", "2.5e-3",
        "--td_lambda", "0.9",
        "--use_adv_reg", "1",
        "--shap_interval", "25",
        "--cme_reg", "2e-3",
        "--shap_kappa", "0.08",
        "--shap_ratio_min", "0.7",
        "--shap_ratio_max", "1.4",
        "--shap_tr_kl", "0.002",
        "--shap_warmup", "60",
        "--shap_topk", "4",
        "--seed", "512",
        "--eval_interval", "10",
        "--succ_threshold", "250",
    ],
    "Ant": [
        "--env", ENV_IDS["Ant"],
        "--episodes_per_update", "4",
        "--horizon", "1000",
        "--epochs", "1500",
        "--actor_ell", "3.0",
        "--centers_init_a", "192",
        "--max_centers_a", "4096",
        "--ald_eps_final_a", "7e-4",
        "--ald_eps_anneal_end_a", "600",
        "--ald_max_add_per_epoch_a", "24",
        "--centers_init_v", "192",
        "--max_centers_v", "30720",
        "--ald_eps_final_v", "7e-4",
        "--ald_eps_anneal_end_v", "600",
        "--ald_max_add_per_epoch_v", "24",
        "--ald_add_interval", "3",
        "--ald_prune_interval", "50",
        "--sigma", "0.8",
        "--sigma_anneal_to", "0.30",
        "--sigma_anneal_end", "1200",
        "--ts_actor_delay", "1",
        "--ts_critic_steps", "3",
        "--lr_actor", "0.05",
        "--pg_clip", "0.6",
        "--pg_scale", "2.0",
        "--value_mode", "mc_ridge",
        "--lr_v", "2.5e-3",
        "--td_lambda", "0.9",
        "--use_adv_reg", "1",
        "--shap_interval", "25",
        "--cme_reg", "2e-3",
        "--shap_kappa", "0.08",
        "--shap_ratio_min", "0.7",
        "--shap_ratio_max", "1.4",
        "--shap_tr_kl", "0.002",
        "--shap_warmup", "60",
        "--shap_topk", "4",
        "--seed", "512",
        "--eval_interval", "1",
        "--succ_threshold", "250",
    ],
}

RKHS_ENV_ARGS: Dict[str, List[str]] = {
    "Pendulum": [
        "--env", ENV_IDS["Pendulum"],
        "--episodes_per_update", "24",
        "--horizon", "200",
        "--epochs", "2000",
        "--actor_ell", "0.8",
        "--centers_init_a", "128",
        "--max_centers_a", "384",
        "--ald_eps_final_a", "2e-4",
        "--ald_eps_anneal_end_a", "300",
        "--ald_max_add_per_epoch_a", "64",
        "--ald_add_interval", "1",
        "--ald_prune_interval", "1",
        "--sigma", "0.35",
        "--sigma_anneal_to", "0.25",
        "--sigma_anneal_end", "300",
        "--restart_p", "0.001",
        "--lr_actor", "1.0",
        "--pg_clip", "0",
        "--pg_scale", "1.0",
        "--seed", "5",
        "--eval_interval", "10",
        "--succ_threshold", "-200",
    ],
    "Walker": [
        "--env", ENV_IDS["Walker"],
        "--episodes_per_update", "12",
        "--horizon", "1600",
        "--epochs", "1500",
        "--actor_ell", "1.0",
        "--centers_init_a", "192",
        "--max_centers_a", "3072",
        "--ald_eps_final_a", "7e-4",
        "--ald_eps_anneal_end_a", "600",
        "--ald_max_add_per_epoch_a", "24",
        "--ald_add_interval", "3",
        "--ald_prune_interval", "50",
        "--sigma", "0.55",
        "--sigma_anneal_to", "0.30",
        "--sigma_anneal_end", "1200",
        "--lr_actor", "0.05",
        "--pg_clip", "0.6",
        "--pg_scale", "2.0",
        "--seed", "512",
        "--eval_interval", "10",
        "--succ_threshold", "250",
    ],
    "Ant": [
        "--env", ENV_IDS["Ant"],
        "--episodes_per_update", "4",
        "--horizon", "1000",
        "--epochs", "1500",
        "--actor_ell", "3.0",
        "--centers_init_a", "192",
        "--max_centers_a", "4096",
        "--ald_eps_final_a", "7e-4",
        "--ald_eps_anneal_end_a", "600",
        "--ald_max_add_per_epoch_a", "24",
        "--ald_add_interval", "3",
        "--ald_prune_interval", "50",
        "--sigma", "0.8",
        "--sigma_anneal_to", "0.30",
        "--sigma_anneal_end", "1200",
        "--lr_actor", "0.05",
        "--pg_clip", "0.6",
        "--pg_scale", "2.0",
        "--seed", "512",
        "--eval_interval", "1",
        "--succ_threshold", "250",
    ],
}


def _entry(script: str, *arg_groups: List[str]) -> Entry:
    args: List[str] = []
    for group in arg_groups:
        args.extend(group)
    return script, args


def _set_arg(args: List[str], flag: str, value: str) -> List[str]:
    updated = list(args)
    for idx, token in enumerate(updated[:-1]):
        if token == flag:
            updated[idx + 1] = value
            return updated
    updated.extend([flag, value])
    return updated


def _continuous_runs(env_name: str) -> Dict[str, Entry]:
    RSA2C_args = RSA2C_ENV_ARGS[env_name]
    RKHS_args = RKHS_ENV_ARGS[env_name]
    two_critic_args = _set_arg(RSA2C_args, "--use_adv_reg", "1")
    return {
        "RSA2C-CME": _entry(RSA2C_SCRIPT, RSA2C_args, ["--shap_backend", "cme", "--shap_enable"]),
        "RSA2C-KME": _entry(RSA2C_SCRIPT, RSA2C_args, ["--shap_backend", "kme", "--shap_enable"]),
        # RKHS-AC is the single-critic no-SHAP baseline.
        "RKHS-AC": _entry(RKHS_AC_SCRIPT, RKHS_args),
        # Advanced AC is the two-critic no-SHAP ablation from the paper.
        "Advanced AC": _entry(RSA2C_SCRIPT, two_critic_args),
        # RSA2C-AC is retained as the no-SHAP RSA2C launcher name.
        "RSA2C-AC": _entry(RSA2C_SCRIPT, two_critic_args),
        "Uniform SHAP": _entry(UNIFORM_SHAP_SCRIPT, RSA2C_args, ["--shap_enable"]),
        "SAC": ("baselines/SAC_SB3.py", ["--env", ENV_IDS[env_name]]),
        "PPO": ("baselines/PPO_SB3.py", ["--env", ENV_IDS[env_name]]),
    }


RUNS: Dict[str, Dict[str, Entry]] = {
    "Pendulum": _continuous_runs("Pendulum"),
    "Walker": _continuous_runs("Walker"),
    "Ant": _continuous_runs("Ant"),
    "LQR": {
        "RSA2C-CME": ("envs/LQR/main.py", ["--env", ENV_IDS["LQR"], "--shap_enable", "--shap_mode", "cme"]),
        "RSA2C-KME": ("envs/LQR/main.py", ["--env", ENV_IDS["LQR"], "--shap_enable", "--shap_mode", "kme"]),
        "Advanced AC": ("envs/LQR/main.py", ["--env", ENV_IDS["LQR"]]),
        "RSA2C-AC": ("envs/LQR/main.py", ["--env", ENV_IDS["LQR"]]),
    },
}

ALG_ORDER = [
    "RSA2C-CME",
    "RSA2C-KME",
    "RKHS-AC",
    "Advanced AC",
    "RSA2C-AC",
    "Uniform SHAP",
    "SAC",
    "PPO",
]


def _norm(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def resolve_env(value: str) -> str:
    key = _norm(value)
    if key not in ENV_ALIASES:
        raise SystemExit(f"Unknown environment: {value}. Use --list to see supported names.")
    return ENV_ALIASES[key]


def resolve_algorithm(value: str) -> str:
    key = _norm(value)
    if key not in ALG_ALIASES:
        raise SystemExit(f"Unknown algorithm: {value}. Use --list to see supported names.")
    return ALG_ALIASES[key]


def print_matrix() -> None:
    header = ["Environment", "Algorithms"]
    rows = []
    for env_name in ["Pendulum", "Walker", "Ant", "LQR"]:
        algorithms = [ALG_DISPLAY[alg] for alg in ALG_ORDER if alg in RUNS[env_name]]
        rows.append([ENV_DISPLAY[env_name], ", ".join(algorithms)])
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    print("  ".join(str(x).ljust(w) for x, w in zip(header, widths)))
    for row in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
    print()
    print("Extra arguments after environment and algorithm are forwarded to the target script.")


def build_command(env_name: str, alg_name: str, forwarded: List[str]) -> Tuple[Path, List[str]]:
    entry = RUNS[env_name].get(alg_name)
    if entry is None:
        raise SystemExit(
            f"No implementation is registered for {ENV_DISPLAY[env_name]}/{ALG_DISPLAY[alg_name]}. "
            "See README.md for the supported code layout."
        )
    rel_script, fixed_args = entry
    script = ROOT / rel_script
    cmd = [sys.executable, str(script), *fixed_args, *forwarded]
    return script, cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RSA2C experiments by environment and algorithm.",
        add_help=True,
    )
    parser.add_argument("environment", nargs="?")
    parser.add_argument("algorithm", nargs="?")
    parser.add_argument("--list", action="store_true", help="show the supported environment/algorithm matrix")
    parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
    parser.add_argument("--cuda-device", default=None, help="set CUDA_VISIBLE_DEVICES for the child process")
    args, forwarded = parser.parse_known_args()

    if args.list:
        print_matrix()
        return 0

    if not args.environment or not args.algorithm:
        parser.print_help()
        return 2

    env_name = resolve_env(args.environment)
    alg_name = resolve_algorithm(args.algorithm)
    script, cmd = build_command(env_name, alg_name, forwarded)

    display = subprocess.list2cmdline(cmd) if os.name == "nt" else " ".join(cmd)
    print(display)
    if args.dry_run:
        return 0

    child_env = os.environ.copy()
    if args.cuda_device is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    completed = subprocess.run(cmd, cwd=str(script.parent), env=child_env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
