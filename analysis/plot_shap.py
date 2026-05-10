#!/usr/bin/env python3
"""Plot SHAP heatmaps or beeswarm plots from saved RSA2C NPZ outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PENDULUM_FEATURES = [r"$cos(\theta)$", r"$sin(\theta)$", r"$\dot{\theta}$"]

WALKER_FEATURES = [
    "hull angle", "hull angle speed", "vel x", "vel y",
    "hip joint 1", "hip joint speed 1", "knee joint 1", "knee joint speed 1", "leg contact 1",
    "hip joint 2", "hip joint speed 2", "knee joint 2", "knee joint speed 2", "leg contact 2",
    "hip joint 3", "hip joint speed 3", "knee joint 3", "knee joint speed 3", "leg contact 3",
    "hip joint 4", "hip joint speed 4", "knee joint 4", "knee joint speed 4", "leg contact 4",
]


def ant_features() -> list[str]:
    names = [
        "torso height", "torso qw", "torso qx", "torso qy", "torso qz",
        "front left hip", "front left ankle", "front right hip", "front right ankle",
        "back left hip", "back left ankle", "back right hip", "back right ankle",
    ]
    names += [
        "torso vx", "torso vy", "torso vz", "torso wx", "torso wy", "torso wz",
        "front left hip vel", "front left ankle vel", "front right hip vel", "front right ankle vel",
        "back left hip vel", "back left ankle vel", "back right hip vel", "back right ankle vel",
    ]
    bodies = [
        "torso", "front left leg", "front left aux", "front left ankle",
        "front right leg", "front right aux", "front right ankle",
        "back left leg", "back left aux", "back left ankle",
        "back right leg", "back right aux", "back right ankle",
    ]
    comps = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    names += [f"{body} {comp}" for body in bodies for comp in comps]
    return names


FEATURES = {
    "Pendulum": PENDULUM_FEATURES,
    "Walker": WALKER_FEATURES,
    "Ant": ant_features(),
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


def load_phi(path: Path) -> tuple[np.ndarray, np.ndarray | None, list[str] | None]:
    data = np.load(path, allow_pickle=True)
    if "phi" in data:
        phi = np.asarray(data["phi"], dtype=float)
    elif "PHI_norm" in data:
        phi = np.asarray(data["PHI_norm"], dtype=float)
    else:
        raise KeyError(f"{path} does not contain 'phi' or 'PHI_norm'")
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
    denom = np.maximum(np.sum(np.abs(phi), axis=1, keepdims=True), 1e-12)
    phi_norm = phi / denom
    epochs = np.asarray(data["epochs"], dtype=float) if "epochs" in data else None
    names = data["feature_names"].tolist() if "feature_names" in data else None
    return phi_norm, epochs, names


def feature_names(env: str, saved_names: list[str] | None, dim: int) -> list[str]:
    if saved_names and len(saved_names) == dim:
        return saved_names
    names = FEATURES.get(env, [])
    if len(names) >= dim:
        return names[:dim]
    return [f"x{i}" for i in range(dim)]


def topk_indices(phi: np.ndarray, names: list[str], topk: int) -> tuple[np.ndarray, list[str]]:
    scores = np.mean(np.abs(phi), axis=0)
    idx = np.argsort(scores)[::-1]
    if topk > 0:
        idx = idx[:topk]
    return idx, [names[i] for i in idx]


def plot_heatmap(phi: np.ndarray, epochs: np.ndarray | None, names: list[str], out: str, topk: int) -> None:
    idx, shown_names = topk_indices(phi, names, topk)
    shown = phi[:, idx]
    x_min = float(epochs[0]) if epochs is not None and len(epochs) else 0.0
    x_max = float(epochs[-1]) if epochs is not None and len(epochs) else float(max(1, shown.shape[0] - 1))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(idx))), dpi=200)
    im = ax.imshow(shown.T, aspect="auto", origin="lower", extent=[x_min, x_max, 0, len(idx)])
    ax.set_yticks(np.arange(len(idx)) + 0.5)
    ax.set_yticklabels(shown_names)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("State feature")
    fig.colorbar(im, ax=ax, label="Normalized SHAP")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


def plot_beeswarm(phi: np.ndarray, names: list[str], out: str, topk: int, jitter: float) -> None:
    idx, shown_names = topk_indices(phi, names, topk)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(idx))), dpi=200)
    for rank, col in enumerate(idx):
        y = rank + rng.uniform(-jitter, jitter, size=phi.shape[0])
        ax.scatter(phi[:, col], y, s=8, alpha=0.55)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(shown_names)
    ax.set_xlabel("Normalized SHAP")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot RSA2C SHAP NPZ outputs.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--kind", choices=["heatmap", "beeswarm"], default="heatmap")
    parser.add_argument("--out", default="shap_plot.pdf")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=0.15)
    args = parser.parse_args()

    env_name = ENV_ALIASES.get(args.env.strip().lower())
    if env_name is None:
        raise SystemExit("Unknown environment. Use Pendulum, Walker, or Ant.")

    phi, epochs, saved_names = load_phi(Path(args.npz))
    names = feature_names(env_name, saved_names, phi.shape[1])
    if args.kind == "heatmap":
        plot_heatmap(phi, epochs, names, args.out, args.topk)
    else:
        plot_beeswarm(phi, names, args.out, args.topk, args.jitter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
