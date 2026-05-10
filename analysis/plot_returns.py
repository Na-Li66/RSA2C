#!/usr/bin/env python3
"""Plot return curves from RSA2C CSV or NPZ outputs.

Examples:
  python analysis/plot_returns.py --curve RSA2C-CME=logs/cme --curve PPO=ppo.csv --out returns.pdf
  python analysis/plot_returns.py --curve Ant=ant_results.npz --x-key episodes --mean-key mean_returns
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _numeric_csv(path: Path, x_key: Optional[str], mean_key: Optional[str], std_key: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if x_key and mean_key:
        out = pd.DataFrame({"x": pd.to_numeric(df[x_key], errors="coerce"),
                            "mean": pd.to_numeric(df[mean_key], errors="coerce")})
        if std_key and std_key in df:
            out["std"] = pd.to_numeric(df[std_key], errors="coerce")
        return out.dropna(subset=["x", "mean"])

    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path)

    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if len(numeric_cols) < 2:
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        numeric_cols = [c for c in numeric_df.columns if numeric_df[c].notna().any()]
        df = numeric_df
    if len(numeric_cols) < 2:
        raise ValueError(f"{path} does not contain at least two numeric columns")

    out = pd.DataFrame({"x": pd.to_numeric(df[numeric_cols[0]], errors="coerce"),
                        "mean": pd.to_numeric(df[numeric_cols[1]], errors="coerce")})
    if len(numeric_cols) >= 3:
        out["std"] = pd.to_numeric(df[numeric_cols[2]], errors="coerce")
    return out.dropna(subset=["x", "mean"])


def _csv_files(path: Path, pattern: str, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.rglob(pattern) if recursive else path.glob(pattern))
    if not files:
        return sorted((path / "return").rglob(pattern)) if (path / "return").exists() else []
    return files


def _aggregate_csvs(files: Iterable[Path], x_key: Optional[str], mean_key: Optional[str], std_key: Optional[str]) -> pd.DataFrame:
    frames = [_numeric_csv(path, x_key, mean_key, std_key)[["x", "mean"]] for path in files]
    if not frames:
        raise FileNotFoundError("No CSV files found for curve")
    merged = pd.concat(frames, ignore_index=True)
    agg = (
        merged.groupby("x")
        .agg(mean=("mean", "mean"), std=("mean", "std"), min=("mean", "min"), max=("mean", "max"), count=("mean", "count"))
        .reset_index()
        .sort_values("x")
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def _load_npz(path: Path, x_key: Optional[str], mean_key: Optional[str], std_key: Optional[str]) -> pd.DataFrame:
    data = np.load(path, allow_pickle=True)
    x_candidates = [x_key, "episodes", "epochs", "epoch", "steps"]
    mean_candidates = [mean_key, "mean_returns", "mean_return", "mean", "returns", "eval_returns"]
    std_candidates = [std_key, "std_returns", "std_return", "std"]

    x_name = next((key for key in x_candidates if key and key in data), None)
    mean_name = next((key for key in mean_candidates if key and key in data), None)
    if mean_name is None:
        raise KeyError(f"{path} does not contain a known return key")
    y = np.asarray(data[mean_name], dtype=float).reshape(-1)
    x = np.asarray(data[x_name], dtype=float).reshape(-1) if x_name else np.arange(len(y), dtype=float)
    out = pd.DataFrame({"x": x[:len(y)], "mean": y[:len(x)]})
    std_name = next((key for key in std_candidates if key and key in data), None)
    if std_name:
        std = np.asarray(data[std_name], dtype=float).reshape(-1)
        out["std"] = std[:len(out)]
    return out


def load_curve(path: Path, pattern: str, recursive: bool, x_key: Optional[str], mean_key: Optional[str], std_key: Optional[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".npz":
        return _load_npz(path, x_key, mean_key, std_key)
    return _aggregate_csvs(_csv_files(path, pattern, recursive), x_key, mean_key, std_key)


def parse_curve(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    return label.strip(), Path(raw_path.strip())


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot return curves from CSV/NPZ outputs.")
    parser.add_argument("--curve", action="append", required=True, help="LABEL=path. Path may be a CSV, NPZ, or directory of CSV files.")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--x-key", default=None)
    parser.add_argument("--mean-key", default=None)
    parser.add_argument("--std-key", default=None)
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--band", choices=["std", "minmax", "none"], default="std")
    parser.add_argument("--out", default="returns.pdf")
    parser.add_argument("--xlabel", default="Epoch")
    parser.add_argument("--ylabel", default="Return")
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    for item in args.curve:
        label, path = parse_curve(item)
        curve = load_curve(path, args.pattern, args.recursive, args.x_key, args.mean_key, args.std_key)
        x = curve["x"].to_numpy(dtype=float)
        y = smooth(curve["mean"].to_numpy(dtype=float), args.smooth)
        ax.plot(x, y, linewidth=2.0, label=label)
        if args.band != "none":
            if args.band == "minmax" and {"min", "max"}.issubset(curve.columns):
                lo = smooth(curve["min"].to_numpy(dtype=float), args.smooth)
                hi = smooth(curve["max"].to_numpy(dtype=float), args.smooth)
            elif "std" in curve:
                std = smooth(curve["std"].fillna(0).to_numpy(dtype=float), args.smooth)
                lo, hi = y - std, y + std
            else:
                continue
            ax.fill_between(x, lo, hi, alpha=0.18)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_xlim(left=0, right=args.x_max)
    if args.y_min is not None or args.y_max is not None:
        ax.set_ylim(bottom=args.y_min, top=args.y_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
