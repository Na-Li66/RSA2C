"""Shared RKHS-AC baseline for Pendulum-v1, BipedalWalker-v3, and Ant-v5."""

import argparse, math, time, random, os, csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Gym/Gymnasium 兼容
try:
    import gymnasium as gym
except ImportError:
    import gym

import torch
from torch import nn

# =========================
# Utils & global dtype
# =========================
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def discount_cumsum_np(x, gamma):
    out = np.zeros_like(x, dtype=np.float64)
    run = 0.0
    for t in reversed(range(len(x))):
        run = x[t] + gamma * run
        out[t] = run
    return out

def t(x, device, dtype=DTYPE):
    return torch.as_tensor(x, device=device, dtype=dtype)

# ---------- RBF helpers (torch) ----------
def pairwise_sqdist_torch(X, Y, ell):
    ell = torch.as_tensor(ell, device=X.device, dtype=X.dtype)
    ell = torch.nan_to_num(ell, nan=1.0, posinf=1.0, neginf=1.0)
    ell = torch.clamp(ell, min=1e-6, max=1e6)
    if ell.ndim == 1:
        ell = ell.reshape(1, -1)
    Xn = X / ell
    Yn = Y / ell
    XX = (Xn**2).sum(dim=1, keepdim=True)                # [N,1]
    YY = (Yn**2).sum(dim=1, keepdim=True).transpose(0,1) # [1,M]
    d2 = XX + YY - 2.0 * (Xn @ Yn.transpose(0,1))        # [N,M]
    d2 = torch.nan_to_num(d2, nan=0.0, posinf=1e12, neginf=0.0)
    return d2

def rbf_kernel_vec_torch(x, centers, ell):
    d2 = pairwise_sqdist_torch(x[None,:], centers, ell)[0]
    k = torch.exp(-0.5 * d2)
    return torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)

def rbf_kernel_mat_torch(X, centers, ell):
    d2 = pairwise_sqdist_torch(X, centers, ell)
    K = torch.exp(-0.5 * d2)
    return torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

# ---------- kmeans++ (numpy, CPU) ----------
def kmeanspp_init(X, k, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("kmeanspp_init: X must be non-empty 2D array")
    N = X.shape[0]
    k = int(max(1, min(k, N)))
    idx0 = rng.integers(N)
    centers = [X[idx0]]
    closest_d2 = np.sum((X - centers[0])**2, axis=1)
    for _ in range(1, k):
        total = float(np.sum(closest_d2))
        if not np.isfinite(total) or total <= 1e-12:
            probs = np.full(N, 1.0 / N, dtype=np.float64)
        else:
            probs = closest_d2 / total
            probs = np.maximum(probs, 0.0)
            s = probs.sum()
            if not np.isfinite(s) or s <= 0.0:
                probs = np.full(N, 1.0 / N, dtype=np.float64)
            else:
                probs /= s
        idx = rng.choice(N, p=probs)
        centers.append(X[idx])
        d2 = np.sum((X - centers[-1])**2, axis=1)
        closest_d2 = np.maximum(0.0, np.minimum(closest_d2, d2))
    return np.stack(centers, axis=0)

# ---------- env helpers ----------
def _format_action(env, a_any):
    sp = env.action_space
    name = type(sp).__name__
    a = np.asarray(a_any)
    if name == "Discrete":
        return int(np.clip(a.squeeze(), 0, sp.n - 1))
    sample = np.asarray(sp.sample())
    low, high = np.asarray(sp.low), np.asarray(sp.high)
    if sample.shape == ():
        return float(np.clip(a.squeeze(), low, high))
    return np.clip(a.reshape(sample.shape), low, high)

def _step_env(env, a_any, allow_restart=True):
    res = env.step(_format_action(env, a_any))
    if isinstance(res, tuple) and len(res) == 5:  # gymnasium
        s2, r, terminated, truncated, info = res
        done = bool(terminated) or bool(truncated)
    else:  # old gym
        s2, r, done, info = res
    if allow_restart and (not done) and getattr(env, "_p_restart", 0.0) > 0.0:
        if env._restart_rng.random() < env._p_restart:
            done = True
            if isinstance(info, dict):
                info["restart"] = True
    return (s2, float(r), bool(done), info)

def _reset_env(env, seed=None):
    try:
        res = env.reset(seed=seed)
    except TypeError:
        res = env.reset()
    if isinstance(res, tuple):
        obs, info = res
    else:
        obs, info = res, {}
    return obs, info

def enable_restart_kernel(env, p_restart=0.0, seed=0):
    env._p_restart = float(p_restart)
    env._restart_rng = np.random.default_rng(seed + 100003)
    return env

# =========================
# State Normalizer
# =========================
class StateNormalizer:
    def __init__(self, mean, std, device):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std  = np.asarray(std, dtype=np.float64)
        self.std  = np.where(np.isfinite(self.std) & (self.std > 0), self.std, 1.0)
        self.t_mean = t(self.mean, device)
        self.t_std  = t(self.std,  device)

    def transform(self, X):  # numpy -> numpy
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean) / self.std

    def transform_torch(self, X_t):  # torch -> torch
        return (X_t - self.t_mean) / self.t_std

# =========================
# ALD Sparse Dictionary
# =========================
class ALDDictionary:
    def __init__(self, ell, eps, max_centers, jitter, device):
        self.device = device
        self.ell = t(ell, device)
        self.eps = float(eps)
        self.max_centers = int(max_centers)
        self.jitter = float(jitter)
        self.centers = torch.zeros((0, 0), device=device, dtype=DTYPE)  # [M,d]
        self.K_inv  = torch.zeros((0, 0), device=device, dtype=DTYPE)   # [M,M]
        self._updates_since_refactor = 0
        self._refactor_every = 256

    @property
    def M(self):
        return int(self.centers.shape[0])

    def _full_refactor(self):
        if self.M == 0:
            self.K_inv = torch.zeros((0,0), device=self.device, dtype=DTYPE); return
        K = rbf_kernel_mat_torch(self.centers, self.centers, self.ell)
        K = 0.5*(K + K.T) + self.jitter*torch.eye(self.M, device=self.device, dtype=DTYPE)
        self.K_inv = torch.linalg.solve(K, torch.eye(self.M, device=self.device, dtype=DTYPE))

    def _refactor_if_needed(self):
        if self._updates_since_refactor >= self._refactor_every:
            self._full_refactor()
            self._updates_since_refactor = 0

    def initialize(self, centers_np):
        self.centers = t(centers_np, self.device)
        if self.M == 0:
            self.K_inv = torch.zeros((0,0), device=self.device, dtype=DTYPE); return
        K = rbf_kernel_mat_torch(self.centers, self.centers, self.ell)
        K = 0.5*(K + K.T) + self.jitter*torch.eye(self.M, device=self.device, dtype=DTYPE)
        self.K_inv = torch.linalg.solve(K, torch.eye(self.M, device=self.device, dtype=DTYPE))

    def _kvec(self, x_n_t):
        if self.M == 0: return torch.zeros((0,), device=self.device, dtype=DTYPE)
        return rbf_kernel_vec_torch(x_n_t, self.centers, self.ell)

    def add_from_batch(self, S_n_np, max_add_per_epoch=64):
        if S_n_np is None or len(S_n_np) == 0: return 0
        X = np.asarray(S_n_np, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        added = 0
        if self.M == 0:
            x0 = t(X[np.random.randint(len(X))], self.device)
            self.centers = x0.reshape(1,-1).clone()
            self.K_inv = torch.tensor([[1.0/(1.0+self.jitter)]], device=self.device, dtype=DTYPE)
            added += 1
            X = X[1:] if len(X) > 1 else X[:0]

        if self.M == 0 or len(X) == 0:
            return added

        with torch.no_grad():
            X_t = t(X, self.device)
            Kx = rbf_kernel_mat_torch(X_t, self.centers, self.ell)   # [N,M]
            Kinv_KxT = self.K_inv @ Kx.T                             # [M,N]
            quad = (Kx * Kinv_KxT.T).sum(dim=1)                      # [N]
            s_all = (1.0 + self.jitter) - quad
            s_all = torch.nan_to_num(s_all, nan=0.0, posinf=0.0, neginf=0.0)

        mask = (s_all > float(self.eps)).cpu().numpy()
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return added

        s_np = s_all.cpu().numpy()
        order = idxs[np.argsort(-s_np[idxs])]
        take = int(min(max_add_per_epoch, self.max_centers - self.M, len(order)))

        for i in order[:take]:
            if self.M >= self.max_centers: break
            x = t(X[i], self.device)
            kx = rbf_kernel_vec_torch(x, self.centers, self.ell)
            s = (1.0 + self.jitter) - (kx @ (self.K_inv @ kx)).item()
            if not np.isfinite(s) or s <= self.eps: continue
            s = max(s, 1e-12)
            s_t = torch.tensor(s, device=self.device, dtype=DTYPE)
            Kinv_k = self.K_inv @ kx
            top_left  = self.K_inv + (Kinv_k[:,None] @ Kinv_k[None,:]) / s_t
            top_right = -Kinv_k[:,None] / s_t
            bot_left  = -Kinv_k[None,:] / s_t
            bot_right = torch.tensor([[1.0/s]], device=self.device, dtype=DTYPE)
            self.K_inv = torch.cat([torch.cat([top_left, top_right], dim=1),
                                    torch.cat([bot_left, bot_right], dim=1)], dim=0)
            self.centers = torch.cat([self.centers, x.reshape(1,-1)], dim=0)
            added += 1
            self._updates_since_refactor += 1
            self._refactor_if_needed()
        return added

    def prune_to_indices(self, keep_idx_np):
        keep_idx = torch.as_tensor(keep_idx_np, device=self.device, dtype=torch.long)
        self.centers = self.centers[keep_idx]
        if self.M == 0:
            self.K_inv = torch.zeros((0,0), device=self.device, dtype=DTYPE); return
        K = rbf_kernel_mat_torch(self.centers, self.centers, self.ell)
        K = 0.5*(K + K.T) + self.jitter*torch.eye(self.M, device=self.device, dtype=DTYPE)
        self.K_inv = torch.linalg.solve(K, torch.eye(self.M, device=self.device, dtype=DTYPE))
        self._updates_since_refactor = 0

# =========================
# Actor (tanh-squashed Gaussian)
# =========================
class KernelGaussianActor:
    """
    tanh-squashed Gaussian:
      u ~ N(mu, sigma^2 I),  a = mid + half * tanh(u)
    PG 用 pre-tanh 的 score: (u - mu)/sigma^2
    """
    def __init__(self, centers_t, ell, sigma, action_low, action_high, normalizer, device):
        self.device = device
        self.centers = centers_t.to(device, DTYPE)  # normalized [M, d_state]
        self.M = int(self.centers.shape[0])
        self.d = int(self.centers.shape[1]) if self.M > 0 else 0
        self.base_ell = t(ell, device)
        self.ell_eff  = t(ell, device)

        if isinstance(action_low, (list, np.ndarray)):
            self.low  = np.asarray(action_low, dtype=np.float64)
            self.high = np.asarray(action_high, dtype=np.float64)
        else:
            self.low  = np.array([float(action_low)],  dtype=np.float64)
            self.high = np.array([float(action_high)], dtype=np.float64)
        self.mid  = (self.high + self.low) / 2.0
        self.half = (self.high - self.low) / 2.0

        self.act_dim = None
        self.alpha = torch.zeros((self.M, 0), device=device, dtype=DTYPE)
        self.sigma = float(sigma)         # pre-tanh std (shared)
        self.normalizer = normalizer

    def _ensure_action_dim(self, env_action_sample=None):
        if self.act_dim is not None: return
        if env_action_sample is None:
            self.act_dim = int(self.low.size)
        else:
            arr = np.asarray(env_action_sample, dtype=np.float32)
            self.act_dim = int(np.prod(arr.shape))
        if self.M == 0:
            self.alpha = torch.zeros((0, self.act_dim), device=self.device, dtype=DTYPE)
        else:
            self.alpha = torch.zeros((self.M, self.act_dim), device=self.device, dtype=DTYPE)
        if self.low.size != self.act_dim:
            self.low  = np.full((self.act_dim,), float(self.low.ravel()[0]),  dtype=np.float64)
            self.high = np.full((self.act_dim,), float(self.high.ravel()[0]), dtype=np.float64)
        self.mid  = (self.high + self.low) / 2.0
        self.half = (self.high - self.low) / 2.0

    def _norm_single_t(self, s_np):
        x = t(s_np, self.device)
        return self.normalizer.transform_torch(x) if self.normalizer is not None else x

    def _norm_batch_t(self, S_np):
        X = t(S_np, self.device)
        return self.normalizer.transform_torch(X) if self.normalizer is not None else X

    def kvec(self, s_np):
        s_n = self._norm_single_t(s_np)
        if self.M == 0: return torch.zeros((0,), device=self.device, dtype=DTYPE)
        k = rbf_kernel_vec_torch(s_n, self.centers, self.ell_eff)
        return torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)

    def kmat(self, S_np):
        S_n = self._norm_batch_t(S_np)
        if self.M == 0:
            return torch.zeros((S_n.shape[0], 0), device=self.device, dtype=DTYPE)
        K = rbf_kernel_mat_torch(S_n, self.centers, self.ell_eff)
        return torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

    def mean(self, s_np):
        if self.act_dim is None: self._ensure_action_dim()
        k = self.kvec(s_np)
        if k.numel() == 0 or self.alpha.numel() == 0:
            return np.zeros((self.act_dim,), dtype=np.float64)
        mu_t = k @ self.alpha
        mu_t = torch.nan_to_num(mu_t, nan=0.0, posinf=0.0, neginf=0.0)
        mu = mu_t.detach().cpu().numpy().astype(np.float64)
        return mu

    def _squash_and_scale(self, u):
        return self.mid + self.half * np.tanh(u)

    def act(self, s_np, stochastic=True, env_action_sample=None):
        self._ensure_action_dim(env_action_sample)
        mu = self.mean(s_np)  # pre-tanh mean
        if not np.all(np.isfinite(mu)):
            mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        if stochastic:
            u = mu + self.sigma * np.random.randn(self.act_dim)
        else:
            u = mu.copy()
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        a = self._squash_and_scale(u)
        return a, mu, u

    # vanilla PG param update on alpha
    def update_vanilla(self, g_t, lr=1e-2, clip_norm=None, adapt_target=0.0, scale_mult=1.0):
        if g_t is None or g_t.numel() == 0:
            return 0.0, 0.0
        g = g_t * float(scale_mult)
        if adapt_target is not None and adapt_target > 0.0:
            gn = torch.linalg.norm(g)
            if torch.isfinite(gn) and gn.item() > 0:
                g = g * (float(adapt_target) / (gn.item() + 1e-12))
        if clip_norm is not None and clip_norm > 0:
            n = torch.linalg.norm(g).item()
            if n > clip_norm:
                g = g * (clip_norm / max(n, 1e-12))
        step = float(lr)
        self.alpha = self.alpha + step * g
        self.alpha = torch.nan_to_num(self.alpha, nan=0.0, posinf=0.0, neginf=0.0)
        return step, float(torch.linalg.norm(step * g).item())

    def on_added_centers(self, n_added):
        if n_added <= 0: return
        if self.act_dim is None:
            self.act_dim = 1
        pad = torch.zeros((n_added, self.alpha.shape[1]), device=self.device, dtype=DTYPE)
        self.alpha = torch.cat([self.alpha, pad], dim=0)
        self.M = int(self.centers.shape[0]); self.d = int(self.centers.shape[1])

    def on_pruned_centers(self, keep_idx_np):
        if self.M == 0:
            self.alpha = torch.zeros((0, self.alpha.shape[1] if self.alpha.ndim==2 else 0),
                                     device=self.device, dtype=DTYPE)
            return
        keep_idx = torch.as_tensor(keep_idx_np, device=self.device, dtype=torch.long)
        self.alpha = self.alpha[keep_idx]
        self.M = int(self.centers.shape[0]); self.d = int(self.centers.shape[1])

# =========================
# 显式 Advantage 核回归器（拟合到 G）
# =========================
class AdvantageKernelRegressor:
    """
    拟合  A_t  ~  <W,  k(s_t) r_t^T>_F  = (W r_t)^T k(s_t)
    其中 r_t = (u_t - mu_t)/sigma^2 (pre-tanh score), k(s_t)∈R^M
    """
    def __init__(self, actor: KernelGaussianActor, reg=1e-3, ema=0.97, device="cpu"):
        self.device = device
        self.actor = actor
        self.M = actor.M
        self.reg = float(reg)
        self.ema = float(ema)
        ad = actor.alpha.shape[1] if actor.alpha.ndim==2 else 0
        self.W = torch.zeros((self.M, ad), device=device, dtype=DTYPE)  # 参数
        self._A_scale_ema = 1.0

    def on_centers_changed(self, new_M):
        old = self.W
        old_M = int(old.shape[0])
        ad = self.actor.alpha.shape[1] if self.actor.alpha.ndim==2 else 0
        self.M = int(new_M)
        W_new = torch.zeros((self.M, ad), device=self.device, dtype=DTYPE)
        keepM, keepD = min(old_M, self.M), min(old.shape[1], ad)
        if keepM>0 and keepD>0:
            W_new[:keepM,:keepD] = old[:keepM,:keepD]
        self.W = W_new

    @torch.no_grad()
    def predict_batch(self, S_np, U_np, MU_np):
        # 返回 \hat A_proj for a batch
        K_s = self.actor.kmat(S_np)                         # [N,M]
        U   = t(U_np,  self.device).reshape(len(S_np), -1)  # [N,ad]
        MU  = t(MU_np, self.device).reshape(len(S_np), -1)  # [N,ad]
        sigma2 = max(self.actor.sigma**2, 1e-12)
        R = (U - MU) / sigma2                               # [N,ad]
        Vr = R @ self.W.T                                   # [N,M]
        A_pred = (Vr * K_s).sum(dim=1)                      # [N]
        return A_pred.detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def fit_batch(self, S_np, U_np, MU_np, A_target_np, lr=1e-2, clip_grad=None, scale_loss=True):
        if self.M == 0: return 0.0
        K_s = self.actor.kmat(S_np)                         # [N,M]
        U   = t(U_np,  self.device).reshape(len(S_np), -1)  # [N,ad]
        MU  = t(MU_np, self.device).reshape(len(S_np), -1)  # [N,ad]
        A_t = t(A_target_np, self.device).reshape(-1)       # [N]
        sigma2 = max(self.actor.sigma**2, 1e-12)
        R = (U - MU) / sigma2                               # [N,ad]

        Vr = R @ self.W.T                                   # [N,M]
        A_pred = (Vr * K_s).sum(dim=1)                      # [N]
        residual = A_pred - A_t                             # [N]

        if scale_loss:
            with torch.no_grad():
                cur = float(torch.sqrt(torch.mean(A_t*A_t)).item() + 1e-6)
                self._A_scale_ema = 0.99*self._A_scale_ema + 0.01*cur
                scale = 1.0/max(self._A_scale_ema, 1.0)
        else:
            scale = 1.0

        coef = (residual * scale).reshape(-1,1)            # [N,1]
        KR = (coef * K_s).T @ R                            # [M,ad]
        gW = KR + self.reg * self.W
        if clip_grad is not None and clip_grad > 0:
            n = torch.linalg.norm(gW).item()
            if n > clip_grad:
                gW = gW * (clip_grad / max(n,1e-12))
        self.W = self.W - float(lr) * gW
        self.W = torch.nan_to_num(self.W, nan=0.0, posinf=0.0, neginf=0.0)
        return float(torch.mean(residual*residual).item())

# =========================
# Rollout + Eval
# =========================
def collect_rollouts(env, actor, episodes, horizon, gamma, noise):
    with torch.no_grad():
        traj = []; states_for_dict = []
        obs_dim = env.observation_space.shape[0]
        act_sample = env.action_space.sample()
        for _ in range(episodes):
            s, info = _reset_env(env)
            ep = {"s": [], "s2": [], "a": [], "u": [], "mu": [], "r": [], "done": []}
            for tstep in range(horizon):
                a, mu, u = actor.act(s, stochastic=True, env_action_sample=act_sample)
                if not np.all(np.isfinite(a)):
                    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                s2, r, done, info2 = _step_env(env, a, allow_restart=True)
                if isinstance(s2, tuple): s2, _ = s2
                if noise > 0:
                    s2 = s2 + np.random.randn(obs_dim) * noise
                if (not np.isfinite(s2).all()) or (not np.isfinite(r)):
                    done = True
                ep["s"].append(s); ep["s2"].append(s2)
                ep["a"].append(np.asarray(a, dtype=np.float64))
                ep["u"].append(np.asarray(u, dtype=np.float64))      # pre-tanh sample
                ep["mu"].append(np.asarray(mu, dtype=np.float64))    # pre-tanh mean
                ep["r"].append(float(r)); ep["done"].append(bool(done))
                states_for_dict.append(s)
                s = s2
                if done: break
            traj.append(ep)

        S   = np.asarray([s for ep in traj for s in ep["s"]],   dtype=np.float64)
        S2  = np.asarray([s2 for ep in traj for s2 in ep["s2"]], dtype=np.float64)
        D   = np.asarray([d for ep in traj for d in ep["done"]], dtype=bool)
        A   = np.asarray([a for ep in traj for a in ep["a"]],    dtype=np.float64)
        U   = np.asarray([u for ep in traj for u in ep["u"]],    dtype=np.float64)  # pre-tanh
        MU  = np.asarray([m for ep in traj for m in ep["mu"]],   dtype=np.float64)  # pre-tanh
        R   = np.asarray([r for ep in traj for r in ep["r"]],    dtype=np.float64)
        lens = np.asarray([len(ep["r"]) for ep in traj], dtype=np.int32)

        for arr in (S, S2, A, U, MU, R):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        G = []
        idx = 0
        for L in lens:
            G.extend(discount_cumsum_np(R[idx:idx+L], gamma))
            idx += L
        G = np.asarray(G, dtype=np.float64)
        return S, S2, D, A, U, MU, R, G, lens, np.asarray(states_for_dict, dtype=np.float64)

def evaluate_policy(env, actor, noise, episodes=5, horizon=200, succ_threshold=250.0):
    with torch.no_grad():
        rets = []; succs = 0
        obs_dim = env.observation_space.shape[0]
        for _ in range(episodes):
            s, info = _reset_env(env)
            ret = 0.0
            for tstep in range(horizon):
                mu = actor.mean(s)
                a  = actor._squash_and_scale(mu)
                if not np.all(np.isfinite(a)):
                    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                s, r, done, info = _step_env(env, a, allow_restart=False)
                if isinstance(s, tuple): s, _ = s
                if noise > 0:
                    s = s + np.random.randn(obs_dim) * noise
                if not (np.isfinite(s).all() and np.isfinite(r)):
                    break
                ret += float(r)
                if done: break
            if not np.isfinite(ret): ret = 0.0
            rets.append(ret)
            if ret > succ_threshold: succs += 1
        mean_ret = float(np.mean(rets)) if len(rets) > 0 else 0.0
        succ_rate = float(succs) / max(1, episodes)
        return mean_ret, succ_rate

# =========================
# Train one seed (single critic, no value critic, no SHAP)
# =========================
def run_one_seed(args, seed, device):
    set_seed(seed)

    env_tr = gym.make(args.env)
    enable_restart_kernel(env_tr, p_restart=args.restart_p, seed=seed)
    try:
        env_tr.reset(seed=seed)
        if hasattr(env_tr.action_space, "seed"): env_tr.action_space.seed(seed)
        if hasattr(env_tr.observation_space, "seed"): env_tr.observation_space.seed(seed)
    except Exception:
        pass

    env_ev = gym.make(args.env)
    enable_restart_kernel(env_ev, p_restart=0.0, seed=seed+777)
    try:
        env_ev.reset(seed=seed+123)
        if hasattr(env_ev.action_space, "seed"): env_ev.action_space.seed(seed+123)
        if hasattr(env_ev.observation_space, "seed"): env_ev.observation_space.seed(seed+123)
    except Exception:
        pass

    # 动作边界
    if isinstance(env_tr.action_space.high, np.ndarray):
        a_high = env_tr.action_space.high.astype(np.float64)
        a_low  = env_tr.action_space.low.astype(np.float64)
    else:
        a_high = np.array([float(env_tr.action_space.high)], dtype=np.float64)
        a_low  = np.array([float(env_tr.action_space.low)],  dtype=np.float64)

    # Warmup 统计
    WARM_EP = max(10, args.episodes_per_update)
    tmp_actor = KernelGaussianActor(
        centers_t=t(np.zeros((1, env_tr.observation_space.shape[0])), device),
        ell=1.0, sigma=args.sigma,
        action_low=a_low, action_high=a_high,
        normalizer=None, device=device
    )
    _, _, _, _, _, _, _, _, _, states_pool = collect_rollouts(env_tr, tmp_actor, WARM_EP, args.horizon, args.gamma, args.noise)
    states_pool = np.nan_to_num(states_pool, nan=0.0, posinf=0.0, neginf=0.0)
    state_mean = states_pool.mean(axis=0)
    state_std  = states_pool.std(axis=0) + 1e-8
    normalizer = StateNormalizer(state_mean, state_std, device)

    base_ell = np.ones_like(state_std) * float(args.actor_ell)

    # 初始字典（仅 Actor）
    states_pool_n = normalizer.transform(states_pool)
    init_k_a = min(args.centers_init_a, args.max_centers_a)
    centers_init_a = kmeanspp_init(states_pool_n, init_k_a, seed=seed)

    # ALD 字典（仅 Actor）
    ald_A = ALDDictionary(ell=base_ell, eps=args.ald_eps_a, max_centers=args.max_centers_a,
                          jitter=1e-5, device=device)
    ald_A.initialize(centers_init_a)

    # 模型
    actor = KernelGaussianActor(ald_A.centers, ell=base_ell, sigma=args.sigma,
                                action_low=a_low, action_high=a_high,
                                normalizer=normalizer, device=device)
    a_reg = AdvantageKernelRegressor(actor, reg=args.adv_reg, ema=args.adv_ema, device=device)

    if seed == args.seed:
        print(f"[Device] {device} | dtype={DTYPE}")
        print(f"[Info] env={args.env} | horizon={args.horizon} | ep/update={args.episodes_per_update} "
              f"| init_A={ald_A.M}/{args.max_centers_a} | base_ell={args.actor_ell} "
              f"| RKHS-AC | Adv=explicit kernel ridge | PG=vanilla(tanh) | SHAP=OFF | VALUE=OFF")

    eval_curve = np.full((args.epochs+1,), np.nan, dtype=np.float64)

    for ep in trange(args.epochs+1):
        # restart 概率退火
        if args.restart_anneal_end > 0:
            frac = max(0.0, 1.0 - ep / max(1, args.restart_anneal_end))
            env_tr._p_restart = args.restart_p * frac
        else:
            env_tr._p_restart = args.restart_p

        # ALD eps 退火（Actor）
        if args.ald_eps_anneal_end_a > 0:
            frac_eps_a = max(0.0, 1.0 - ep / max(1, args.ald_eps_anneal_end_a))
            ald_A.eps = max(args.ald_eps_final_a, args.ald_eps_a * frac_eps_a)

        # ===== 采样 =====
        S, S2, DONE, A, U, MU, R, G, lens, states_batch = collect_rollouts(
            env_tr, actor, args.episodes_per_update, args.horizon, args.gamma, args.noise)

        # ===== 字典扩展/修剪（仅 Actor）=====
        do_add = (ep % max(1, args.ald_add_interval) == 0)
        if do_add:
            S_n = normalizer.transform(states_batch)
            added_A = ald_A.add_from_batch(S_n, max_add_per_epoch=args.ald_max_add_per_epoch_a)
            if added_A > 0:
                actor.centers = ald_A.centers; actor.on_added_centers(added_A)
                a_reg.actor = actor; a_reg.on_centers_changed(ald_A.M)

        do_prune = (ep % max(1, args.ald_prune_interval) == 0)
        if do_prune and ald_A.M > args.max_centers_a:
            keep_A = np.argsort(actor.alpha.detach().cpu().abs().numpy().sum(axis=1))[::-1][:args.max_centers_a]
            keep_A.sort()
            ald_A.prune_to_indices(keep_A)
            actor.centers = ald_A.centers; actor.on_pruned_centers(keep_A)
            a_reg.actor = actor; a_reg.on_centers_changed(ald_A.M)

        # ===== Advantage 目标：直接用回报 G（可选 clip 与标准化）=====
        Ahat = G.copy()
        if args.adv_clip is not None and args.adv_clip > 0:
            Ahat = np.clip(Ahat, -float(args.adv_clip), float(args.adv_clip))
        std = Ahat.std()
        if std > 1e-6:
            Ahat = (Ahat - Ahat.mean()) / (std + 1e-8)

        # ===== Advantage 显式核回归（拟合到 Ahat）=====
        _ = a_reg.fit_batch(S, U, MU, Ahat, lr=args.adv_lr, clip_grad=args.adv_clip_grad, scale_loss=True)
        A_proj = a_reg.predict_batch(S, U, MU)  # 投影后的优势

        # ===== Policy gradient（用 A_proj 降低方差）=====
        K_s = actor.kmat(S)                                    # [N,M]
        U_t = t(U, device).reshape(len(S), -1)
        MU_t= t(MU,device).reshape(len(S), -1)
        sigma2 = max(actor.sigma**2, 1e-12)
        Rscore = (U_t - MU_t) / sigma2                         # [N,ad]
        Aproj_t = t(A_proj, device).reshape(-1,1)              # [N,1]
        K_sa = Rscore * Aproj_t                                # [N,ad]
        g = K_s.T @ K_sa / max(len(S),1)                       # [M,ad]

        # ===== Actor update =====
        lr_step, step_norm = actor.update_vanilla(
            g, lr=args.lr_actor,
            clip_norm=(args.pg_clip if args.pg_clip>0 else None),
            adapt_target=args.pg_adapt_target,
            scale_mult=args.pg_scale
        )
        if args.max_alpha_norm > 0:
            an = float(torch.linalg.norm(actor.alpha).item())
            if an > args.max_alpha_norm:
                actor.alpha *= (args.max_alpha_norm / (an + 1e-12))

        # sigma 退火
        if ep < args.sigma_anneal_end:
            frac_s = 1.0 - ep / max(1, args.sigma_anneal_end)
            actor.sigma = args.sigma_anneal_to + (args.sigma - args.sigma_anneal_to) * frac_s
        else:
            actor.sigma = max(args.sigma_anneal_to, 1e-3)

        # 评估 & 打印
        if (ep % args.eval_interval == 0) or (ep == 0):
            eval_ret, succ_rate = evaluate_policy(env_ev, actor, args.eval_noise,
                                                  episodes=args.eval_episodes, horizon=args.horizon,
                                                   succ_threshold=args.succ_threshold)
            alpha_norm = float(torch.linalg.norm(actor.alpha).item()) if actor.alpha.numel() else 0.0
            Wnorm = float(torch.linalg.norm(a_reg.W).item()) if a_reg.W.numel() else 0.0
            print(f"{ep:04d} A* | eval={eval_ret:.2f} | succ={succ_rate:.2f} "
                  f"| M_A={ald_A.M} | lr={lr_step:.3g} step={step_norm:.3g} "
                  f"| sigma={actor.sigma:.3g} | ||alpha||={alpha_norm:.3g} | ||W||={Wnorm:.3g} | device={device}")
            eval_curve[ep] = eval_ret
            print("[diag]", "|a|>0.98:", np.mean(np.abs(A) > 0.98))

    # 前向填充
    last = np.nan
    for i in range(eval_curve.shape[0]):
        if np.isnan(eval_curve[i]): eval_curve[i] = last
        else: last = eval_curve[i]

    env_tr.close(); env_ev.close()
    return eval_curve

# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="BipedalWalker-v3")
    p.add_argument("--episodes_per_update", type=int, default=12)
    p.add_argument("--horizon", type=int, default=1600)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--gamma", type=float, default=0.99)

    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", type=str, default=default_dev)

    p.add_argument("--actor_ell", type=float, default=1.0)

    # 仅 Actor 字典
    p.add_argument("--centers_init_a", type=int, default=192)
    p.add_argument("--max_centers_a", type=int, default=3072)
    p.add_argument("--ald_eps_a", type=float, default=1e-3)
    p.add_argument("--ald_eps_final_a", type=float, default=7e-4)
    p.add_argument("--ald_eps_anneal_end_a", type=int, default=600)
    p.add_argument("--ald_max_add_per_epoch_a", type=int, default=24)

    p.add_argument("--ald_add_interval", type=int, default=3)
    p.add_argument("--ald_prune_interval", type=int, default=50)
    p.add_argument("--adv_ema", type=float, default=0.0,
               help="EMA smoothing strength for advantage target scale (compat).")

    p.add_argument("--sigma", type=float, default=0.55)
    p.add_argument("--sigma_anneal_to", type=float, default=0.30)
    p.add_argument("--sigma_anneal_end", type=int, default=1200)

    p.add_argument("--restart_p", type=float, default=0.000)
    p.add_argument("--restart_anneal_end", type=int, default=0)

    # PG/Actor
    p.add_argument("--lr_actor", type=float, default=0.05)
    p.add_argument("--pg_clip", type=float, default=0.6)
    p.add_argument("--pg_adapt_target", type=float, default=0.0,
                   help=">0 时将 PG 范数重标定到目标值；默认关闭。")
    p.add_argument("--pg_scale", type=float, default=2.0,
                   help="对 vanilla PG 梯度做整体乘法缩放。")

    # Advantage 显式核回归器
    p.add_argument("--adv_lr", type=float, default=5e-3,
                   help="Advantage 核回归器的学习率（SGD）。")
    p.add_argument("--adv_reg", type=float, default=1e-3,
                   help="岭回归正则 λ。")
    p.add_argument("--adv_clip_grad", type=float, default=5.0,
                   help="显式回归器的梯度裁剪范数。")

    # 评估与输出
    p.add_argument("--seed", type=int, default=512)
    p.add_argument("--multi_seeds", type=int, default=1)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--succ_threshold", type=float, default=250.0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--eval_noise", type=float, default=0.0)
    p.add_argument("--max_alpha_norm", type=float, default=1e4)
    p.add_argument("--out_csv", type=str, default="eval_returns.csv")
    p.add_argument("--out_dir", type=str, default="return_logs")
    p.add_argument("--out_png", type=str, default="eval_curve.png")
    p.add_argument("--adv_clip", type=float, default=10.0,
                   help="Clip G in [-adv_clip, adv_clip] before standardization; <=0 disables.")

    args = p.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    seeds = [args.seed + i for i in range(args.multi_seeds)]
    all_curves = []
    t0 = time.time()
    for s in seeds:
        print(f"[Run] seed={s}")
        curve = run_one_seed(args, s, device)
        all_curves.append(curve)
    all_curves = np.stack(all_curves, axis=0)
    mean_curve = np.nanmean(all_curves, axis=0)
    std_curve  = np.nanstd(all_curves, axis=0)
    T = mean_curve.shape[0]
    epochs = np.arange(T)

    out_csv_path = os.path.join(args.out_dir, f"eval_return_seed{args.seed}_RKHS_AC_{args.noise}.csv")
    header = ["epoch", "mean", "std"] + [f"seed_{i}" for i in range(len(seeds))]
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t_i in range(T):
            row = [t_i, float(mean_curve[t_i]), float(std_curve[t_i])] + [float(all_curves[i,t_i]) for i in range(len(seeds))]
            w.writerow(row)

    plt.figure(figsize=(7,4))
    plt.plot(epochs, mean_curve, label="mean eval_det")
    plt.xlabel("Epoch"); plt.ylabel("Eval return (deterministic)")
    plt.title(f"{args.env} | mean over {len(seeds)} seeds (RKHS-AC | single critic | NO SHAP)")
    plt.legend(); plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"[Done] saved CSV to {out_csv_path}, figure to {args.out_png} | time={time.time()-t0:.1f}s | device={device}")

if __name__ == "__main__":
    main()
