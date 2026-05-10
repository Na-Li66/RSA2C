# filename: main_adv_explicit.py
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

# =========================
# Profiling helpers (runtime + FLOPs estimates)
# =========================
def _sync(dev):
    try:
        if torch.cuda.is_available() and str(dev).startswith("cuda"):
            torch.cuda.synchronize()
    except Exception:
        pass

def _ts(dev):
    _sync(dev); return time.perf_counter()

def human_flops(x):
    units = ["FLOPs","KFLOPs","MFLOPs","GFLOPs","TFLOPs","PFLOPs"]
    i = 0
    x = float(x)
    while x >= 1000.0 and i < len(units)-1:
        x /= 1000.0; i += 1
    return f"{x:.3f} {units[i]}"

def flops_rbf_kmat(N, M, d):
    # RBF K(X[N,d], C[M,d]): 2*N*M*d (matmul) + ~6*N*M 其它标量操作
    return 2.0*N*M*d + 6.0*N*M

def flops_adv_forward(N, M, ad):
    # Vr = Rscore[N,ad] @ W.T[ad,M]   ~ 2*N*ad*M
    # A_pred = sum( Vr * K_s )[N]     ~ 2*N*M
    return 2.0*N*ad*M + 2.0*N*M

def flops_adv_grad(N, M, ad):
    # gW = ((residual*scale)*K_s).T[M,N] @ Rscore[N,ad] ~ 2*M*N*ad
    # + reg*W ~ M*ad（忽略）
    return 2.0*M*N*ad

def flops_pg(N, M, ad):
    # K_sa = Rscore * Aproj[:,None]   ~ N*ad
    # g = K_s.T[M,N] @ K_sa[N,ad]     ~ 2*M*N*ad
    return N*ad + 2.0*M*N*ad

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
            Kinv_KxT = self.K_inv @ Kx.T                              # [M,N]
            quad = (Kx * Kinv_KxT.T).sum(dim=1)                       # [N]
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

    # SHAP -> ell scaling
    def set_shap_weights(self, phi_np, eps=1e-3, normalize=True,
                         kappa=0.25, ratio_min=0.5, ratio_max=2.0):
        phi = np.asarray(phi_np, dtype=np.float64)
        phi = np.where(np.isfinite(phi), phi, 1.0)
        phi = np.maximum(phi, float(eps))
        if normalize:
            m = float(np.mean(phi))
            if np.isfinite(m) and m > 0:
                phi = phi / m
        rho = 1.0 / np.sqrt(np.maximum(phi, float(eps)))
        rho = np.clip(rho, float(ratio_min), float(ratio_max))
        ratio = (1.0 - float(kappa)) + float(kappa) * rho
        ratio = np.where(np.isfinite(ratio), ratio, 1.0)
        ratio = np.clip(ratio, float(ratio_min), float(ratio_max))
        ratio_t = t(ratio, self.device)
        ell_eff = self.base_ell * ratio_t
        ell_eff = torch.nan_to_num(ell_eff, nan=1.0, posinf=1e3, neginf=1.0)
        self.ell_eff = torch.clamp(ell_eff, min=1e-6, max=1e6)

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
# Value Critic (semi-gradient TD(λ))
# =========================
class KernelValue:
    def __init__(self, centers_t, ell, reg, normalizer, device):
        self.device = device
        self.centers = centers_t.to(device, DTYPE)
        self.M = int(self.centers.shape[0])
        self.ell = t(ell, device)
        self.reg = float(reg)
        self.coef = torch.zeros((self.M,), device=device, dtype=DTYPE)
        self.normalizer = normalizer

    def on_centers_changed(self, new_M):
        old_w = self.coef
        old_M = int(old_w.numel())
        self.M = int(new_M)
        if old_M == 0:
            self.coef = torch.zeros((self.M,), device=self.device, dtype=DTYPE)
        else:
            w_new = torch.zeros((self.M,), device=self.device, dtype=DTYPE)
            keep = min(old_M, self.M)
            if keep > 0: w_new[:keep] = old_w[:keep]
            self.coef = w_new

    def _norm_batch_t(self, S_np_or_t):
        if isinstance(S_np_or_t, np.ndarray):
            X = t(S_np_or_t, self.device)
        else:
            X = S_np_or_t.to(self.device, DTYPE)
        return self.normalizer.transform_torch(X) if self.normalizer is not None else X

    def fit_td_lambda_step(self, S_np, R_np, S2_np, DONE_np, gamma=0.99, lam=0.9, lr=1e-2, steps=1):
        if self.M == 0:
            self.coef = torch.zeros((0,), device=self.device, dtype=DTYPE); return
        S_n  = self._norm_batch_t(S_np)
        S2_n = self._norm_batch_t(S2_np)
        Phi  = rbf_kernel_mat_torch(S_n,  self.centers, self.ell)   # [N,M]
        Phi2 = rbf_kernel_mat_torch(S2_n, self.centers, self.ell)   # [N,M]
        R_t  = t(R_np, self.device).reshape(-1)
        D_t  = t(DONE_np.astype(np.float64), self.device).reshape(-1)
        w = self.coef.clone()
        N = Phi.shape[0]
        for _ in range(max(1, int(steps))):
            z = torch.zeros((self.M,), device=self.device, dtype=DTYPE)
            for ti in range(N):
                phi_t  = Phi[ti]
                phi_tp = Phi2[ti]
                done_t = D_t[ti].item() > 0.5
                v_tp = 0.0 if done_t else float((phi_tp @ w).item())
                delta = R_t[ti].item() + gamma * v_tp - float((phi_t @ w).item())
                z = gamma * lam * z + phi_t
                w = w + lr * (delta * z - self.reg * w)
                if done_t:
                    z.zero_()
        self.coef = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, S_np):
        if self.M == 0:
            return np.zeros((len(S_np),), dtype=np.float64)
        X_n = self._norm_batch_t(S_np)
        K = rbf_kernel_mat_torch(X_n, self.centers, self.ell)
        out = (K @ self.coef).detach().cpu().numpy().astype(np.float64)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

# =========================
# 显式 Advantage 核回归器（compatible features）
# =========================
class AdvantageKernelRegressor:
    """
    目标: 拟合  A_t  ~  (W r_t)^T k(s_t)  ，r_t = (u_t - mu_t)/sigma^2
    我们用小批岭回归的 SGD 更新，并返回 A_proj 供 PG 直接使用（避免二次前向）。
    """
    def __init__(self, actor: KernelGaussianActor, reg=1e-3, ema=0.0, device="cpu"):
        self.device = device
        self.actor = actor
        self.M = actor.M
        self.reg = float(reg)
        self.ema = float(ema)
        ad = actor.alpha.shape[1] if actor.alpha.ndim==2 else 0
        self.W = torch.zeros((self.M, ad), device=device, dtype=DTYPE)
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
    def fit_batch(self, K_s, Rscore, A_target_np, lr=1e-2, clip_grad=None, scale_loss=True):
        """
        入参直接用预计算的:
            K_s: [N,M]   Rscore: [N,ad]   A_target_np: [N]
        返回: (mse_loss, A_proj_np)
        """
        if self.M == 0: return 0.0, np.zeros_like(A_target_np)
        A_t = t(A_target_np, self.device).reshape(-1)           # [N]

        # 预测 A_proj
        Vr = Rscore @ self.W.T                                  # [N,M]
        A_pred = (Vr * K_s).sum(dim=1)                          # [N]
        residual = A_pred - A_t                                 # [N]

        # 可选：尺度 EMA，避免初期爆炸
        if scale_loss:
            cur = float(torch.sqrt(torch.mean(A_t*A_t)).item() + 1e-6)
            self._A_scale_ema = 0.99*self._A_scale_ema + 0.01*cur
            scale = 1.0/max(self._A_scale_ema, 1.0)
        else:
            scale = 1.0

        # gW = ( (residual*scale)*K_s )^T @ Rscore  + reg*W
        coef = (residual * scale).reshape(-1,1)                 # [N,1]
        gW = (coef * K_s).T @ Rscore                             # [M,ad]
        gW = gW + self.reg * self.W

        if clip_grad is not None and clip_grad > 0:
            n = torch.linalg.norm(gW).item()
            if n > clip_grad:
                gW = gW * (clip_grad / max(n,1e-12))

        self.W = self.W - float(lr) * gW
        self.W = torch.nan_to_num(self.W, nan=0.0, posinf=0.0, neginf=0.0)

        return float(torch.mean(residual*residual).item()), A_pred.detach().cpu().numpy().astype(np.float64)

# =========================
# SHAP helpers（保留，只有 shap_enable 才实例化）
# =========================
def _safe_exp(logX):
    m = torch.amax(logX, dim=-1, keepdim=True)
    return torch.exp(logX - m) * torch.exp(m.clamp(max=50.0) - m)

class ShapCME:
    def __init__(self, value_critic: KernelValue, normalizer: StateNormalizer,
                 base_ell, reg=1e-3, ell_scale=1.0, device="cpu"):
        self.vc = value_critic
        self.norm = normalizer
        self.base_ell = t(base_ell, device)
        self.reg = float(reg)
        self.ell_scale = float(ell_scale)
        self.device = device
    @torch.no_grad()
    def phi_for_batch_exact_fast(self, S_targets, bg, n_perms=8, sync_perm=True,
                                 work_dtype=torch.float32):
        dev = self.device; wdt = work_dtype
        if len(bg) == 0 or len(S_targets) == 0:
            d = bg.shape[1] if len(bg) > 0 else (S_targets.shape[1] if len(S_targets) > 0 else 1)
            return np.ones((d,), dtype=np.float64)
        bg_t = torch.as_tensor(bg, device=dev, dtype=wdt)
        tg_t = torch.as_tensor(S_targets, device=dev, dtype=wdt)
        B, d = bg_t.shape;  T = tg_t.shape[0]
        bg_n = self.norm.transform_torch(bg_t.to(self.vc.centers.device, self.vc.centers.dtype)).to(dev, wdt)
        tg_n = self.norm.transform_torch(tg_t.to(self.vc.centers.device, self.vc.centers.dtype)).to(dev, wdt)
        ell = (self.base_ell.to(dev) * self.ell_scale).to(wdt)
        ell = torch.nan_to_num(ell, nan=1.0, posinf=1e3, neginf=1.0)
        ell = torch.clamp(ell, min=1e-6, max=1e6)
        inv_ell2 = (1.0 / torch.clamp(ell, min=1e-6)**2).to(wdt)
        C   = self.vc.centers.to(dev, wdt)
        eta = self.vc.coef.to(dev, wdt).reshape(-1)
        M   = C.shape[0]
        Lbg_bg, Lbg_ctr, Ltg_ctr, Lbg_tgt = [], [], [], []
        for i in range(d):
            sc = inv_ell2[i]
            xi_b = bg_n[:, i]; xi_t = tg_n[:, i]; xi_c = C[:, i]
            Lbg_bg.append( -0.5 * (xi_b[:,None] - xi_b[None,:])**2 * sc )
            Lbg_ctr.append(-0.5 * (xi_b[:,None] - xi_c[None,:])**2 * sc )
            Ltg_ctr.append(-0.5 * (xi_t[:,None] - xi_c[None,:])**2 * sc )
            Lbg_tgt.append(-0.5 * (xi_b[:,None] - xi_t[None,:])**2 * sc )
        log_B_total = torch.stack(Lbg_ctr, dim=0).sum(0)  # [B,M]
        Kbg = _safe_exp(log_B_total)                      # [B,M]
        v_bg_mean = float((Kbg @ eta).mean())
        if not np.isfinite(v_bg_mean): v_bg_mean = 0.0
        perms = [torch.randperm(d, device=dev)] if sync_perm else \
                [torch.randperm(d, device=dev) for _ in range(n_perms)]
        phi = torch.zeros(d, device=dev, dtype=wdt)
        for pidx in range(n_perms):
            perm = perms[0] if sync_perm else perms[pidx]
            logA = torch.zeros((T, M), device=dev, dtype=wdt)
            logB_prefix = torch.zeros((B, M), device=dev, dtype=wdt)
            logKXT = torch.zeros((B, T), device=dev, dtype=wdt)
            KX = torch.ones((B, B), device=dev, dtype=wdt)
            v_prev = torch.full((T,), v_bg_mean, device=dev, dtype=wdt)
            for j in range(d):
                i = int(perm[j].item())
                KX         = KX * _safe_exp(Lbg_bg[i])
                logKXT     = logKXT     + Lbg_tgt[i]
                logA       = logA       + Ltg_ctr[i]
                logB_prefix= logB_prefix+ Lbg_ctr[i]
                n = B
                K_reg = KX + (n * self.reg) * torch.eye(B, device=dev, dtype=wdt)
                KXT   = _safe_exp(logKXT)
                try:
                    L = torch.linalg.cholesky(K_reg)
                    W = torch.cholesky_solve(KXT, L)
                except RuntimeError:
                    W = torch.linalg.lstsq(K_reg, KXT).solution
                A_C = _safe_exp(logA)
                B_R = _safe_exp(log_B_total - logB_prefix)
                S   = B_R.transpose(0,1) @ W
                v_now = (A_C * S.transpose(0,1)) @ eta
                inc = (v_now - v_prev).mean()
                if torch.isfinite(inc): phi[i] += inc
                v_prev = v_now
        phi = (phi / max(1, n_perms))
        if not torch.isfinite(phi).all(): phi = torch.ones_like(phi)
        return phi.detach().cpu().numpy()

class ShapKME:
    def __init__(self, value_critic: KernelValue, normalizer: StateNormalizer,
                 base_ell, ell_scale=1.0, device="cpu"):
        self.vc = value_critic
        self.norm = normalizer
        self.base_ell = t(base_ell, device)
        self.ell_scale = float(ell_scale)
        self.device = device
    @torch.no_grad()
    def phi_for_batch_fast(self, S_targets, bg, n_perms=8, sync_perm=True,
                           work_dtype=torch.float32):
        dev = self.device; wdt = work_dtype
        if len(bg) == 0 or len(S_targets) == 0:
            d = bg.shape[1] if len(bg) > 0 else (S_targets.shape[1] if len(S_targets) > 0 else 1)
            return np.ones((d,), dtype=np.float64)
        bg_t = torch.as_tensor(bg, device=dev, dtype=wdt)
        tg_t = torch.as_tensor(S_targets, device=dev, dtype=wdt)
        B, d = bg_t.shape;  T = tg_t.shape[0]
        bg_n = self.norm.transform_torch(bg_t.to(self.vc.centers.device, self.vc.centers.dtype)).to(dev, wdt)
        tg_n = self.norm.transform_torch(tg_t.to(self.vc.centers.device, self.vc.centers.dtype)).to(dev, wdt)
        ell = (self.base_ell.to(dev) * self.ell_scale).to(wdt)
        ell = torch.nan_to_num(ell, nan=1.0, posinf=1e3, neginf=1.0)
        ell = torch.clamp(ell, min=1e-6, max=1e6)
        inv_ell2 = (1.0 / torch.clamp(ell, min=1e-6)**2).to(wdt)
        C   = self.vc.centers.to(dev, wdt)
        eta = self.vc.coef.to(dev, wdt).reshape(-1)
        M   = C.shape[0]
        if M == 0:
            return np.ones((d,), dtype=np.float64)
        Lbg_ctr, Ltg_ctr = [], []
        for i in range(d):
            sc = inv_ell2[i]
            xi_b = bg_n[:, i]; xi_t = tg_n[:, i]; xi_c = C[:, i]
            Lbg_ctr.append(-0.5 * (xi_b[:,None] - xi_c[None,:])**2 * sc)
            Ltg_ctr.append(-0.5 * (xi_t[:,None] - xi_c[None,:])**2 * sc)
        log_B_total = torch.stack(Lbg_ctr, dim=0).sum(0)
        v_bg_mean = float((_safe_exp(log_B_total) @ eta).mean())
        if not np.isfinite(v_bg_mean): v_bg_mean = 0.0
        perms = [torch.randperm(d, device=dev)] if sync_perm else \
                [torch.randperm(d, device=dev) for _ in range(n_perms)]
        phi = torch.zeros(d, device=dev, dtype=wdt)
        for pidx in range(n_perms):
            perm = perms[0] if sync_perm else perms[pidx]
            logA = torch.zeros((T, M), device=dev, dtype=wdt)
            logB_prefix = torch.zeros((B, M), device=dev, dtype=wdt)
            v_prev = torch.full((T,), v_bg_mean, device=dev, dtype=wdt)
            for j in range(d):
                i = int(perm[j].item())
                logA         = logA         + Ltg_ctr[i]
                logB_prefix  = logB_prefix  + Lbg_ctr[i]
                A_C = _safe_exp(logA)
                Q = _safe_exp(log_B_total - logB_prefix).mean(dim=0)
                v_now = A_C @ (eta * Q)
                inc = (v_now - v_prev).mean()
                if torch.isfinite(inc): phi[i] += inc
                v_prev = v_now
        phi = phi / max(1, n_perms)
        if not torch.isfinite(phi).all(): phi = torch.ones_like(phi)
        return phi.detach().cpu().numpy()

# =========================
# Rollout + Eval
# =========================
def collect_rollouts(env, actor, episodes, horizon, gamma, noise, diag_actions=False):
    with torch.no_grad():
        traj = []; states_for_dict = []
        obs_dim = env.observation_space.shape[0]
        act_sample = env.action_space.sample()
        for _ in range(episodes):
            s, info = _reset_env(env)
            ep = {"s": [], "s2": [], "u": [], "mu": [], "r": [], "done": []}
            if diag_actions:
                ep["a"] = []
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
                if diag_actions: ep["a"].append(np.asarray(a, dtype=np.float64))
                ep["u"].append(np.asarray(u, dtype=np.float64))      # pre-tanh
                ep["mu"].append(np.asarray(mu, dtype=np.float64))    # pre-tanh
                ep["r"].append(float(r)); ep["done"].append(bool(done))
                states_for_dict.append(s)
                s = s2
                if done: break
            traj.append(ep)

        S   = np.asarray([s for ep in traj for s in ep["s"]],   dtype=np.float64)
        S2  = np.asarray([s2 for ep in traj for s2 in ep["s2"]], dtype=np.float64)
        D   = np.asarray([d for ep in traj for d in ep["done"]], dtype=bool)
        U   = np.asarray([u for ep in traj for u in ep["u"]],    dtype=np.float64)  # pre-tanh
        MU  = np.asarray([m for ep in traj for m in ep["mu"]],   dtype=np.float64)  # pre-tanh
        R   = np.asarray([r for ep in traj for r in ep["r"]],    dtype=np.float64)
        lens= np.asarray([len(ep["r"]) for ep in traj], dtype=np.int32)

        for arr in (S, S2, U, MU, R):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        G = []
        idx = 0
        for L in lens:
            G.extend(discount_cumsum_np(R[idx:idx+L], gamma))
            idx += L
        G = np.asarray(G, dtype=np.float64)

        if diag_actions:
            A = np.asarray([a for ep in traj for a in ep["a"]], dtype=np.float64)
        else:
            A = None

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
# Train one seed (explicit Advantage regressor) + profiling
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
    *_, states_pool = collect_rollouts(
        env_tr, tmp_actor, WARM_EP, args.horizon, args.gamma, args.noise, diag_actions=False)
    states_pool = np.nan_to_num(states_pool, nan=0.0, posinf=0.0, neginf=0.0)
    state_mean = states_pool.mean(axis=0)
    state_std  = states_pool.std(axis=0) + 1e-8
    normalizer = StateNormalizer(state_mean, state_std, device)

    base_ell = np.ones_like(state_std) * float(args.actor_ell)

    # 初始字典
    states_pool_n = normalizer.transform(states_pool)
    init_k_a = min(args.centers_init_a, args.max_centers_a)
    init_k_v = min(args.centers_init_v, args.max_centers_v)
    centers_init_a = kmeanspp_init(states_pool_n, init_k_a, seed=seed)
    centers_init_v = kmeanspp_init(states_pool_n, init_k_v, seed=seed+99)

    # ALD 字典
    ald_A = ALDDictionary(ell=base_ell, eps=args.ald_eps_a, max_centers=args.max_centers_a,
                          jitter=1e-5, device=device)
    ald_V = ALDDictionary(ell=base_ell, eps=args.ald_eps_v, max_centers=args.max_centers_v,
                          jitter=1e-5, device=device)
    ald_A.initialize(centers_init_a)
    ald_V.initialize(centers_init_v)

    # 模型
    actor = KernelGaussianActor(ald_A.centers, ell=base_ell, sigma=args.sigma,
                                action_low=a_low, action_high=a_high,
                                normalizer=normalizer, device=device)
    v_critic = KernelValue(ald_V.centers, ell=base_ell, reg=args.v_reg,
                           normalizer=normalizer, device=device)
    a_reg = AdvantageKernelRegressor(actor, reg=args.adv_reg, ema=args.adv_ema, device=device)

    # SHAP 期望估计器（仅在开启时实例化）
    shap_est = None
    if args.shap_enable:
        if args.shap_backend == "cme":
            shap_est = ShapCME(v_critic, normalizer, base_ell,
                               reg=args.cme_reg, ell_scale=args.cme_ell_scale,
                               device=device)
        else:
            shap_est = ShapKME(v_critic, normalizer, base_ell,
                               ell_scale=args.cme_ell_scale, device=device)
    shap_history = []
    phi_ema_vec = None

    if seed == args.seed:
        print(f"[Device] {device} | dtype={DTYPE}")
        print(f"[Info] env={args.env} | horizon={args.horizon} | ep/update={args.episodes_per_update} "
              f"| init_A={ald_A.M}/{args.max_centers_a} | init_V={ald_V.M}/{args.max_centers_v} "
              f"| base_ell={args.actor_ell} | Value=TD(λ) | Adv=explicit kernel ridge | "
              f"PG=vanilla(tanh) | SHAP={'ON' if args.shap_enable else 'OFF'}")

    # ---- profiling containers ----
    prof_rows = [] if args.profile_epoch else None
    prof_avg_rows = [] if args.profile_epoch else None
    W = int(max(1, args.profile_avg_window))

    eval_curve = np.full((args.epochs+1,), np.nan, dtype=np.float64)

    for ep in trange(args.epochs+1):
        _t0 = _ts(device)
        t_roll=t_dict=t_value=t_pre=t_adv=t_pg=t_actor=t_eval=0.0
        f_total = 0.0

        # restart 概率退火
        if args.restart_anneal_end > 0:
            frac = max(0.0, 1.0 - ep / max(1, args.restart_anneal_end))
            env_tr._p_restart = args.restart_p * frac
        else:
            env_tr._p_restart = args.restart_p

        # ALD eps 退火
        if args.ald_eps_anneal_end_a > 0:
            frac_eps_a = max(0.0, 1.0 - ep / max(1, args.ald_eps_anneal_end_a))
            ald_A.eps = max(args.ald_eps_final_a, args.ald_eps_a * frac_eps_a)
        if args.ald_eps_anneal_end_v > 0:
            frac_eps_v = max(0.0, 1.0 - ep / max(1, args.ald_eps_anneal_end_v))
            ald_V.eps = max(args.ald_eps_final_v, args.ald_eps_v * frac_eps_v)

        # ===== 采样 =====
        _a = _ts(device)
        S, S2, DONE, A, U, MU, R, G, lens, states_batch = collect_rollouts(
            env_tr, actor, args.episodes_per_update, args.horizon, args.gamma, args.noise,
            diag_actions=args.diag_actions)
        t_roll = _ts(device) - _a

        # ===== 字典扩展/修剪 =====
        _a = _ts(device)
        do_add = (ep % max(1, args.ald_add_interval) == 0)
        if do_add:
            S_n = normalizer.transform(states_batch)
            added_A = ald_A.add_from_batch(S_n, max_add_per_epoch=args.ald_max_add_per_epoch_a)
            if added_A > 0:
                actor.centers = ald_A.centers; actor.on_added_centers(added_A)
                a_reg.actor = actor; a_reg.on_centers_changed(ald_A.M)
            added_V = ald_V.add_from_batch(S_n, max_add_per_epoch=args.ald_max_add_per_epoch_v)
            if added_V > 0:
                v_critic.centers = ald_V.centers
                v_critic.on_centers_changed(ald_V.M)

        do_prune = (ep % max(1, args.ald_prune_interval) == 0)
        if do_prune:
            if ald_A.M > args.max_centers_a:
                keep_A = np.argsort(actor.alpha.detach().cpu().abs().numpy().sum(axis=1))[::-1][:args.max_centers_a]
                keep_A.sort()
                ald_A.prune_to_indices(keep_A)
                actor.centers = ald_A.centers; actor.on_pruned_centers(keep_A)
                a_reg.actor = actor; a_reg.on_centers_changed(ald_A.M)
            if ald_V.M > args.max_centers_v:
                old_M = ald_V.M
                if v_critic.coef.numel() == old_M:
                    keep_V = np.argsort(np.abs(v_critic.coef.detach().cpu().numpy()))[::-1][:args.max_centers_v]
                else:
                    keep_V = np.arange(args.max_centers_v)
                keep_V.sort()
                if v_critic.coef.numel() == old_M:
                    v_critic.coef = v_critic.coef[torch.as_tensor(keep_V, device=v_critic.device, dtype=torch.long)]
                ald_V.prune_to_indices(keep_V)
                v_critic.centers = ald_V.centers
                v_critic.on_centers_changed(ald_V.M)
        t_dict = _ts(device) - _a

        # ===== Value Critic：TD(λ) =====
        _a = _ts(device)
        v_critic.fit_td_lambda_step(S, R, S2, DONE, gamma=args.gamma,
                                    lam=args.td_lambda, lr=args.lr_v, steps=args.ts_critic_steps)
        V = v_critic.predict(S)
        t_value = _ts(device) - _a

        # A_hat
        Ahat = G - V
        if args.adv_clip is not None and args.adv_clip > 0:
            Ahat = np.clip(Ahat, -float(args.adv_clip), float(args.adv_clip))
        std = Ahat.std()
        if std > 1e-6:
            Ahat = (Ahat - Ahat.mean()) / (std + 1e-8)

        # ===== 预计算 K_s 与 score =====
        _a = _ts(device)
        K_s   = actor.kmat(S)                               # [N,M_a]
        U_t   = t(U, device).reshape(len(S), -1)
        MU_t  = t(MU,device).reshape(len(S), -1)
        sigma2= max(actor.sigma**2, 1e-12)
        Rscore= (U_t - MU_t) / sigma2                       # [N,ad]
        t_pre = _ts(device) - _a

        # FLOPs: K_s
        N = int(S.shape[0]); d = int(S.shape[1])
        M_a = int(ald_A.M); M_v = int(ald_V.M)
        ad  = int(actor.alpha.shape[1]) if actor.alpha.ndim==2 else 0
        f_total += flops_rbf_kmat(N, M_a, d)

        # ===== Advantage 显式核回归 =====
        _a = _ts(device)
        _, A_proj = a_reg.fit_batch(K_s, Rscore, Ahat, lr=args.adv_lr,
                                    clip_grad=args.adv_clip_grad, scale_loss=True)
        t_adv = _ts(device) - _a

        # FLOPs: Adv 前向+梯度
        f_total += flops_adv_forward(N, M_a, ad) + flops_adv_grad(N, M_a, ad)

        # ===== SHAP（按需，不计 FLOPs）=====
        shap_applied = False
        if args.shap_enable and shap_est is not None and (ep % max(1, args.shap_interval) == 0) and (ep >= args.shap_warmup):
            B = min(len(states_batch), args.shap_bg_size)
            Tt = min(len(states_batch), args.shap_targets)
            idx_bg = np.random.choice(len(states_batch), size=B, replace=(B > len(states_batch)))
            idx_tg = np.random.choice(len(states_batch), size=Tt, replace=(Tt > len(states_batch)))
            bg = states_batch[idx_bg]; tg = states_batch[idx_tg]
            work_dtype = torch.float32 if args.shap_dtype.lower()=="float32" else torch.float64
            if isinstance(shap_est, ShapCME):
                phi_avg = shap_est.phi_for_batch_exact_fast(tg, bg, n_perms=args.shap_perms,
                                                            sync_perm=bool(args.shap_sync_perm),
                                                            work_dtype=work_dtype)
            else:
                phi_avg = shap_est.phi_for_batch_fast(tg, bg, n_perms=args.shap_perms,
                                                      sync_perm=bool(args.shap_sync_perm),
                                                      work_dtype=work_dtype)
            if args.shap_ema > 0.0:
                if (phi_ema_vec is None):
                    phi_ema_vec = phi_avg.copy()
                else:
                    phi_ema_vec = float(args.shap_ema)*phi_ema_vec + (1.0-float(args.shap_ema))*phi_avg
                phi_to_apply = phi_ema_vec.copy()
            else:
                phi_to_apply = phi_avg.copy()
            if args.shap_topk > 0 and args.shap_topk < len(phi_to_apply):
                idx = np.argsort(np.abs(phi_to_apply - 1.0))[::-1]
                keep = idx[:args.shap_topk]
                mask = np.zeros_like(phi_to_apply, dtype=bool); mask[keep] = True
                phi_to_apply = np.where(mask, phi_to_apply, 1.0)
            K_old = actor.kmat(tg); mu_old = (K_old @ actor.alpha).detach()
            old_ell = actor.ell_eff.clone()
            actor.set_shap_weights(phi_to_apply, eps=args.shap_floor, normalize=True,
                                   kappa=args.shap_kappa,
                                   ratio_min=args.shap_ratio_min, ratio_max=args.shap_ratio_max)
            K_new = actor.kmat(tg); mu_new = (K_new @ actor.alpha).detach()
            sigma2 = max(actor.sigma**2, 1e-12)
            kl_est = float(((mu_new - mu_old)**2).mean() / (2.0 * sigma2))
            if not np.isfinite(kl_est): kl_est = 0.0
            if args.shap_tr_kl > 0.0 and kl_est > args.shap_tr_kl:
                scale = math.sqrt(max(1e-12, args.shap_tr_kl)/kl_est)
                actor.ell_eff = old_ell
                actor.set_shap_weights(phi_to_apply, eps=args.shap_floor, normalize=True,
                                       kappa=float(args.shap_kappa)*scale,
                                       ratio_min=args.shap_ratio_min, ratio_max=args.shap_ratio_max)
                if seed == args.seed:
                    print(f"[SHAP-TR] structural KL {kl_est:.4g} > {args.shap_tr_kl:.4g}, shrink kappa by {scale:.3f}")
            shap_applied = True

        # ===== Policy gradient（用 A_proj）=====
        _a = _ts(device)
        Aproj_t = t(A_proj, device).reshape(-1,1)                # [N,1]
        K_sa = Rscore * Aproj_t                                  # [N,ad]
        g = K_s.T @ K_sa / max(len(S),1)                         # [M,ad]
        t_pg = _ts(device) - _a

        # FLOPs: PG
        f_total += flops_pg(N, M_a, ad)

        # ===== Actor update =====
        _a = _ts(device)
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
        t_actor = _ts(device) - _a

        # sigma 退火
        if ep < args.sigma_anneal_end:
            frac_s = 1.0 - ep / max(1, args.sigma_anneal_end)
            actor.sigma = args.sigma_anneal_to + (args.sigma - args.sigma_anneal_to) * frac_s
        else:
            actor.sigma = max(args.sigma_anneal_to, 1e-3)

        # 评估 & 打印
        _a = _ts(device)
        if (ep % args.eval_interval == 0) or (ep == 0):
            eval_h = min(args.eval_horizon, args.horizon)
            eval_ret, succ_rate = evaluate_policy(env_ev, actor, args.eval_noise,
                                                  episodes=args.eval_episodes, horizon=eval_h,
                                                  succ_threshold=250.0)
            marker = "A*"
            shap_note = ""
            if args.shap_enable and shap_est is not None and (ep % max(1, args.shap_interval) == 0) and (ep >= args.shap_warmup):
                shap_note = f" W↑" if shap_applied else " W-"
            alpha_norm = float(torch.linalg.norm(actor.alpha).item()) if actor.alpha.numel() else 0.0
            Wnorm = float(torch.linalg.norm(a_reg.W).item()) if a_reg.W.numel() else 0.0
            print(f"{ep:04d} {marker}{shap_note} | eval={eval_ret:.2f} | succ={succ_rate:.2f} "
                  f"| M_A={ald_A.M} M_V={ald_V.M} | lr={lr_step:.3g} step={step_norm:.3g} "
                  f"| sigma={actor.sigma:.3g} | ||alpha||={alpha_norm:.3g} | ||W||={Wnorm:.3g} | device={device}")
            eval_curve[ep] = eval_ret
            if args.diag_actions and A is not None:
                print("[diag]", "|a|>0.98:", np.mean(np.abs(A) > 0.98))
        t_eval = _ts(device) - _a

        t_total = _ts(device) - _t0

        # ===== profiling rows =====
        if args.profile_epoch:
            row = {
                "epoch": ep,
                "time_total_ms": 1000.0*t_total,
                "roll_ms": 1000.0*t_roll,
                "dict_ms": 1000.0*t_dict,
                "value_ms":1000.0*t_value,
                "precomp_ms":1000.0*t_pre,
                "advfit_ms":1000.0*t_adv,
                "pg_ms": 1000.0*t_pg,
                "actor_ms": 1000.0*t_actor,
                "eval_ms": 1000.0*t_eval,
                "N": N, "d": d, "M_a": M_a, "M_v": M_v, "ad": ad,
                "flops_total": float(f_total)
            }
            prof_rows.append(row)

            win = prof_rows[-W:]
            def _avg(key): return float(np.mean([r[key] for r in win])) if len(win)>0 else 0.0
            print(
                f"[EP{ep:04d} AVG/{len(win)}] "
                f"total={_avg('time_total_ms'):.1f}ms | roll={_avg('roll_ms'):.1f} | "
                f"dict={_avg('dict_ms'):.1f} | value={_avg('value_ms'):.1f} | pre={_avg('precomp_ms'):.1f} | "
                f"adv={_avg('advfit_ms'):.1f} | pg={_avg('pg_ms'):.1f} | actor={_avg('actor_ms'):.1f} | eval={_avg('eval_ms'):.1f} || "
                f"FLOPs total≈{human_flops(_avg('flops_total'))}"
            )
            prof_avg_rows.append({
                "epoch": ep, "avg_window": len(win),
                "time_total_ms": _avg('time_total_ms'),
                "roll_ms": _avg('roll_ms'),
                "dict_ms": _avg('dict_ms'),
                "value_ms": _avg('value_ms'),
                "precomp_ms": _avg('precomp_ms'),
                "advfit_ms": _avg('advfit_ms'),
                "pg_ms": _avg('pg_ms'),
                "actor_ms": _avg('actor_ms'),
                "eval_ms": _avg('eval_ms'),
                "flops_total": _avg('flops_total'),
            })

    # 前向填充
    last = np.nan
    for i in range(eval_curve.shape[0]):
        if np.isnan(eval_curve[i]): eval_curve[i] = last
        else: last = eval_curve[i]

    # ===== dump profiling CSVs =====
    if args.profile_epoch and prof_rows:
        os.makedirs(args.out_dir, exist_ok=True)
        # 原始逐 epoch
        out_prof_raw = os.path.join(args.out_dir, f"seed{seed}_" + args.profile_csv)
        keys_raw = ["epoch","time_total_ms","roll_ms","dict_ms","value_ms","precomp_ms",
                    "advfit_ms","pg_ms","actor_ms","eval_ms","N","d","M_a","M_v","ad","flops_total"]
        with open(out_prof_raw, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys_raw); w.writeheader()
            for row in prof_rows: w.writerow(row)
        # 平均（滑动窗口）
        out_prof_avg = os.path.join(args.out_dir, f"seed{seed}_avgW{W}_" + args.profile_avg_csv)
        keys_avg = ["epoch","avg_window","time_total_ms","roll_ms","dict_ms","value_ms","precomp_ms",
                    "advfit_ms","pg_ms","actor_ms","eval_ms","flops_total"]
        with open(out_prof_avg, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys_avg); w.writeheader()
            for row in prof_avg_rows: w.writerow(row)
        print(f"[PROFILE] raw per-epoch -> {out_prof_raw}")
        print(f"[PROFILE] averaged (window={W}) -> {out_prof_avg}")

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
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99)

    default_dev = "cpu"
    p.add_argument("--device", type=str, default=default_dev)

    p.add_argument("--actor_ell", type=float, default=1.0)

    p.add_argument("--centers_init_a", type=int, default=192)
    p.add_argument("--max_centers_a", type=int, default=3072)
    p.add_argument("--ald_eps_a", type=float, default=1e-3)
    p.add_argument("--ald_eps_final_a", type=float, default=7e-4)
    p.add_argument("--ald_eps_anneal_end_a", type=int, default=600)
    p.add_argument("--ald_max_add_per_epoch_a", type=int, default=24)

    p.add_argument("--centers_init_v", type=int, default=192)
    p.add_argument("--max_centers_v", type=int, default=3072)
    p.add_argument("--ald_eps_v", type=float, default=1e-3)
    p.add_argument("--ald_eps_final_v", type=float, default=7e-4)
    p.add_argument("--ald_eps_anneal_end_v", type=int, default=600)
    p.add_argument("--ald_max_add_per_epoch_v", type=int, default=24)

    p.add_argument("--ald_add_interval", type=int, default=3)
    p.add_argument("--ald_prune_interval", type=int, default=50)

    p.add_argument("--sigma", type=float, default=0.55)
    p.add_argument("--sigma_anneal_to", type=float, default=0.30)
    p.add_argument("--sigma_anneal_end", type=int, default=1200)

    p.add_argument("--restart_p", type=float, default=0.000)
    p.add_argument("--restart_anneal_end", type=int, default=0)

    p.add_argument("--lam_adv", type=float, default=3e-3)   # not used now
    p.add_argument("--ema_adv", type=float, default=0.97)   # not used now
    p.add_argument("--v_reg", type=float, default=1e-3)

    p.add_argument("--ts_actor_delay", type=int, default=1)
    p.add_argument("--ts_critic_steps", type=int, default=3)
    p.add_argument("--lr_actor", type=float, default=0.05)
    p.add_argument("--pg_clip", type=float, default=0.6)
    p.add_argument("--pg_adapt_target", type=float, default=0.0,
                   help=">0 时将 PG 范数重标定到目标值；默认关闭。")
    p.add_argument("--pg_scale", type=float, default=2.0,
                   help="对 vanilla PG 梯度做整体乘法缩放。")

    p.add_argument("--lr_v", type=float, default=2.5e-3)
    p.add_argument("--td_lambda", type=float, default=0.90)

    # Advantage 显式核回归器超参
    p.add_argument("--adv_lr", type=float, default=5e-3,
                   help="Advantage 核回归器的学习率（SGD）。")
    p.add_argument("--adv_reg", type=float, default=1e-3,
                   help="岭回归正则 λ。")
    p.add_argument("--adv_clip_grad", type=float, default=5.0,
                   help="显式回归器的梯度裁剪范数。")

    # SHAP
    p.add_argument("--shap_backend", type=str, choices=["cme","kme"], default="cme")
    p.add_argument("--shap_enable", action="store_true")
    p.add_argument("--shap_interval", type=int, default=25)
    p.add_argument("--shap_perms", type=int, default=8)
    p.add_argument("--shap_targets", type=int, default=64)
    p.add_argument("--shap_bg_size", type=int, default=256)
    p.add_argument("--shap_floor", type=float, default=0.3)
    p.add_argument("--shap_ema", type=float, default=0.95)
    p.add_argument("--cme_reg", type=float, default=2e-3)
    p.add_argument("--cme_ell_scale", type=float, default=1.0)
    p.add_argument("--shap_dtype", type=str, choices=["float32","float64"], default="float32")
    p.add_argument("--shap_sync_perm", type=int, default=1)
    p.add_argument("--shap_kappa", type=float, default=0.08)
    p.add_argument("--shap_ratio_min", type=float, default=0.7)
    p.add_argument("--shap_ratio_max", type=float, default=1.4)
    p.add_argument("--shap_tr_kl", type=float, default=0.002)
    p.add_argument("--shap_warmup", type=int, default=60)
    p.add_argument("--shap_topk", type=int, default=4)
    p.add_argument("--adv_ema", type=float, default=0.0,
               help="对 Advantage 目标的尺度/统计做 EMA 平滑的强度（当前实现未使用，可留作占位）。")

    # 诊断与评估加速
    p.add_argument("--diag_actions", action="store_true",
                   help="开启后收集并打印 |a|>0.98 诊断；默认关闭以节省时间/内存。")
    p.add_argument("--eval_horizon", type=int, default=400,
                   help="评估时使用的较短 horizon，以加速评估。")

    p.add_argument("--save_shap", action="store_true")
    p.add_argument("--save_ell_eff", action="store_true")
    p.add_argument("--shap_out_dir", type=str, default="shap_logs")
    p.add_argument("--shap_out_tag", type=str, default="exp1")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--eval_noise", type=float, default=0.0)

    p.add_argument("--seed", type=int, default=512)
    p.add_argument("--multi_seeds", type=int, default=1)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--eval_interval", type=int, default=10)

    p.add_argument("--max_alpha_norm", type=float, default=1e4)

    p.add_argument("--out_csv", type=str, default="eval_returns.csv")
    p.add_argument("--out_dir", type=str, default="return_logs")
    p.add_argument("--out_png", type=str, default="eval_curve.png")

    p.add_argument("--adv_clip", type=float, default=10.0,
                   help="Clip A_hat in [-adv_clip, adv_clip] before standardization; <=0 disables.")

    # ===== profiling（平均输出） =====
    p.add_argument("--profile_epoch", action="store_true",
                   help="Record runtime & FLOPs; prints sliding-average instead of per-epoch only.")
    p.add_argument("--profile_avg_window", type=int, default=10,
                   help="Sliding window size for averaging runtime & FLOPs in logs/CSV.")
    p.add_argument("--profile_csv", type=str, default="epoch_profile_raw.csv",
                   help="CSV filename for raw per-epoch profiling (saved under out_dir).")
    p.add_argument("--profile_avg_csv", type=str, default="epoch_profile_avg.csv",
                   help="CSV filename for averaged profiling (saved under out_dir).")

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

    out_csv_path = os.path.join(args.out_dir, f"eval_return_seed{args.seed}_{args.shap_backend}_{args.shap_enable}_{args.noise}.csv")
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
    title_suffix = f" + SHAP-{args.shap_backend.upper()}" if args.shap_enable else ""
    plt.title(f"{args.env} | mean over {len(seeds)} seeds (explicit Adv kernel reg + tanh-squash{title_suffix})")
    plt.legend(); plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"[Done] saved CSV to {out_csv_path}, figure to {args.out_png} | time={time.time()-t0:.1f}s | device={device}")

if __name__ == "__main__":
    main()
