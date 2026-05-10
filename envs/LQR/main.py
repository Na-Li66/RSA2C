import argparse, math, time, random, os, csv
import numpy as np
import matplotlib.pyplot as plt

# Gym/Gymnasium 兼容
try:
    import gymnasium as gym
except ImportError:
    import gym

# ---------- PyTorch ----------
import torch
from torch import nn

# ---------- SciPy（用于 DARE） ----------
from scipy.linalg import solve_discrete_are

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
    Xn = X / ell
    Yn = Y / ell
    XX = (Xn**2).sum(dim=1, keepdim=True)
    YY = (Yn**2).sum(dim=1, keepdim=True).transpose(0,1)
    d2 = XX + YY - 2.0 * (Xn @ Yn.transpose(0,1))
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
def _format_action(env, a_scalar):
    sample = env.action_space.sample()
    if isinstance(sample, np.ndarray):
        arr = np.asarray(a_scalar, dtype=sample.dtype)
        if sample.shape == (1,):
            return arr.reshape(1)
        return arr.reshape(sample.shape)
    return float(a_scalar)

def _step_env(env, a_scalar, allow_restart=True):
    res = env.step(_format_action(env, a_scalar))
    if isinstance(res, tuple) and len(res) == 5:  # gymnasium
        s2, r, terminated, truncated, info = res
        done = bool(terminated) or bool(truncated)
    else:
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
# Linearized Inverted Pendulum LQR Env
# =========================
class LinearizedInvertedPendulumLQREnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 dt=0.02,
                 gamma=0.99,
                 x_max=10.0,
                 u_max=10.0,
                 max_episode_steps=200,
                 process_noise_std=0.0):
        super().__init__()
        self.dt = float(dt)
        self.gamma = float(gamma)

        # 物理参数
        M = 1.0   # cart mass
        m = 0.1   # pole mass
        l = 0.5   # half pole length
        g = 9.8

        # Linearized dynamics around upright (theta=0)
        A_c = np.array([
            [0.0,              1.0,            0.0,                       0.0],
            [0.0,              0.0,         -(m * g) / M,                 0.0],
            [0.0,              0.0,            0.0,                       1.0],
            [0.0,              0.0,  (M + m) * g / (M * l),               0.0]
        ], dtype=np.float64)

        B_c = np.array([
            [0.0],
            [1.0 / M],
            [0.0],
            [-1.0 / (M * l)]
        ], dtype=np.float64)

        # Euler 离散
        self.A = np.eye(4) + A_c * self.dt
        self.B = B_c * self.dt

        # LQR cost matrices
        self.Q = np.diag([1.0, 0.1, 10.0, 0.1]).astype(np.float64)
        self.R = np.array([[0.01]], dtype=np.float64)

        self.n = 4
        self.m = 1

        self.x_max = float(x_max)
        self.u_max = float(u_max)
        self.max_episode_steps = int(max_episode_steps)
        self.process_noise_std = float(process_noise_std)

        high_x = np.ones(self.n, dtype=np.float32) * self.x_max
        high_u = np.ones(self.m, dtype=np.float32) * self.u_max

        self.observation_space = gym.spaces.Box(
            low=-high_x, high=high_x, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-high_u, high=high_u, dtype=np.float32
        )

        self.state = None
        self._step_count = 0

        self._P_opt = None
        self._K_opt = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        low = np.array([-0.5, -0.5, -0.1, -0.5])
        high = np.array([0.5, 0.5, 0.1, 0.5])
        rng = np.random.default_rng(seed)
        self.state = rng.uniform(low=low, high=high).astype(np.float64)
        self._step_count = 0
        obs = self.state.astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        self._step_count += 1
        u = np.asarray(action, dtype=np.float64).reshape(self.m)
        u = np.clip(u, -self.u_max, self.u_max)

        x = self.state
        w = self.process_noise_std * np.random.randn(self.n)
        x_next = self.A @ x + self.B @ u + w

        cost = x.T @ self.Q @ x + u.T @ self.R @ u
        reward = -float(cost)

        self.state = x_next
        truncated = (
            np.any(np.abs(self.state) > self.x_max) or
            self._step_count >= self.max_episode_steps
        )
        terminated = False

        obs = self.state.astype(np.float32)
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    # LQR analytic solution with discount gamma
    def compute_optimal_gain(self):
        if self._P_opt is not None and self._K_opt is not None:
            return self._K_opt, self._P_opt
        sqrt_g = math.sqrt(self.gamma)
        A_bar = sqrt_g * self.A
        B_bar = sqrt_g * self.B
        P = solve_discrete_are(A_bar, B_bar, self.Q, self.R)
        K = np.linalg.solve(self.R + B_bar.T @ P @ B_bar,
                            B_bar.T @ P @ A_bar)
        self._P_opt = P
        self._K_opt = K
        return K, P

    def optimal_value(self, x):
        """
        返回 discounted cost-to-go J*(x) = x^T P x
        """
        _, P = self.compute_optimal_gain()
        x = np.asarray(x, dtype=np.float64).reshape(self.n)
        return float(x.T @ P @ x)

    def optimal_action(self, x):
        K, _ = self.compute_optimal_gain()
        x = np.asarray(x, dtype=np.float64).reshape(self.n)
        u = -K @ x
        return np.clip(u, -self.u_max, self.u_max)

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

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean) / self.std

    def transform_torch(self, X_t):
        return (X_t - self.t_mean) / self.t_std

# =========================
# ALD Sparse Dictionary (torch)
# =========================
class ALDDictionary:
    def __init__(self, ell, eps, max_centers, jitter, device):
        self.device = device
        self.ell = t(ell, device)
        self.eps2 = float(eps)
        self.max_centers = int(max_centers)
        self.jitter = float(jitter)
        self.centers = torch.zeros((0,1), device=device, dtype=DTYPE)
        self.K_inv  = torch.zeros((0,0), device=device, dtype=DTYPE)

    @property
    def M(self):
        return int(self.centers.shape[0])

    def initialize(self, centers_np):
        self.centers = t(centers_np, self.device)
        if self.M == 0:
            self.K_inv = torch.zeros((0,0), device=self.device, dtype=DTYPE)
            return
        K = rbf_kernel_mat_torch(self.centers, self.centers, self.ell)
        K = 0.5*(K + K.T) + self.jitter*torch.eye(self.M, device=self.device, dtype=DTYPE)
        self.K_inv = torch.linalg.solve(K, torch.eye(self.M, device=self.device, dtype=DTYPE))

    def _kvec(self, x_n_t):
        if self.M == 0: return torch.zeros((0,), device=self.device, dtype=DTYPE)
        return rbf_kernel_vec_torch(x_n_t, self.centers, self.ell)

    def add_point(self, x_n_t):
        if self.M >= self.max_centers: return False
        if self.M == 0:
            self.centers = x_n_t.reshape(1,-1).clone()
            self.K_inv = torch.tensor([[1.0/(1.0+self.jitter)]], device=self.device, dtype=DTYPE)
            return True
        kx = self._kvec(x_n_t)
        s  = (1.0 + self.jitter) - (kx @ (self.K_inv @ kx)).item()
        if not np.isfinite(s): s = 0.0
        if s <= self.eps2: return False
        s = max(s, 1e-12)
        s_t = torch.tensor(s, device=self.device, dtype=DTYPE)
        Kinv_k = self.K_inv @ kx
        top_left  = self.K_inv + (Kinv_k[:,None] @ Kinv_k[None,:]) / s_t
        top_right = -Kinv_k[:,None] / s_t
        bot_left  = -Kinv_k[None,:] / s_t
        bot_right = torch.tensor([[1.0/s]], device=self.device, dtype=DTYPE)
        self.K_inv = torch.cat([torch.cat([top_left, top_right], dim=1),
                                torch.cat([bot_left, bot_right], dim=1)], dim=0)
        self.centers = torch.cat([self.centers, x_n_t.reshape(1,-1)], dim=0)
        return True

    def add_from_batch(self, S_n_np, max_add_per_epoch=64):
        added = 0
        idxs = np.arange(S_n_np.shape[0])
        np.random.shuffle(idxs)
        for i in idxs:
            if self.M >= self.max_centers or added >= max_add_per_epoch:
                break
            x = S_n_np[i]
            if not np.all(np.isfinite(x)):
                continue
            if self.add_point(t(x, self.device)):
                added += 1
        return added

    def prune_to_indices(self, keep_idx_np):
        keep_idx = torch.as_tensor(keep_idx_np, device=self.device, dtype=torch.long)
        self.centers = self.centers[keep_idx]
        if self.M == 0:
            self.K_inv = torch.zeros((0,0), device=self.device, dtype=DTYPE)
            return
        K = rbf_kernel_mat_torch(self.centers, self.centers, self.ell)
        K = 0.5*(K + K.T) + self.jitter*torch.eye(self.M, device=self.device, dtype=DTYPE)
        self.K_inv = torch.linalg.solve(K, torch.eye(self.M, device=self.device, dtype=DTYPE))

# =========================
# Actor / Critics (torch)
# =========================
class KernelGaussianActor:
    def __init__(self, centers_t, ell, sigma, action_low, action_high, normalizer, device):
        self.device = device
        self.centers = centers_t.to(device, DTYPE)
        self.M = int(self.centers.shape[0])
        self.d = int(self.centers.shape[1]) if self.M > 0 else 0
        self.base_ell = t(ell, device)
        self.ell_eff  = t(ell, device)
        self.sigma = float(sigma)
        self.alpha = torch.zeros((self.M,), device=device, dtype=DTYPE)
        self.low = float(action_low); self.high = float(action_high)
        self.normalizer = normalizer

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
        if self.M == 0:
            return 0.0
        k = self.kvec(s_np)
        mu_t = k @ self.alpha
        mu_t = torch.nan_to_num(mu_t, nan=0.0, posinf=0.0, neginf=0.0)
        mu = float(mu_t.item())
        if not np.isfinite(mu):
            mu = 0.0
        return mu

    def act(self, s_np, stochastic=True):
        mu = self.mean(s_np)
        if not np.isfinite(mu):
            mu = 0.0
        a_raw = mu + (self.sigma * np.random.randn() if stochastic else 0.0)
        if not np.isfinite(a_raw):
            a_raw = mu
        a = np.clip(a_raw, self.low, self.high)
        return a, mu, a_raw

    def update_vanilla(self, g_t, lr=1e-2, clip_norm=None):
        if g_t is None or g_t.numel() == 0:
            return 0.0, 0.0
        g = g_t
        if clip_norm is not None and clip_norm > 0:
            n = torch.linalg.norm(g).item()
            if n > clip_norm:
                g = g * (clip_norm / max(n, 1e-12))
        step = float(lr)
        self.alpha = self.alpha + step * g
        self.alpha = torch.nan_to_num(self.alpha, nan=0.0, posinf=0.0, neginf=0.0)
        return step, float(torch.linalg.norm(step * g).item())

    def update_natural(self, w_t, target_kl, Fisher_mat=None, max_scale=5.0):
        if w_t is None or w_t.numel() == 0:
            return 0.0, 0.0
        step = w_t * 0.0
        return 0.0, float(torch.linalg.norm(step).item())

    def on_added_centers(self, n_added):
        if n_added <= 0: return
        self.alpha = torch.nn.functional.pad(self.alpha, (0, n_added), mode='constant', value=0.0)
        self.M = int(self.centers.shape[0]); self.d = int(self.centers.shape[1])

    def on_pruned_centers(self, keep_idx_np):
        if self.M == 0:
            self.alpha = torch.zeros((0,), device=self.device, dtype=DTYPE); return
        keep_idx = torch.as_tensor(keep_idx_np, device=self.device, dtype=torch.long)
        self.alpha = self.alpha[keep_idx]
        self.M = int(self.centers.shape[0]); self.d = int(self.centers.shape[1])

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
        self.M = int(new_M)
        self.coef = torch.zeros((self.M,), device=self.device, dtype=DTYPE)

    def _norm_batch_t(self, S_np_or_t):
        if isinstance(S_np_or_t, np.ndarray):
            X = t(S_np_or_t, self.device)
        else:
            X = S_np_or_t.to(self.device, DTYPE)
        return self.normalizer.transform_torch(X) if self.normalizer is not None else X

    def fit_td_lambda_step(self, S_np, R_np, S2_np, DONE_np, gamma=0.99, lam=0.9, lr=1e-2, steps=1):
        if self.M == 0:
            self.coef = torch.zeros((0,), device=self.device, dtype=DTYPE)
            return
        S_n  = self._norm_batch_t(S_np)
        S2_n = self._norm_batch_t(S2_np)
        Phi  = rbf_kernel_mat_torch(S_n,  self.centers, self.ell)
        Phi2 = rbf_kernel_mat_torch(S2_n, self.centers, self.ell)
        R_t  = t(R_np, self.device).reshape(-1)
        D_t  = t(DONE_np.astype(np.float64), self.device).reshape(-1)
        w = self.coef.clone()
        N = Phi.shape[0]
        for _ in range(max(1, int(steps))):
            z = torch.zeros((self.M,), device=self.device, dtype=DTYPE)
            idx = torch.randperm(N, device=self.device)
            for t_i in idx:
                phi_t  = Phi[t_i]
                phi_tp = Phi2[t_i]
                done_t = D_t[t_i].item() > 0.5
                v_tp = 0.0 if done_t else float((phi_tp @ w).item())
                delta = R_t[t_i].item() + gamma * v_tp - float((phi_t @ w).item())
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
# Compatible Advantage Critic
# =========================
class CompatibleAdvCritic:
    def __init__(self, actor: KernelGaussianActor, lam, ema, device):
        self.device = device
        self.actor = actor
        self.M = actor.M
        self.lam = float(lam)
        self.ema = float(ema)
        self.F = torch.zeros((self.M, self.M), device=device, dtype=DTYPE)
        self.g = torch.zeros((self.M,), device=device, dtype=DTYPE)
        self.w = torch.zeros((self.M,), device=device, dtype=DTYPE)

    def update(self, S_np, A_RAW_np, MU_np, Ahat_np):
        sigma2 = max(self.actor.sigma ** 2, 1e-12)
        K_s = self.actor.kmat(S_np)
        if K_s.shape[1] == 0:
            self.M = 0
            self.F = torch.zeros((0,0), device=self.device, dtype=DTYPE)
            self.g = torch.zeros((0,),  device=self.device, dtype=DTYPE)
            self.w = torch.zeros((0,),  device=self.device, dtype=DTYPE)
            return self.F.clone(), self.g.clone()

        A_RAW = t(A_RAW_np, self.device).reshape(-1)
        MU    = t(MU_np,    self.device).reshape(-1)
        Ahat  = t(Ahat_np,  self.device).reshape(-1)

        K_sa = ((A_RAW - MU) / sigma2).unsqueeze(-1) * K_s
        N = K_s.shape[0]
        F_batch = (K_s.T @ K_s) / (sigma2 * max(N,1))
        g_batch = (K_sa.T @ Ahat) / max(N,1)

        if self.ema > 0.0 and self.F.numel() != 0:
            self.F = self.ema * self.F + (1.0 - self.ema) * F_batch
            self.g = self.ema * self.g + (1.0 - self.ema) * g_batch
            g_used = self.g
        else:
            self.F, self.g = F_batch, g_batch
            g_used = g_batch

        self.F = torch.nan_to_num(self.F, nan=0.0, posinf=0.0, neginf=0.0)
        self.g = torch.nan_to_num(self.g, nan=0.0, posinf=0.0, neginf=0.0)
        self.w = torch.zeros_like(self.g)
        return F_batch.clone(), g_used.clone()

    def update_iter(self, S_np, A_RAW_np, MU_np, Ahat_np, lr=1e-1, steps=1):
        return self.update(S_np, A_RAW_np, MU_np, Ahat_np)

    def on_centers_changed(self, new_M):
        self.M = int(new_M)
        self.F = torch.zeros((self.M, self.M), device=self.device, dtype=DTYPE)
        self.g = torch.zeros((self.M,), device=self.device, dtype=DTYPE)
        self.w = torch.zeros((self.M,), device=self.device, dtype=DTYPE)

# =========================
# RKHS–SHAP：CME（exact, accelerated）
# =========================
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
        dev = self.device
        wdt = work_dtype

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
            xi_b = bg_n[:, i]
            xi_t = tg_n[:, i]
            xi_c = C[:,   i]
            Lbg_bg.append( -0.5 * (xi_b[:,None] - xi_b[None,:])**2 * sc )
            Lbg_ctr.append(-0.5 * (xi_b[:,None] - xi_c[None,:])**2 * sc )
            Ltg_ctr.append(-0.5 * (xi_t[:,None] - xi_c[None,:])**2 * sc )
            Lbg_tgt.append(-0.5 * (xi_b[:,None] - xi_t[None,:])**2 * sc )

        log_B_total = torch.stack(Lbg_ctr, dim=0).sum(0)

        Kbg = torch.exp(log_B_total)
        v_bg_mean = float((Kbg @ eta).mean())
        if not np.isfinite(v_bg_mean):
            v_bg_mean = 0.0

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

                KX        = KX * torch.exp(Lbg_bg[i])
                logKXT    = logKXT   + Lbg_tgt[i]
                logA      = logA     + Ltg_ctr[i]
                logB_prefix = logB_prefix + Lbg_ctr[i]

                n = B
                K_reg = KX + (n * self.reg) * torch.eye(B, device=dev, dtype=wdt)
                KXT   = torch.exp(logKXT)
                try:
                    L = torch.linalg.cholesky(K_reg)
                    W = torch.cholesky_solve(KXT, L)
                except RuntimeError:
                    W = torch.linalg.lstsq(K_reg, KXT).solution

                A_C = torch.exp(logA)
                B_R = torch.exp(log_B_total - logB_prefix)
                S   = B_R.transpose(0,1) @ W
                v_now = (A_C * S.transpose(0,1)) @ eta
                inc = (v_now - v_prev).mean()
                if torch.isfinite(inc):
                    phi[i] += inc
                v_prev = v_now

        phi = (phi / max(1, n_perms))
        if not torch.isfinite(phi).all():
            phi = torch.ones_like(phi)
        return phi.detach().cpu().numpy()

# =========================
# RKHS–SHAP：KME（梯度版）
# =========================
class ShapKME:
    """
    RKHS-SHAP (KME-style): 用 value function 对输入状态的梯度，
    在特征空间内衡量每个维度的重要性。
    """
    def __init__(self, value_critic: KernelValue, normalizer: StateNormalizer,
                 base_ell, ell_scale=1.0, device="cpu"):
        self.vc = value_critic
        self.norm = normalizer
        self.base_ell = t(base_ell, device)
        self.ell_scale = float(ell_scale)
        self.device = device

    @torch.no_grad()
    def _sanitize_ell(self, ell_t):
        ell_t = torch.nan_to_num(ell_t, nan=1.0, posinf=1e3, neginf=1.0)
        ell_t = torch.clamp(ell_t, min=1e-6, max=1e6)
        return ell_t

    def phi_for_batch_grad(self, S_targets, work_dtype=torch.float32):
        """
        输入: S_targets [T,d]
        输出: φ_kme [d]，每个维度的重要性
        """
        dev = self.device
        wdt = work_dtype

        if S_targets is None or len(S_targets) == 0:
            d = self.base_ell.numel()
            return np.ones((d,), dtype=np.float64)

        S = torch.as_tensor(S_targets, device=dev, dtype=wdt, requires_grad=True)

        S_n = self.norm.transform_torch(
            S.to(self.vc.centers.device, self.vc.centers.dtype)
        ).to(dev, wdt)

        ell = (self.base_ell.to(dev) * self.ell_scale).to(wdt)
        ell = self._sanitize_ell(ell)

        C = self.vc.centers.to(dev, wdt)
        eta = self.vc.coef.to(dev, wdt).reshape(-1)

        K = rbf_kernel_mat_torch(S_n, C, ell)  # [T,M]
        v = (K @ eta).mean()                   # 标量

        grads = torch.autograd.grad(v, S, retain_graph=False, create_graph=False)[0]  # [T,d]

        phi = grads.abs().mean(dim=0)
        phi = torch.nan_to_num(phi, nan=1.0, posinf=1.0, neginf=1.0)
        return phi.detach().cpu().numpy()

# =========================
# Rollout + Eval
# =========================
def collect_rollouts(env, actor, episodes, horizon, gamma, noise):
    traj = []; states_for_dict = []
    obs_dim = env.observation_space.shape[0]
    for _ in range(episodes):
        s, info = _reset_env(env)
        ep = {"s": [], "s2": [], "a": [], "a_raw": [], "mu": [], "r": [], "done": []}
        for tstep in range(horizon):
            a, mu, a_raw = actor.act(s, stochastic=True)
            if not np.isfinite(a):
                a = 0.0
            s2, r, done, info2 = _step_env(env, a, allow_restart=True)
            if isinstance(s2, tuple): s2, _ = s2
            if noise > 0:
                s2 = s2 + np.random.randn(obs_dim) * noise
            if (not np.isfinite(s2).all()) or (not np.isfinite(r)):
                done = True
            ep["s"].append(s); ep["s2"].append(s2)
            ep["a"].append(float(a)); ep["a_raw"].append(float(a_raw)); ep["mu"].append(float(mu))
            ep["r"].append(float(r)); ep["done"].append(bool(done))
            states_for_dict.append(s)
            s = s2
            if done: break
        traj.append(ep)

    S   = np.asarray([s for ep in traj for s in ep["s"]], dtype=np.float64)
    S2  = np.asarray([s2 for ep in traj for s2 in ep["s2"]], dtype=np.float64)
    D   = np.asarray([d for ep in traj for d in ep["done"]], dtype=bool)
    A   = np.asarray([a for ep in traj for a in ep["a"]], dtype=np.float64)
    AR  = np.asarray([ar for ep in traj for ar in ep["a_raw"]], dtype=np.float64)
    MU  = np.asarray([m for ep in traj for m in ep["mu"]], dtype=np.float64)
    R   = np.asarray([r for ep in traj for r in ep["r"]], dtype=np.float64)
    lens = np.asarray([len(ep["r"]) for ep in traj], dtype=np.int32)

    for arr in (S, S2, A, AR, MU, R):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    G = []
    idx = 0
    for L in lens:
        G.extend(discount_cumsum_np(R[idx:idx+L], gamma))
        idx += L
    G = np.asarray(G, dtype=np.float64)
    return S, S2, D, A, AR, MU, R, G, lens, np.asarray(states_for_dict, dtype=np.float64)

def evaluate_policy(env, actor, noise, episodes=5, horizon=200):
    rets = []
    obs_dim = env.observation_space.shape[0]
    for _ in range(episodes):
        s, info = _reset_env(env)
        ret = 0.0
        for tstep in range(horizon):
            a, _, _ = actor.act(s, stochastic=False)
            if not np.isfinite(a):
                a = 0.0
            s, r, done, info = _step_env(env, a, allow_restart=False)
            if isinstance(s, tuple): s, _ = s
            if noise > 0:
                s = s + np.random.randn(obs_dim) * noise
            if not (np.isfinite(s).all() and np.isfinite(r)):
                break
            ret += float(r)
            if done: break
        if not np.isfinite(ret):
            ret = 0.0
        rets.append(ret)
    return float(np.mean(rets)) if len(rets) > 0 else 0.0

# =========================
# Gap wrt optimal value function (LIP-LQR)
# =========================
def compute_value_error_opt(env, actor, gamma=0.99, episodes=20, horizon=200):
    """
    对每个 episode 的初始状态 x0:
        J^*(x0) = x0^T P x0 （env.optimal_value）
        J^π(x0) = - sum_t γ^t r_t  （当前策略 rollout 的折扣 cost）
    返回平均 (J^π(x0) - J^*(x0))，即相对于最优策略的 cost gap。
    只在 LIP-LQR 有意义。
    """
    if not hasattr(env, "optimal_value"):
        return np.nan

    gaps = []
    obs_dim = env.observation_space.shape[0]

    for _ in range(episodes):
        s, info = _reset_env(env)
        x0 = np.asarray(s, dtype=np.float64).reshape(obs_dim)
        J_opt = env.optimal_value(x0)

        # rollout 当前策略，估计 J^π(x0)
        ret = 0.0
        discount = 1.0
        done = False
        for tstep in range(horizon):
            a, _, _ = actor.act(s, stochastic=False)
            if not np.isfinite(a):
                a = 0.0
            s, r, done, info = _step_env(env, a, allow_restart=False)
            if isinstance(s, tuple): s, _ = s
            if not (np.isfinite(s).all() and np.isfinite(r)):
                break
            # reward = -cost
            ret += discount * float(r)
            discount *= gamma
            if done:
                break

        # J^π = discounted cost = -ret
        J_pi = -ret
        gap = J_pi - J_opt
        if np.isfinite(gap):
            gaps.append(gap)

    if len(gaps) == 0:
        return np.nan
    return float(np.mean(gaps))

# =========================
# Plot value function error (2D slice)
# =========================
def plot_value_error_2d(env, v_critic, filename_prefix="value_error",
                        grid_points=81, dim_idx=(0,1), state_fix=None,
                        use_cost_view=True):
    if not hasattr(env, "optimal_value"):
        print("[Value-Plot] env has no optimal_value; skip.")
        return

    obs_dim = env.observation_space.shape[0]
    d0, d1 = dim_idx
    if d0 < 0 or d1 < 0 or d0 >= obs_dim or d1 >= obs_dim or d0 == d1:
        print(f"[Value-Plot] invalid dim_idx={dim_idx}, skip.")
        return

    if state_fix is None:
        state_fix = np.zeros(obs_dim, dtype=np.float64)
    else:
        state_fix = np.asarray(state_fix, dtype=np.float64)
        if state_fix.shape[0] != obs_dim:
            print("[Value-Plot] state_fix dim mismatch, skip.")
            return

    low  = np.array(env.observation_space.low,  dtype=np.float64)
    high = np.array(env.observation_space.high, dtype=np.float64)
    low  = np.where(np.isfinite(low),  low,  -2.0)
    high = np.where(np.isfinite(high), high,  2.0)

    x1 = np.linspace(low[d0], high[d0], grid_points)
    x2 = np.linspace(low[d1], high[d1], grid_points)
    X1, X2 = np.meshgrid(x1, x2)

    states = []
    for i in range(grid_points):
        for j in range(grid_points):
            s = state_fix.copy()
            s[d0] = X1[i, j]
            s[d1] = X2[i, j]
            states.append(s)
    states = np.asarray(states, dtype=np.float64)

    V_est = v_critic.predict(states)
    V_opt_raw = np.array([env.optimal_value(s) for s in states], dtype=np.float64)

    if use_cost_view:
        V_est_cost = -V_est
        V_opt_cost = V_opt_raw
    else:
        V_est_cost = V_est
        V_opt_cost = V_opt_raw

    err = V_est_cost - V_opt_cost
    abs_err = np.abs(err)

    Err_grid = err.reshape(grid_points, grid_points)
    AbsErr_grid = abs_err.reshape(grid_points, grid_points)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    im1 = plt.imshow(Err_grid, origin="lower",
                     extent=[x1[0], x1[-1], x2[0], x2[-1]],
                     aspect="auto")
    plt.colorbar(im1)
    plt.xlabel(f"state[{d0}]")
    plt.ylabel(f"state[{d1}]")
    plt.title("V_est_cost - V_opt_cost")

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(AbsErr_grid, origin="lower",
                     extent=[x1[0], x1[-1], x2[0], x2[-1]],
                     aspect="auto")
    plt.colorbar(im2)
    plt.xlabel(f"state[{d0}]")
    plt.ylabel(f"state[{d1}]")
    plt.title("|V_est_cost - V_opt_cost|")

    plt.tight_layout()
    out_path = f"{filename_prefix}_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Value-Plot] saved heatmap to {out_path}")

    plt.figure(figsize=(5, 4))
    plt.hist(abs_err, bins=50, alpha=0.8)
    plt.xlabel("|V_est_cost - V_opt_cost|")
    plt.ylabel("count")
    plt.title("Distribution of abs error on grid (2D slice)")
    out_hist = f"{filename_prefix}_hist.png"
    plt.tight_layout()
    plt.savefig(out_hist, dpi=150)
    plt.close()
    print(f"[Value-Plot] saved hist to {out_hist}")

# =========================
# New: 4D grid value error (for final figure & expected gap)
# =========================
def evaluate_4d_grid_error(env, v_critic, filename_prefix="value_error4d",
                           grid_points_per_dim=9, use_cost_view=True):
    if not hasattr(env, "optimal_value"):
        print("[Value-4D] env has no optimal_value; skip.")
        return np.nan

    obs_space = env.observation_space
    obs_dim = obs_space.shape[0]
    if obs_dim != 4:
        print(f"[Value-4D] obs_dim={obs_dim} != 4, skip 4D grid.")
        return np.nan

    low  = np.array(obs_space.low,  dtype=np.float64)
    high = np.array(obs_space.high, dtype=np.float64)
    low  = np.where(np.isfinite(low),  low,  -2.0)
    high = np.where(np.isfinite(high), high,  2.0)

    axes = [np.linspace(low[i], high[i], grid_points_per_dim) for i in range(obs_dim)]
    X0, X1, X2, X3 = np.meshgrid(axes[0], axes[1], axes[2], axes[3], indexing="ij")
    states = np.stack([X0, X1, X2, X3], axis=-1).reshape(-1, obs_dim)

    V_est = v_critic.predict(states)
    V_opt_raw = np.array([env.optimal_value(s) for s in states], dtype=np.float64)

    if use_cost_view:
        V_est_cost = -V_est
        V_opt_cost = V_opt_raw
    else:
        V_est_cost = V_est
        V_opt_cost = V_opt_raw

    abs_err = np.abs(V_est_cost - V_opt_cost)
    mean_abs_err = float(abs_err.mean())

    plt.figure(figsize=(6, 4))
    plt.hist(abs_err, bins=80, alpha=0.85)
    plt.xlabel("|V_est_cost - V_opt_cost|")
    plt.ylabel("count")
    plt.title("Distribution of abs error on 4D grid")
    out_hist = f"{filename_prefix}_4d_hist.png"
    plt.tight_layout()
    plt.savefig(out_hist, dpi=150)
    plt.close()
    print(f"[Value-4D] saved 4D abs-error hist to {out_hist}")

    out_npz = f"{filename_prefix}_4d_grid_error.npz"
    np.savez(out_npz,
             states=states.astype(np.float32),
             abs_err=abs_err.astype(np.float32),
             V_est_cost=V_est_cost.astype(np.float32),
             V_opt_cost=V_opt_cost.astype(np.float32))
    print(f"[Value-4D] saved grid error data to {out_npz}")
    print(f"[Value-4D] mean abs gap on 4D grid = {mean_abs_err:.6f}")

    return mean_abs_err

# =========================
# Train one seed
# =========================
def run_one_seed(args, seed, device):
    set_seed(seed)

    # 训练环境 / 评估环境
    if args.env == "LIP-LQR":
        env_tr = LinearizedInvertedPendulumLQREnv(
            dt=0.02,
            gamma=args.gamma,
            max_episode_steps=args.horizon,
            process_noise_std=args.noise
        )
        enable_restart_kernel(env_tr, p_restart=args.restart_p, seed=seed)
        env_ev = LinearizedInvertedPendulumLQREnv(
            dt=0.02,
            gamma=args.gamma,
            max_episode_steps=args.horizon,
            process_noise_std=0.0
        )
        enable_restart_kernel(env_ev, p_restart=0.0, seed=seed+777)
    else:
        env_tr = gym.make(args.env)
        enable_restart_kernel(env_tr, p_restart=args.restart_p, seed=seed)
        env_ev = gym.make(args.env)
        enable_restart_kernel(env_ev, p_restart=0.0, seed=seed+777)

    try:
        env_tr.reset(seed=seed)
        if hasattr(env_tr.action_space, "seed"): env_tr.action_space.seed(seed)
        if hasattr(env_tr.observation_space, "seed"): env_tr.observation_space.seed(seed)
    except Exception:
        pass
    try:
        env_ev.reset(seed=seed+123)
        if hasattr(env_ev.action_space, "seed"): env_ev.action_space.seed(seed+123)
        if hasattr(env_ev.observation_space, "seed"): env_ev.observation_space.seed(seed+123)
    except Exception:
        pass

    obs_dim = env_tr.observation_space.shape[0]
    if isinstance(env_tr.action_space.high, np.ndarray):
        assert int(np.prod(env_tr.action_space.shape)) == 1, "This code assumes 1-D (scalar) action."
        a_high = float(env_tr.action_space.high[0]); a_low = float(env_tr.action_space.low[0])
    else:
        a_high = float(env_tr.action_space.high); a_low = float(env_tr.action_space.low)

    # 预热 + normalizer
    WARM_EP = max(4, args.episodes_per_update//3)
    tmp_actor = KernelGaussianActor(centers_t=t(np.zeros((1, obs_dim)), device),
                                    ell=1.0, sigma=args.sigma,
                                    action_low=a_low, action_high=a_high,
                                    normalizer=None, device=device)
    _, _, _, _, _, _, _, _, _, states_pool = collect_rollouts(env_tr, tmp_actor, WARM_EP, args.horizon, args.gamma, args.noise)
    states_pool = np.nan_to_num(states_pool, nan=0.0, posinf=0.0, neginf=0.0)
    state_mean = states_pool.mean(axis=0)
    state_std  = states_pool.std(axis=0) + 1e-8
    normalizer = StateNormalizer(state_mean, state_std, device)

    base_ell = np.ones_like(state_std) * float(args.actor_ell)

    # kmeans++ 初始化中心
    states_pool_n = normalizer.transform(states_pool)
    init_k_a = min(args.centers_init_a, args.max_centers_a)
    init_k_v = min(args.centers_init_v, args.max_centers_v)
    centers_init_a = kmeanspp_init(states_pool_n, init_k_a, seed=seed)
    centers_init_v = kmeanspp_init(states_pool_n, init_k_v, seed=seed+99)

    # ALD 字典
    ald_A = ALDDictionary(ell=base_ell, eps=args.ald_eps_a, max_centers=args.max_centers_a,
                          jitter=1e-8, device=device)
    ald_V = ALDDictionary(ell=base_ell, eps=args.ald_eps_v, max_centers=args.max_centers_v,
                          jitter=1e-8, device=device)
    ald_A.initialize(centers_init_a)
    ald_V.initialize(centers_init_v)

    # Actor / Critics
    actor = KernelGaussianActor(ald_A.centers, ell=base_ell, sigma=args.sigma,
                                action_low=a_low, action_high=a_high,
                                normalizer=normalizer, device=device)
    v_critic = KernelValue(ald_V.centers, ell=base_ell, reg=args.v_reg,
                           normalizer=normalizer, device=device)
    a_critic = CompatibleAdvCritic(actor, lam=args.lam_adv, ema=args.ema_adv, device=device)

    # SHAP: CME + KME
    shap_cme = ShapCME(v_critic, normalizer, base_ell,
                       reg=args.cme_reg, ell_scale=args.cme_ell_scale,
                       device=device)
    shap_kme = ShapKME(v_critic, normalizer, base_ell,
                       ell_scale=args.cme_ell_scale,
                       device=device)

    shap_history = []
    phi_ema_vec = None

    if seed == args.seed:
        print(f"[Device] {device} | dtype={DTYPE}")
        shap_mode_str = f"SHAP={args.shap_mode.upper()}" if args.shap_enable else "SHAP=off"
        print(f"[Info] env={args.env} | horizon={args.horizon} | ep/update={args.episodes_per_update} "
              f"| init_A={ald_A.M}/{args.max_centers_a} | init_V={ald_V.M}/{args.max_centers_v} "
              f"| base_ell={args.actor_ell} | Value=TD(λ) | PG=vanilla (lr_actor) | {shap_mode_str} "
              f"(warmup={args.shap_warmup}, kappa={args.shap_kappa}, tr_kl={args.shap_tr_kl})")

    eval_curve = np.full((args.epochs+1,), np.nan, dtype=np.float64)
    # 新：与最优 value function 的 cost gap 曲线
    value_err_curve = np.full((args.epochs+1,), np.nan, dtype=np.float64)

    for ep in range(args.epochs+1):
        # restart 退火
        if args.restart_anneal_end > 0:
            frac = max(0.0, 1.0 - ep / max(1, args.restart_anneal_end))
            env_tr._p_restart = args.restart_p * frac
        else:
            env_tr._p_restart = args.restart_p

        # ALD epsilon 退火
        if args.ald_eps_anneal_end_a > 0:
            frac_eps_a = max(0.0, 1.0 - ep / max(1, args.ald_eps_anneal_end_a))
            ald_A.eps2 = max(args.ald_eps_final_a, args.ald_eps_a * frac_eps_a)
        if args.ald_eps_anneal_end_v > 0:
            frac_eps_v = max(0.0, 1.0 - ep / max(1, args.ald_eps_anneal_end_v))
            ald_V.eps2 = max(args.ald_eps_final_v, args.ald_eps_v * frac_eps_v)

        # 收集 on-policy 样本
        S, S2, DONE, A, A_RAW, MU, R, G, lens, states_batch = collect_rollouts(
            env_tr, actor, args.episodes_per_update, args.horizon, args.gamma, args.noise)

        # ALD 字典更新
        S_n = normalizer.transform(states_batch)
        added_A = ald_A.add_from_batch(S_n, max_add_per_epoch=args.ald_max_add_per_epoch_a)
        if added_A > 0:
            actor.centers = ald_A.centers; actor.on_added_centers(added_A)
            a_critic.actor = actor; a_critic.on_centers_changed(ald_A.M)

        added_V = ald_V.add_from_batch(S_n, max_add_per_epoch=args.ald_max_add_per_epoch_v)
        if added_V > 0:
            v_critic.centers = ald_V.centers; v_critic.on_centers_changed(ald_V.M)

        # prune actor centers
        if ald_A.M > args.max_centers_a:
            keep_A = np.argsort(actor.alpha.detach().cpu().abs().numpy())[::-1][:args.max_centers_a]
            keep_A.sort()
            ald_A.prune_to_indices(keep_A)
            actor.centers = ald_A.centers; actor.on_pruned_centers(keep_A)
            a_critic.actor = actor; a_critic.on_centers_changed(ald_A.M)

        # prune critic centers
        if ald_V.M > args.max_centers_v:
            if v_critic.coef.numel() == ald_V.M:
                keep_V = np.argsort(v_critic.coef.detach().cpu().abs().numpy())[::-1][:args.max_centers_v]
            else:
                keep_V = np.arange(args.max_centers_v)
            keep_V.sort()
            ald_V.prune_to_indices(keep_V)
            v_critic.centers = ald_V.centers; v_critic.on_centers_changed(ald_V.M)

        # TD(λ) value 更新
        v_critic.fit_td_lambda_step(S, R, S2, DONE, gamma=args.gamma,
                                    lam=args.td_lambda, lr=args.lr_v, steps=args.ts_critic_steps)
        V = v_critic.predict(S)
        Ahat = G - V
        Ahat = (Ahat - Ahat.mean()) / (Ahat.std() + 1e-8)

        # SHAP
        shap_applied = False

        if args.shap_enable and (ep % max(1, args.shap_interval) == 0) and (ep >= args.shap_warmup):
            B = min(len(states_batch), args.shap_bg_size)
            Tt = min(len(states_batch), args.shap_targets)
            idx_bg = np.random.choice(len(states_batch), size=B, replace=(B > len(states_batch)))
            idx_tg = np.random.choice(len(states_batch), size=Tt, replace=(Tt > len(states_batch)))
            bg = states_batch[idx_bg]
            tg = states_batch[idx_tg]

            work_dtype = torch.float32 if args.shap_dtype.lower() == "float32" else torch.float64

            phi_cme = None
            phi_kme = None

            if args.shap_mode in ["cme", "both"]:
                phi_cme = shap_cme.phi_for_batch_exact_fast(
                    tg, bg, n_perms=args.shap_perms,
                    sync_perm=bool(args.shap_sync_perm),
                    work_dtype=work_dtype
                )

            if args.shap_mode in ["kme", "both"]:
                phi_kme = shap_kme.phi_for_batch_grad(
                    tg,
                    work_dtype=work_dtype
                )

            if args.shap_mode == "cme":
                phi_raw = phi_cme
            elif args.shap_mode == "kme":
                phi_raw = phi_kme
            else:  # both
                eps = 1e-8
                phi_c = phi_cme / (np.mean(phi_cme) + eps)
                phi_k = phi_kme / (np.mean(phi_kme) + eps)
                phi_raw = 0.5 * (phi_c + phi_k)

            phi_raw = np.asarray(phi_raw, dtype=np.float64)
            phi_raw = np.where(np.isfinite(phi_raw), phi_raw, 1.0)

            if args.shap_ema > 0.0:
                if phi_ema_vec is None:
                    phi_ema_vec = phi_raw.copy()
                else:
                    phi_ema_vec = float(args.shap_ema) * phi_ema_vec + (1.0 - float(args.shap_ema)) * phi_raw
                phi_to_apply = phi_ema_vec.copy()
            else:
                phi_to_apply = phi_raw.copy()

            if args.shap_topk > 0 and args.shap_topk < len(phi_to_apply):
                idx = np.argsort(np.abs(phi_to_apply - 1.0))[::-1]
                keep = idx[:args.shap_topk]
                mask = np.zeros_like(phi_to_apply, dtype=bool); mask[keep] = True
                phi_to_apply = np.where(mask, phi_to_apply, 1.0)

            # trust-region 控制 ell_eff 的变化
            K_old = actor.kmat(tg)
            mu_old = (K_old @ actor.alpha).detach()
            old_ell = actor.ell_eff.clone()

            actor.set_shap_weights(phi_to_apply, eps=args.shap_floor, normalize=True,
                                   kappa=args.shap_kappa,
                                   ratio_min=args.shap_ratio_min, ratio_max=args.shap_ratio_max)

            K_new = actor.kmat(tg)
            mu_new = (K_new @ actor.alpha).detach()
            sigma2 = max(actor.sigma**2, 1e-12)
            kl_est = float(((mu_new - mu_old)**2).mean() / (2.0 * sigma2))
            if not np.isfinite(kl_est):
                kl_est = 0.0

            if args.shap_tr_kl > 0.0 and kl_est > args.shap_tr_kl:
                scale = math.sqrt(max(1e-12, args.shap_tr_kl) / kl_est)
                actor.ell_eff = old_ell
                actor.set_shap_weights(phi_to_apply, eps=args.shap_floor, normalize=True,
                                       kappa=float(args.shap_kappa) * scale,
                                       ratio_min=args.shap_ratio_min, ratio_max=args.shap_ratio_max)
                if seed == args.seed:
                    print(f"[SHAP-TR] structural KL {kl_est:.4g} > {args.shap_tr_kl:.4g}, "
                          f"shrink kappa by {scale:.3f}")
            shap_applied = True

            if args.save_shap:
                ell_snapshot = actor.ell_eff.detach().cpu().numpy().copy() if args.save_ell_eff else None
                shap_history.append((int(ep), phi_to_apply.astype(float).copy(), ell_snapshot))

        # advantage critic 更新
        if args.adv_solver == "closed":
            F_batch, g = a_critic.update(S, A_RAW, MU, Ahat)
        else:
            F_batch, g = a_critic.update_iter(S, A_RAW, MU, Ahat,
                                              lr=args.lr_adv, steps=args.ts_critic_steps)

        # actor 更新
        do_actor = (ep % max(1, args.ts_actor_delay) == 0)
        if do_actor:
            lr_used = args.lr_actor
            lr_step, step_norm = actor.update_vanilla(g, lr=lr_used, clip_norm=None)
            if args.max_alpha_norm > 0:
                an = float(torch.linalg.norm(actor.alpha).item())
                if an > args.max_alpha_norm:
                    actor.alpha *= (args.max_alpha_norm / (an + 1e-12))
        else:
            lr_step, step_norm = 0.0, 0.0

        # sigma 退火
        if ep < args.sigma_anneal_end:
            frac_s = 1.0 - ep / max(1, args.sigma_anneal_end)
            actor.sigma = args.sigma_anneal_to + (args.sigma - args.sigma_anneal_to) * frac_s
        else:
            actor.sigma = max(args.sigma_anneal_to, 1e-3)

        # 评估
        if (ep % args.eval_interval == 0) or (ep == 0):
            eval_ret = evaluate_policy(env_ev, actor, args.noise, episodes=args.eval_episodes, horizon=args.horizon)

            # 与最优 value function 的 cost gap
            mean_abs_err_opt = compute_value_error_opt(
                env_ev, actor,
                gamma=args.gamma,
                episodes=args.err_mc_episodes,
                horizon=args.horizon
            )
            value_err_curve[ep] = mean_abs_err_opt

            marker = "A*" if do_actor else "A-"
            shap_note = ""
            if args.shap_enable and (ep % max(1, args.shap_interval) == 0) and (ep >= args.shap_warmup):
                shap_note = " W↑(SHAP)" if shap_applied else " W-(skip)"
            print(f"{ep:04d} {marker}{shap_note} | eval={eval_ret:.2f} | M_A={ald_A.M} M_V={ald_V.M} "
                  f"| lr={lr_step:.3g} step={step_norm:.3g} "
                  f"| mean_abs_err_opt={mean_abs_err_opt:.3g} | device={device}")
            eval_curve[ep] = eval_ret

    # 前向填充
    last = np.nan
    for i in range(eval_curve.shape[0]):
        if np.isnan(eval_curve[i]): eval_curve[i] = last
        else: last = eval_curve[i]

    last_err = np.nan
    for i in range(value_err_curve.shape[0]):
        if np.isnan(value_err_curve[i]): value_err_curve[i] = last_err
        else: last_err = value_err_curve[i]

    # 保存 value_err_curve & 曲线
    epochs_arr = np.arange(len(value_err_curve))
    out_err_csv = os.path.join(args.out_dir, f"value_error_curve_seed{seed}.csv")
    with open(out_err_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "mean_abs_err_opt"])
        for e_i, v_i in zip(epochs_arr, value_err_curve):
            w.writerow([int(e_i), float(v_i) if np.isfinite(v_i) else np.nan])
    print(f"[Value-Curve] saved OPT value gap curve to {out_err_csv}")

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_arr, value_err_curve)
    plt.xlabel("Epoch")
    plt.ylabel("Mean (J^π - J^*) on init states")
    plt.title(f"{args.env} opt-gap vs epoch (seed={seed})")
    plt.tight_layout()
    out_err_png = os.path.join(args.out_dir, f"value_error_curve_seed{seed}.png")
    plt.savefig(out_err_png, dpi=150)
    plt.close()
    print(f"[Value-Curve] saved plot to {out_err_png}")

    # 保存 SHAP 记录
    if args.save_shap:
        os.makedirs(args.shap_out_dir, exist_ok=True)
        tag = f"_{args.shap_out_tag}" if args.shap_out_tag else ""
        out_csv_shap = os.path.join(args.shap_out_dir, f"shap_cme_seed{seed}{tag}.csv")
        d = obs_dim = env_tr.observation_space.shape[0]
        header = ["epoch"] + [f"phi_{i}" for i in range(d)]
        if args.save_ell_eff:
            header += [f"ell_{i}" for i in range(d)]
        with open(out_csv_shap, "w", newline="") as f:
            wcsv = csv.writer(f); wcsv.writerow(header)
            for ep_i, phi_i, ell_i in shap_history:
                row = [ep_i] + [float(x) for x in phi_i]
                if args.save_ell_eff:
                    if ell_i is None:
                        row += [np.nan for _ in range(d)]
                    else:
                        row += [float(x) for x in ell_i]
                wcsv.writerow(row)
        print(f"[SHAP] saved {len(shap_history)} records to {out_csv_shap}")

    # 画 2D slice 的 value 误差（可选）
    try:
        if hasattr(env_tr, "optimal_value"):
            if args.env == "LIP-LQR":
                dim_idx = (2, 3)
                state_fix = np.zeros(env_tr.observation_space.shape[0], dtype=np.float64)
            else:
                dim_idx = (0, 1)
                state_fix = None
            plot_value_error_2d(
                env_tr,
                v_critic,
                filename_prefix=os.path.join(args.out_dir, f"value_error_seed{seed}"),
                dim_idx=dim_idx,
                state_fix=state_fix,
                use_cost_view=True
            )
        else:
            print("[Value-Plot] env has no optimal_value; skip.")
    except Exception as e:
        print(f"[Value-Plot] plotting failed: {e}")

    # 最后 4D grid 下相对 J* 的 gap 分布
    if hasattr(env_ev, "optimal_value"):
        mean_abs_err_grid = evaluate_4d_grid_error(
            env_ev,
            v_critic,
            filename_prefix=os.path.join(args.out_dir, f"value_error_seed{seed}"),
            grid_points_per_dim=args.err_grid_points,
            use_cost_view=True
        )
        print(f"[Value-4D] FINAL expected abs gap (uniform 4D grid) = {mean_abs_err_grid:.6f}")

    env_tr.close(); env_ev.close()
    return eval_curve

# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Pendulum-v1",
                   help='可以设为 "Pendulum-v1" 或 "LIP-LQR"')
    p.add_argument("--episodes_per_update", type=int, default=24)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99)

    default_dev = "cpu"
    p.add_argument("--device", type=str, default=default_dev)

    p.add_argument("--actor_ell", type=float, default=0.8)

    p.add_argument("--centers_init_a", type=int, default=64)
    p.add_argument("--max_centers_a", type=int, default=384)
    p.add_argument("--ald_eps_a", type=float, default=1e-2)
    p.add_argument("--ald_eps_final_a", type=float, default=2e-4)
    p.add_argument("--ald_eps_anneal_end_a", type=int, default=300)
    p.add_argument("--ald_max_add_per_epoch_a", type=int, default=8)

    p.add_argument("--centers_init_v", type=int, default=64)
    p.add_argument("--max_centers_v", type=int, default=384)
    p.add_argument("--ald_eps_v", type=float, default=1e-2)
    p.add_argument("--ald_eps_final_v", type=float, default=2e-4)
    p.add_argument("--ald_eps_anneal_end_v", type=int, default=300)
    p.add_argument("--ald_max_add_per_epoch_v", type=int, default=8)

    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--sigma_anneal_to", type=float, default=0.25)
    p.add_argument("--sigma_anneal_end", type=int, default=300)

    p.add_argument("--restart_p", type=float, default=0.0)
    p.add_argument("--restart_anneal_end", type=int, default=0)

    p.add_argument("--target_kl", type=float, default=0.01)
    p.add_argument("--lam_adv", type=float, default=3e-3)
    p.add_argument("--ema_adv", type=float, default=0.97)
    p.add_argument("--v_reg", type=float, default=1e-3)

    p.add_argument("--ts_actor_delay", type=int, default=5)
    p.add_argument("--ts_actor_kl_scale", type=float, default=0.5)
    p.add_argument("--ts_critic_steps", type=int, default=2)
    p.add_argument("--lr_actor", type=float, default=1.0)
    p.add_argument("--adv_solver", type=str, choices=["closed","gd"], default="closed")
    p.add_argument("--lr_adv", type=float, default=1e-1)

    p.add_argument("--lr_v", type=float, default=1e-2)
    p.add_argument("--td_lambda", type=float, default=0.9)

    p.add_argument("--shap_enable", action="store_true")
    p.add_argument("--shap_interval", type=int, default=10)
    p.add_argument("--shap_perms", type=int, default=8)
    p.add_argument("--shap_targets", type=int, default=64)
    p.add_argument("--shap_bg_size", type=int, default=256)
    p.add_argument("--shap_floor", type=float, default=0.3)
    p.add_argument("--shap_ema", type=float, default=0.95)
    p.add_argument("--cme_reg", type=float, default=1e-3)
    p.add_argument("--cme_ell_scale", type=float, default=1.0)
    p.add_argument("--shap_dtype", type=str, choices=["float32","float64"], default="float32")
    p.add_argument("--shap_sync_perm", type=int, default=1)

    p.add_argument("--shap_kappa", type=float, default=0.25)
    p.add_argument("--shap_ratio_min", type=float, default=0.5)
    p.add_argument("--shap_ratio_max", type=float, default=2.0)
    p.add_argument("--shap_tr_kl", type=float, default=0.005)
    p.add_argument("--shap_warmup", type=int, default=5)
    p.add_argument("--shap_topk", type=int, default=-1)

    # CME / KME / both
    p.add_argument(
        "--shap_mode", type=str,
        choices=["cme", "kme", "both"],
        default="cme",
        help="Which RKHS-SHAP variant to use when --shap_enable is on."
    )

    p.add_argument("--save_shap", action="store_true")
    p.add_argument("--save_ell_eff", action="store_true")
    p.add_argument("--shap_out_dir", type=str, default="shap_logs_noise")
    p.add_argument("--shap_out_tag", type=str, default="exp1")
    p.add_argument("--noise", type=float, default=0.0)

    p.add_argument("--seed", type=int, default=245)
    p.add_argument("--multi_seeds", type=int, default=1)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--eval_interval", type=int, default=10)

    p.add_argument("--max_alpha_norm", type=float, default=1e4)

    # 误差估计相关参数（这里用作与最优 value function 的 gap 的 episode 数）
    p.add_argument("--err_mc_episodes", type=int, default=20,
                   help="episodes used to estimate J^π - J* gap per epoch")
    p.add_argument("--err_grid_points", type=int, default=9,
                   help="4D 网格每维的点数，用于最终 gap 直方图和期望")

    p.add_argument("--out_csv", type=str, default="eval_returns.csv")
    p.add_argument("--out_dir", type=str, default="return_logs_noise")
    p.add_argument("--out_png", type=str, default="eval_curve.png")

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

    header = ["epoch", "mean", "std"] + [f"seed_{i}" for i in range(len(seeds))]
    out_csv_path = os.path.join(args.out_dir, f"eval_return_seed{args.seed}_{args.shap_enable}_{args.noise}.csv")
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t_i in range(T):
            row = [t_i, float(mean_curve[t_i]), float(std_curve[t_i])] + [float(all_curves[i,t_i]) for i in range(len(seeds))]
            w.writerow(row)

    plt.figure(figsize=(7,4))
    plt.plot(epochs, mean_curve, label="mean eval_det")
    plt.xlabel("Epoch"); plt.ylabel("Eval return (deterministic)")
    title_suffix = f" + SHAP-{args.shap_mode.upper()}" if args.shap_enable else ""
    plt.title(f"{args.env} | mean over {len(seeds)} seeds (two-timescale + vanilla PG{title_suffix})")
    plt.legend(); plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"[Done] saved CSV to {out_csv_path}, figure to {args.out_png} | time={time.time()-t0:.1f}s | device={device}")

if __name__ == "__main__":
    main()
