import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# ========= 观测加高斯噪声的 Wrapper =========
class NoisyObservationWrapper(gym.ObservationWrapper):
    """
    对连续观测添加 N(0, noise_std^2) 高斯噪声（固定方差）。
    """

    def __init__(self, env, noise_std: float = 0.0):
        super().__init__(env)
        self.noise_std = float(noise_std)

    def observation(self, observation):
        if self.noise_std <= 0.0:
            return observation
        obs = np.array(observation, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=obs.shape)
        return obs + noise


def make_env(env_id: str, seed: int = 0, obs_noise_std: float = 0.0):
    """
    创建单个带 Monitor 的环境，用于 DummyVecEnv
    """

    def _init():
        env = gym.make(env_id)
        # 先加观测噪声，再加 Monitor（不影响 episode 统计）
        env = NoisyObservationWrapper(env, noise_std=obs_noise_std)
        env = Monitor(env)  # 让 info 里包含 'episode': {'r', 'l'}
        try:
            env.reset(seed=seed)
        except TypeError:
            # 老 gym 的兼容
            env.seed(seed)
        return env

    return _init


class EpisodeReturnCallback(BaseCallback):
    """
    训练过程中，每当一个 episode 结束，就记录：
      - 累计 episode 编号 ep_idx
      - 这一条 episode 的总回报 ep_return

    如果设置了 target_episodes，当累计 episode 数 >= target_episodes 时，
    返回 False 让 SB3 提前停止训练，这样就实现“固定总 episode 数”。
    """

    def __init__(self, target_episodes: int = None, verbose: int = 0):
        super().__init__(verbose)
        self.ep_idx = []
        self.ep_returns = []
        self._ep_counter = 0
        self.target_episodes = target_episodes

    def _on_step(self) -> bool:
        # SB3 在 vec env 下，locals 里一般有 'infos' 和 'dones'
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)

        if infos is None or dones is None:
            return True

        # DummyVecEnv 的维度通常是 1，但这里用通用写法
        for i, done in enumerate(dones):
            if done and isinstance(infos[i], dict) and "episode" in infos[i]:
                self._ep_counter += 1
                ep_info = infos[i]["episode"]
                ep_r = float(ep_info.get("r", 0.0))
                self.ep_idx.append(self._ep_counter)
                self.ep_returns.append(ep_r)

                if self.verbose > 0:
                    print(
                        f"[train] episode #{self._ep_counter}, "
                        f"return={ep_r:.1f}, "
                        f"length={ep_info.get('l', 'N/A')}"
                    )

                # 如果到达目标 episode 数，就要求停止训练
                if (
                    self.target_episodes is not None
                    and self._ep_counter >= self.target_episodes
                ):
                    if self.verbose > 0:
                        print(
                            f"[callback] Reached target_episodes = "
                            f"{self.target_episodes}, stopping training."
                        )
                    return False  # 返回 False -> SB3 提前停止

        return True


def train_algo(
    algo_name: str,
    env_id: str = "Ant-v5",
    total_episodes: int = 1000,
    seed: int = 0,
    device: str = "auto",
    obs_noise_std: float = 0.0,
):
    """
    用 Stable-Baselines3 训练一个算法（PPO 或 SAC），
    固定“总 episode 数”为 total_episodes。

    可以选择对状态加固定方差的高斯噪声（obs_noise_std）。

    返回：
      episodes: [N_ep]，真正意义上的“训练 episode 数”（1, 2, 3, ...）
      returns:  [N_ep]，对应 episode 的总回报
    """
    assert algo_name.lower() in ["ppo", "sac"], "algo_name 只能是 'ppo' 或 'sac'"

    # 单环境向量化
    train_env = DummyVecEnv(
        [make_env(env_id, seed=seed, obs_noise_std=obs_noise_std)]
    )

    algo_name_l = algo_name.lower()
    if algo_name_l == "ppo":
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=seed,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            learning_rate=3e-4,
            device=device,  # GPU / CPU
        )
    else:  # SAC
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=seed,
            buffer_size=1_000_000,
            batch_size=64,
            gamma=0.99,
            learning_rate=3e-4,
            train_freq=(1, "step"),
            gradient_steps=1,
            device=device,  # GPU / CPU
        )

    # 设置目标 episode 数的 callback
    callback = EpisodeReturnCallback(
        target_episodes=total_episodes,
        verbose=0,
    )

    # 注意：learn 仍然需要 total_timesteps，我们给一个比较大的上限，
    # 实际会在 callback 返回 False 时提前停止。
    max_timesteps = int(1e9)

    print(
        f"[{algo_name.upper()}] 开始训练，目标 episodes = {total_episodes}, "
        f"device = {device}, obs_noise_std = {obs_noise_std}"
    )
    model.learn(
        total_timesteps=max_timesteps,
        callback=callback,
        reset_num_timesteps=False,
    )

    train_env.close()

    episodes = np.array(callback.ep_idx, dtype=np.int64)
    returns = np.array(callback.ep_returns, dtype=np.float64)

    print(
        f"[{algo_name.upper()}] 训练结束，共记录到 {len(episodes)} 个 episode。"
    )

    return episodes, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Ant-v5")
    parser.add_argument(
        "--total_episodes",
        type=int,
        default=1000,
        help="总训练 episode 数（而不是总 step 数）",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--algos",
        type=str,
        nargs="+",
        default=["sac"],
        help="选择要跑的算法，例如: --algos ppo sac",
    )
    parser.add_argument(
        "--out_prefix", type=str, default="ant_v5_sb3_train_episode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "auto"],
        help="SB3 的 device 参数（cpu/cuda/auto）",
    )
    parser.add_argument(
        "--obs_noise_std",
        type=float,
        default=0.0,
        help="观测加的高斯噪声标准差（0 表示不加噪声）",
    )
    args = parser.parse_args()

    results = {}

    for algo_name in args.algos:
        algo_name_l = algo_name.lower()
        print(f"\n===== 训练 {algo_name_l.upper()} on {args.env} =====")
        episodes, rets = train_algo(
            algo_name=algo_name_l,
            env_id=args.env,
            total_episodes=args.total_episodes,
            seed=args.seed,
            device=args.device,
            obs_noise_std=args.obs_noise_std,
        )
        results[f"{algo_name_l}_episodes"] = episodes
        results[f"{algo_name_l}_returns"] = rets

    # 保存 npz
    npz_path = f"sac_{args.env}_{args.seed}_{args.obs_noise_std}_results.npz"
    np.savez(npz_path, **results)
    print(f"\n结果已保存到: {npz_path}")

    # 画图：横坐标是“训练 episode 数”
    plt.figure(figsize=(8, 5))
    for algo_name in args.algos:
        algo_name_l = algo_name.lower()
        episodes = results.get(f"{algo_name_l}_episodes", None)
        rets = results.get(f"{algo_name_l}_returns", None)
        if episodes is None or rets is None or len(episodes) == 0:
            continue
        plt.plot(episodes, rets, label=algo_name_l.upper())

    plt.xlabel("Training Episode")
    plt.ylabel("Episode Return (train env)")
    plt.title(
        f"{args.env}: PPO vs SAC (SB3, x = training episode, "
        f"obs_noise_std={args.obs_noise_std})"
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    img_path = f"sac_{args.env}_{args.seed}_{args.obs_noise_std}_learning_curve.png"
    plt.tight_layout()
    plt.savefig(img_path, dpi=200)
    print(f"学习曲线已保存为图片: {img_path}")


if __name__ == "__main__":
    main()
