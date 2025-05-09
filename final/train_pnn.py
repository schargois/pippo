import gymnasium as gym
from stable_baselines3 import PPO
from CustomACNetwork import CustomActorCriticPolicy, CustomNetwork
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from ProgNet import ProgColumn
from Callback import PPOCallback
from stable_baselines3.common.env_util import make_vec_env

from tqdm import trange

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import logging

# os.environ["MUJOCO_GL"] = "glfw"

repo_path = os.path.abspath("./final/Metaworld")
sys.path.insert(0, repo_path)
from metaworld import MT1
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.policies.sawyer_hammer_v2_policy import SawyerHammerV2Policy
from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
)

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

# training_iterations = 40960
# training_iterations = 2048
# training_iterations = 10240
# training_iterations = 20480
training_iterations = 102400
eval_episodes = 100
bc_epochs = 400
verbose = 0
hyperparams = {
    "n_steps": 1024,
    "learning_rate": 5e-4,
    "batch_size": 128,
    "ent_coef": 0.001,
}

datetime_str = time.strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    filename=f"run_{datetime_str}.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    filemode="w",
)

logger = logging.getLogger()


def next_model(model, env):
    policy_model = model.policy.mlp_extractor.policy_net
    value_model = model.policy.mlp_extractor.value_net
    policy_columns_lst = []
    value_columns_lst = []
    for i in range(policy_model.numCols):
        column = policy_model.getColumn(i)
        policy_columns_lst.append(column)
    for i in range(value_model.numCols):
        column = value_model.getColumn(i)
        value_columns_lst.append(column)

    model = PPO(
        CustomActorCriticPolicy,
        env,
        verbose=verbose,
        policy_kwargs={
            "policy_columns": policy_columns_lst,
            "value_columns": value_columns_lst,
            "new_column": True,
        },
        **hyperparams,
    )
    return model


def test_on_env(
    vec_environment, gym_env, model, num_episodes=eval_episodes, progress=True
):
    total_rew = 0
    iterate = trange(num_episodes) if progress else range(num_episodes)

    # Collect reward in vectorized env
    for _ in iterate:
        obs = vec_environment.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            next_obs, reward, done, _ = vec_environment.step(action)
            # vec_environment.render()      # Uncomment to render
            total_rew += reward
            obs = next_obs

    total_success = 0
    iterate = trange(num_episodes) if progress else range(num_episodes)

    # Collect successes from gym_env
    for _ in iterate:
        obs, _ = gym_env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs)
            next_obs, reward, done, truncated, info = gym_env.step(action)
            obs = next_obs

            if info.get("success", 0):
                total_success += 1
                break

    return (total_success / num_episodes), (total_rew / num_episodes).item()


class RandomGoalWrapper(gym.Wrapper):
    def __init__(self, env_class, task_list, render=False, seed=42):
        self.task_list = task_list
        render_mode = "human" if render else None
        self.env_class = env_class
        self.seed = seed
        env = env_class(render_mode=render_mode, seed=self.seed)
        super().__init__(env)

    def reset(self, **kwargs):
        task = random.choice(self.task_list)
        self.env.set_task(task)
        return self.env.reset(**kwargs)


def make_envs(env_class, train_tasks, test_tasks, render=False, normalizer_source=None):
    train_env = RandomGoalWrapper(env_class, train_tasks, render=render, seed=seed)
    test_env = RandomGoalWrapper(env_class, test_tasks, render=render)

    train_vec_env = DummyVecEnv([lambda: train_env])
    test_vec_env = DummyVecEnv([lambda: test_env])

    if normalizer_source is None:
        train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)
    else:
        # Clone normalization stats from the previous tier
        train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)
        train_vec_env.obs_rms = normalizer_source.obs_rms
        train_vec_env.ret_rms = normalizer_source.ret_rms

    test_vec_env = VecNormalize(test_vec_env, norm_obs=True, norm_reward=True)
    test_vec_env.obs_rms = train_vec_env.obs_rms
    test_vec_env.ret_rms = train_vec_env.ret_rms
    test_vec_env.training = False
    test_vec_env.norm_reward = False

    return train_vec_env, test_vec_env, train_env, test_env


def warm_start(
    model,
    vec_env,
    expert_policy,
    num_steps=10024,
    batch_size=128,
    bc_epochs=bc_epochs,
    task_name="task",
    test_vec_env=None,
):
    """
    Pre-train both the policy and value networks using behavior cloning from the expert policy.
    """

    print("Collecting expert data for behavior cloning...")
    logger.info("Collecting expert data for behavior cloning...")
    obs_list, act_list, rew_list = [], [], []

    env = vec_env.venv.envs[0]
    obs, _ = env.reset()

    for _ in trange(num_steps):
        action = expert_policy.get_action(obs)

        obs_list.append(obs)
        act_list.append(action)

        obs, reward, done, truncated, info = env.step(action)
        rew_list.append(reward)

        if info.get("success", 0):
            obs, _ = env.reset()
    vec_env.obs_rms.mean = np.mean(obs_list, axis=0)
    vec_env.obs_rms.var = np.var(obs_list, axis=0)
    vec_env.obs_rms.count = len(obs_list)

    obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
    act_tensor = torch.tensor(np.array(act_list), dtype=torch.float32)

    returns = []
    discounted_sum = 0.0
    gamma = model.gamma
    for r in reversed(rew_list):
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    returns_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
    mean, std = returns_tensor.mean(), returns_tensor.std()
    returns_tensor = (returns_tensor - mean) / (std + 1e-8)

    dataset = TensorDataset(obs_tensor, act_tensor, returns_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    print("Training actor and critic networks via behavior cloning...")
    logger.info("Training actor and critic networks via behavior cloning...")
    policy.train()

    actor_losses = []
    critic_losses = []
    reward_vals = []
    success_vals = []
    eval_epochs = []

    for epoch in range(bc_epochs):
        total_actor_loss = 0
        total_critic_loss = 0

        for batch_obs, batch_acts, batch_rets in loader:
            batch_obs = batch_obs.to(policy.device)
            batch_acts = batch_acts.to(policy.device)
            batch_rets = batch_rets.to(policy.device)

            actions_pred, value_pred, _ = policy.forward(batch_obs)

            actor_loss = F.mse_loss(actions_pred, batch_acts)
            critic_loss = F.mse_loss(value_pred, batch_rets)

            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        avg_actor_loss = total_actor_loss / len(loader)
        avg_critic_loss = total_critic_loss / len(loader)
        actor_losses.append(avg_actor_loss)
        critic_losses.append(avg_critic_loss)

        if epoch % 20 == 0 or epoch == bc_epochs - 1:
            print(
                f"Epoch {epoch+1}/{bc_epochs} | "
                f"Actor Loss: {avg_actor_loss:.5f} | "
                f"Critic Loss: {avg_critic_loss:.5f}"
            )
            success, reward = test_on_env(
                test_vec_env,
                test_vec_env.venv.envs[0],
                model,
                num_episodes=20,
                progress=False,
            )
            reward_vals.append(reward)
            success_vals.append(success)
            eval_epochs.append(epoch)
            logger.info(
                f"Epoch {epoch+1}/{bc_epochs} | "
                f"Actor Loss: {total_actor_loss/len(loader):.5f} | "
                f"Critic Loss: {total_critic_loss/len(loader):.5f}"
            )

    print("Warm start complete.")
    logger.info("Warm start complete.")
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/warm_start_rew_plot_{task_name}.png"
    plt.figure()
    plt.plot(eval_epochs, reward_vals, label="Avg Reward (20 episodes)", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title(f"Warm Start Reward - {task_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

    plot_path = f"plots/warm_start_suc_plot_{task_name}.png"
    plt.figure()
    plt.plot(eval_epochs, success_vals, label="Success Rate (20 episodes)", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Success Rate")
    plt.title(f"Warm Start Success Rate - {task_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")


def train_tier(
    save_path,
    model,
    vec_env,
    test_vec_env,
    bc_policy=None,
    train_ppo=True,
    use_pnn=True,
    bc_epochs=bc_epochs,
):
    """
    set model argument to Normal PPO if not using PNN
    """
    callback = PPOCallback(
        verbose=1, save_path=save_path, eval_env=test_vec_env, logger=logger
    )

    if use_pnn:
        if model is None:
            model = PPO(
                CustomActorCriticPolicy, vec_env, verbose=verbose, **hyperparams
            )
        else:
            model = next_model(model, vec_env)
    elif model is None:
        print("Model is None and not using PNN, will not train across tasks.")
        logger.info("Model is None and not using PNN, will not train across tasks.")
        model = PPO("MlpPolicy", vec_env, verbose=verbose, **hyperparams)

    print("Training model...")
    logger.info("Training model...")
    if bc_policy is not None:
        warm_start(
            model,
            vec_env,
            bc_policy,
            task_name=save_path,
            test_vec_env=test_vec_env,
            bc_epochs=bc_epochs,
        )
        print("Saving model after warm start...")
        logger.info("Saving model after warm start...")
        PPO.save(model, "warm-" + save_path)
        evaluate_model(
            "warm-" + save_path,
            test_vec_env,
            test_vec_env.venv.envs[0],
            f"Warm Start {save_path}",
        )
    if train_ppo:
        model.learn(training_iterations, callback=callback)
    vec_env.close()
    return model


def evaluate_model(model_path, env_vec, env_raw, label):
    model = PPO.load(model_path, env=env_vec)
    success, reward = test_on_env(env_vec, env_raw, model)
    print(f"{label} Total reward:", reward)
    print(f"{label} Success percentage:", success)
    logger.info(f"{label} Total reward: {reward}")
    logger.info(f"{label} Success percentage: {success}")
