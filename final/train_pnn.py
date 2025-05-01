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

# os.environ["MUJOCO_GL"] = "glfw"

repo_path = os.path.abspath("./final/Metaworld")
sys.path.insert(0, repo_path)
from metaworld import MT1
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.policies.sawyer_hammer_v2_policy import SawyerHammerV2Policy

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

training_iterations = 20480
verbose = 0


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
        },
    )
    return model


def test_on_env(vec_environment, gym_env, model, num_episodes=20, progress=True):
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
    def __init__(self, env_class, task_list, render=False):
        self.task_list = task_list
        render_mode = "human" if render else None
        super().__init__(env_class(render_mode=render_mode))

    def reset(self, **kwargs):
        task = random.choice(self.task_list)
        self.env.set_task(task)
        return self.env.reset(**kwargs)


def make_envs(env_class, train_tasks, test_tasks, render=False, normalizer_source=None):
    train_env = RandomGoalWrapper(env_class, train_tasks, render=render)
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
    model, vec_env, expert_policy, num_steps=10024, batch_size=128, bc_epochs=100
):
    """
    Pre-train both the policy and value networks using behavior cloning from the expert policy.
    """

    print("Collecting expert data for behavior cloning...")
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
    policy.train()
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

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch+1}/{bc_epochs} | "
                f"Actor Loss: {total_actor_loss/len(loader):.5f} | "
                f"Critic Loss: {total_critic_loss/len(loader):.5f}"
            )

    print("Warm start complete.")


def train_tier(save_path, model, vec_env, test_vec_env, bc_policy=None):
    callback = PPOCallback(verbose=1, save_path=save_path, eval_env=test_vec_env)
    if model is None:
        model = PPO(CustomActorCriticPolicy, vec_env, verbose=verbose)
    else:
        model = next_model(model, vec_env)

    if bc_policy is not None:
        warm_start(model, vec_env, bc_policy)
        print("Saving model after warm start...")
        PPO.save(model, "warm-" + save_path)
        evaluate_model(
            "warm-" + save_path,
            test_vec_env,
            test_vec_env.venv.envs[0],
            f"Warm Start {save_path}",
        )
    model.learn(training_iterations, callback=callback)
    vec_env.close()
    return model


def evaluate_model(model_path, env_vec, env_raw, label):
    model = PPO.load(model_path, env=env_vec)
    success, reward = test_on_env(env_vec, env_raw, model)
    print(f"{label} Total reward:", reward)
    print(f"{label} Success percentage:", success)


############################
###### BEGIN TRAINING ######
############################
model = None
shared_normalizer = None
all_test_envs = {}
tiers = [
    {"name": "reach-v2", "label": "Reach", "policy": SawyerReachV2Policy()},
    {
        "name": "pick-place-v2",
        "label": "Pick Place",
        "policy": SawyerPickPlaceV2Policy(),
    },
    {"name": "hammer-v2", "label": "Hammer", "policy": SawyerHammerV2Policy()},
]

for i, tier in enumerate(tiers):
    task_name = tier["name"]
    label = tier["label"]

    mt1 = MT1(task_name, seed=seed)
    all_tasks = mt1.train_tasks
    train_tasks, test_tasks = all_tasks[:-10], all_tasks[-10:]

    train_vec_env, test_vec_env, train_env, test_env = make_envs(
        mt1.train_classes[task_name],
        train_tasks,
        test_tasks,
        render=False,
        normalizer_source=shared_normalizer,
    )

    model = train_tier(
        task_name, model, train_vec_env, test_vec_env, tier.get("policy")
    )

    evaluate_model(task_name, test_vec_env, test_env, label)

    all_test_envs[label] = (test_vec_env, test_env)

    for prev_label, (prev_vec_env, prev_raw_env) in all_test_envs.items():
        if prev_label == label:
            continue
        evaluate_model(
            task_name, prev_vec_env, prev_raw_env, f"{prev_label} (After {label})"
        )

    if i == 0:
        shared_normalizer = train_vec_env
