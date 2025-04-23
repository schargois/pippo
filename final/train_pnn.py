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

# training_iterations = 40960
training_iterations = 1024
# training_iterations = 20480
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


def test_on_env(vec_environment, gym_env, model, num_episodes=100, progress=True):
    total_rew = 0
    iterate = trange(num_episodes) if progress else range(num_episodes)

    # Collect reward in vectorized env
    for _ in iterate:
        obs = vec_environment.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            next_obs, reward, done, _ = vec_environment.step(action)
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
        self.needs_new_task = True
        render_mode = "human" if render else None
        super().__init__(env_class(render_mode=render_mode))

    def reset(self, **kwargs):
        task = random.choice(self.task_list)
        self.env.set_task(task)
        return self.env.reset(**kwargs)


def warm_start(model, vec_env, expert_policy, num_steps=1024):
    """
    Pre-train the policy network using behavior cloning from the expert policy.
    """
    pass


def make_envs(env_class, train_tasks, test_tasks, render=False):
    train_env = RandomGoalWrapper(env_class, train_tasks, render=render)
    test_env = RandomGoalWrapper(env_class, test_tasks, render=render)

    train_vec_env = DummyVecEnv([lambda: train_env])
    test_vec_env = DummyVecEnv([lambda: test_env])

    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)
    test_vec_env = VecNormalize(test_vec_env, norm_obs=True, norm_reward=True)
    test_vec_env.training = False
    test_vec_env.norm_reward = False

    return train_vec_env, test_vec_env, train_env, test_env


def train_tier(save_path, model, vec_env, test_vec_env, bc_policy=None):
    callback = PPOCallback(verbose=1, save_path=save_path, eval_env=test_vec_env)
    if model is None:
        model = PPO(CustomActorCriticPolicy, vec_env, verbose=verbose)
    else:
        model = next_model(model, vec_env)

    # Warm start the model with BC policy if provided
    if bc_policy is not None:
        warm_start(model, vec_env, bc_policy)

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
        mt1.train_classes[task_name], train_tasks, test_tasks, render=False
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
