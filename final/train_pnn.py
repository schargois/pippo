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
import copy

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

# training_iterations = 1
# training_iterations = 4096
# training_iterations = 20480
# training_iterations = 40960
training_iterations = 102400
eval_episodes = 100
# training_iterations = 1024000
verbose = 0
reach_hyperparams = {
    # "learning_rate": 3e-5,
    # "num_steps": 40960,
    # "learning_rate": 1e-3,
    # "batch_size": 4096,
    # "learning_rate": 3e-4,
    "n_steps": 2048,
    "learning_rate": 5e-4,
    "batch_size": 128,
    "ent_coef": 0.001,
}
pick_place_hyperparams = {
    # "learning_rate": 3e-5,
    # "num_steps": 40960,
    # "learning_rate": 1e-3,
    # "batch_size": 4096,
    # "learning_rate": 3e-4,
    "n_steps": 2048,
    "learning_rate": 5e-4,
    "batch_size": 128,
    "ent_coef": 0.001,
}
hammer_hyperparams = {
    # "learning_rate": 3e-5,
    # "num_steps": 40960,
    # "learning_rate": 1e-3,
    # "batch_size": 4096,
    # "learning_rate": 3e-4,
    "n_steps": 2048,
    "learning_rate": 5e-4,
    "batch_size": 128,
    "ent_coef": 0.001,
}


def next_model(model, env, label, hyperparams={}):

    policy_model = model.policy.action_net
    value_model = model.policy.value_net

    policy_column_dict_lst = []
    value_column_dict_lst = []

    for column in policy_model.columns:
        column.freeze()
        policy_column_dict_lst.append(column.state_dict())
    for column in value_model.columns:
        column.freeze()
        value_column_dict_lst.append(column.state_dict())

    mlp_policy_model = model.policy.mlp_extractor.policy_net
    mlp_policy_column_dict_lst = []
    mlp_value_model = model.policy.mlp_extractor.value_net
    mlp_value_column_dict_lst = []

    for column in mlp_policy_model.columns:
        column.freeze()
        mlp_policy_column_dict_lst.append(column.state_dict())
    for column in mlp_value_model.columns:
        column.freeze()
        mlp_value_column_dict_lst.append(column.state_dict())

    model = PPO(
        CustomActorCriticPolicy,
        env,
        verbose=verbose,
        policy_kwargs={
            "action_net_columns": policy_column_dict_lst,
            "value_net_columns": value_column_dict_lst,
            "mlp_policy_columns": mlp_policy_column_dict_lst,
            "mlp_value_columns": mlp_value_column_dict_lst,
            "new_column": True,
        },
        **hyperparams,
    )

    return model


def test_on_env(
    vec_environment, gym_env, model, num_episodes=eval_episodes, progress=True, seed=42
):
    total_rew = 0
    iterate = trange(num_episodes) if progress else range(num_episodes)

    # Collect reward in vectorized env
    for _ in iterate:
        # reset the environment with a seed
        # vec_environment.seed(seed)
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
        # obs, _ = gym_env.reset(seed=seed)
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

    # Collect successes from vectorized env
    total_vec_success = 0
    iterate = trange(num_episodes) if progress else range(num_episodes)
    for _ in iterate:
        # vec_environment.seed(seed)
        obs = vec_environment.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            next_obs, reward, done, info = vec_environment.step(action)
            obs = next_obs

            if info[0].get("success", 0):
                total_vec_success += 1
                break

    return (
        (total_success / num_episodes),
        (total_rew / num_episodes).item(),
        (total_vec_success / num_episodes),
    )


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
    test_env = RandomGoalWrapper(env_class, test_tasks, render=render, seed=seed)

    train_vec_env = DummyVecEnv([lambda: train_env])
    test_vec_env = DummyVecEnv([lambda: test_env])

    if normalizer_source is None:
        train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)
    else:
        # Clone normalization stats from the previous tier
        train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)
        # train_vec_env.obs_rms = normalizer_source.obs_rms
        # train_vec_env.ret_rms = normalizer_source.ret_rms

    test_vec_env = VecNormalize(test_vec_env, norm_obs=True, norm_reward=True)
    test_vec_env.obs_rms = train_vec_env.obs_rms
    test_vec_env.ret_rms = train_vec_env.ret_rms
    test_vec_env.training = False
    test_vec_env.norm_reward = False

    return train_vec_env, test_vec_env, train_env, test_env


def warm_start(
    model, vec_env, expert_policy, num_steps=10024, batch_size=128, bc_epochs=200
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


def train_tier(
    save_path, model, vec_env, test_vec_env, bc_policy=None, pnn=True, hyperparams={}
):
    callback = PPOCallback(verbose=1, save_path=save_path, eval_env=test_vec_env)
    print(f"pnn: {pnn}")
    if model is None:
        model = PPO(CustomActorCriticPolicy, vec_env, verbose=verbose, **hyperparams)
    elif pnn:
        model = next_model(model, vec_env, save_path, hyperparams=hyperparams)

    # if bc_policy is not None:
    #     warm_start(model, vec_env, bc_policy)
    #     print("Saving model after warm start...")
    #     PPO.save(model, "warm-" + save_path)
    #     evaluate_model(
    #         "warm-" + save_path,
    #         test_vec_env,
    #         test_vec_env.venv.envs[0],
    #         f"Warm Start {save_path}",
    #     )
    print("Training model...")
    print(f"policy net column map: {model.policy.mlp_extractor.policy_net.colMap}")
    for column in model.policy.mlp_extractor.policy_net.columns:
        print(f"Column ID: {column.colID} is frozen: {column.isFrozen}")
    print(f"action net column map: {model.policy.action_net.colMap}")
    for column in model.policy.action_net.columns:
        print(f"Column ID: {column.colID} is frozen: {column.isFrozen}")
    print(f"value net column map: {model.policy.value_net.colMap}")
    for column in model.policy.value_net.columns:
        print(f"Column ID: {column.colID} is frozen: {column.isFrozen}")
    model.learn(training_iterations, callback=callback, progress_bar=True)
    vec_env.close()
    return model


def evaluate_model(model_path, env_vec, env_raw, label, hardcode=-1):
    model = PPO.load(model_path, env=env_vec)
    model.policy.mlp_extractor.output_index_hardcode = hardcode
    model.policy.action_net.output_index_hardcode = hardcode
    model.policy.value_net.output_index_hardcode = hardcode
    success, reward, vec_success = test_on_env(env_vec, env_raw, model)
    print(f"{label} Total reward:", reward)
    print(f"{label} Success percentage:", success)
    print(f"{label} Vectorized success percentage:", vec_success)


############################
###### BEGIN TRAINING ######
############################
model = None
shared_normalizer = None
all_test_envs = {}
tiers = [
    {
        "name": "reach-v2",
        "label": "Reach",
        "policy": SawyerReachV2Policy(),
        "hyperparams": reach_hyperparams,
    },
    {
        "name": "pick-place-v2",
        "label": "Pick Place",
        "policy": SawyerPickPlaceV2Policy(),
        "hyperparams": pick_place_hyperparams,
    },
    {
        "name": "hammer-v2",
        "label": "Hammer",
        "policy": SawyerHammerV2Policy(),
        "hyperparams": hammer_hyperparams,
    },
]

for i in reversed(range(1, 2)):
    pnn = i
    shared_normalizer = None
    for i, tier in enumerate(tiers):
        task_name = tier["name"]
        label = tier["label"]
        hyperparams = tier["hyperparams"]

        env_class = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{task_name}-goal-observable"]
        mt1 = MT1(task_name, seed=seed)
        train_tasks, test_tasks = mt1.train_tasks[:-10], mt1.train_tasks[-10:]

        train_vec_env, test_vec_env, train_env, test_env = make_envs(
            env_class,
            train_tasks,
            test_tasks,
            render=False,
            normalizer_source=shared_normalizer,
        )

        model = train_tier(
            task_name,
            model,
            train_vec_env,
            test_vec_env,
            tier.get("policy"),
            pnn=pnn,
            hyperparams=hyperparams,
        )

        print(f"Evaluating model on {task_name}...")
        # observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
        #     f"{task_name}-goal-observable"
        # ]
        # test_env = observable_cls(seed=42)
        # test_vec_env = DummyVecEnv([lambda: test_env])

        evaluate_model(task_name, test_vec_env, test_env, label)

        all_test_envs[label] = (test_vec_env, test_env)

        for prev_label, (prev_vec_env, prev_raw_env) in all_test_envs.items():
            if prev_label == label:
                continue
            evaluate_model(
                task_name, prev_vec_env, prev_raw_env, f"{prev_label} (After {label})"
            )
            # # print(f"Number of Columns: {model.policy.mlp_extractor.policy_net.numCols}")
            for index, id in enumerate(model.policy.action_net.colMap.keys()):
                # print(f"Column ID: {model.policy.mlp_extractor.policy_net.getColumn(id).colID}")
                # print(f"Column Frozen: {model.policy.mlp_extractor.policy_net.getColumn(id).isFrozen}")
                # print(f"Column Parents: {model.policy.mlp_extractor.policy_net.getColumn(i).parentCols}")
                evaluate_model(
                    task_name,
                    prev_vec_env,
                    prev_raw_env,
                    f"{prev_label} (Specific Column {id}, {index})",
                    hardcode=id,
                )

        if i == 0:
            shared_normalizer = train_vec_env
