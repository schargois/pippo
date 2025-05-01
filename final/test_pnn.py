import gymnasium as gym
from stable_baselines3 import PPO
from CustomACNetwork import CustomActorCriticPolicy, CustomNetwork
from stable_baselines3.common.vec_env import DummyVecEnv
from ProgNet import ProgColumn
from Callback import PPOCallback
from stable_baselines3.common.env_util import make_vec_env

from tqdm import trange


import sys
import os
import time
import random
import numpy as np

os.environ["MUJOCO_GL"] = "glfw"

repo_path = os.path.abspath("./final/Metaworld")
sys.path.insert(0, repo_path)
from metaworld import MT1
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy as p

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

# training_iterations = 40960
# training_iterations = 1024
training_iterations = 20480
verbose = 0


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


def test_on_env(
    vec_environment, gym_env, model, num_episodes=5, progress=True, render=False
):
    total_rew = 0
    iterate = trange(num_episodes) if progress else range(num_episodes)

    # for _ in iterate:
    #     obs = vec_environment.reset()
    #     done = False

    #     while not done:
    #         action, _ = model.predict(obs)
    #         next_obs, reward, done, _ = vec_environment.step(action)
    #         if isinstance(reward, np.ndarray):
    #             reward = reward.item()
    #         total_rew += reward
    #         obs = next_obs

    total_success = 0
    iterate = trange(num_episodes) if progress else range(num_episodes)

    for _ in iterate:
        obs, _ = gym_env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            if render:
                gym_env.render()

            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = gym_env.step(action)

            if info.get("success", 0):
                total_success += 1
                break

    return (total_success / num_episodes), (total_rew / num_episodes)


########################################
######### First Tier Training ##########
########################################

mt1_reach = MT1("reach-v2", seed=seed)

reach_env = RandomGoalWrapper(
    mt1_reach.train_classes["reach-v2"], mt1_reach.train_tasks
)
reach_vec_env = DummyVecEnv([lambda: reach_env])

reach_test_env_test = RandomGoalWrapper(
    mt1_reach.train_classes["reach-v2"], mt1_reach.train_tasks, render=True
)
reach_test_vec_env = DummyVecEnv([lambda: reach_test_env_test])


reach_callback = PPOCallback(
    verbose=1, save_path="reach-v2", eval_env=reach_test_vec_env
)

############################################
############### Second Tier ################
############################################

mt1_pick_place = MT1("pick-place-v2", seed=seed)

pick_place_env = RandomGoalWrapper(
    mt1_pick_place.train_classes["pick-place-v2"], mt1_pick_place.train_tasks
)
pick_place_vec_env = DummyVecEnv([lambda: pick_place_env])

pick_place_test_env = RandomGoalWrapper(
    mt1_pick_place.train_classes["pick-place-v2"],
    mt1_pick_place.train_tasks,
    render=True,
)
pick_place_test_vec_env = DummyVecEnv([lambda: pick_place_test_env])

pick_place_callback = PPOCallback(
    verbose=1, save_path="pick-place-v2", eval_env=pick_place_test_vec_env
)


############################################
########### Third Tier Training ############
############################################

mt1_hammer = MT1("hammer-v2", seed=seed)

hammer_env = RandomGoalWrapper(
    mt1_hammer.train_classes["hammer-v2"], mt1_hammer.train_tasks
)
hammer_vec_env = DummyVecEnv([lambda: hammer_env])

hammer_test_env = RandomGoalWrapper(
    mt1_hammer.train_classes["hammer-v2"], mt1_hammer.train_tasks, render=True
)
hammer_test_vec_env = DummyVecEnv([lambda: hammer_test_env])

hammer_callback = PPOCallback(
    verbose=1, save_path="hammer-v2", eval_env=hammer_test_vec_env
)

# Choose model to use
model = PPO.load("warm-reach-v2")

success_pick_place_percentage, total_pick_place_reward = test_on_env(
    pick_place_test_vec_env, pick_place_test_env, model, render=True
)
print("Pick Place Total reward:", total_pick_place_reward)
print("Pick Place Success percentage:", success_pick_place_percentage)

success_reach_percentage, total_reach_reward = test_on_env(
    reach_test_vec_env, reach_test_env_test, model, render=True
)
print("Reach Total reward:", total_reach_reward)
print("Reach Success percentage:", success_reach_percentage)

success_hammer_percentage, total_hammer_reward = test_on_env(
    hammer_test_vec_env, hammer_test_env, model, render=True
)
hammer_test_vec_env.close()
print("Hammer Total reward:", total_hammer_reward)
print("Hammer Success percentage:", success_hammer_percentage)
############################################################
