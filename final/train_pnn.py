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

# os.environ["MUJOCO_GL"] = "glfw"

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

    model = PPO(CustomActorCriticPolicy, env, verbose=verbose, policy_kwargs={"policy_columns": policy_columns_lst, "value_columns": value_columns_lst})
    return model

def test_on_env(vec_environment, gym_env, model, num_episodes=100, progress=True):
    total_rew = 0
    iterate = (trange(num_episodes) if progress else range(num_episodes))

    for _ in iterate:
        obs = vec_environment.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            next_obs, reward, done, _ = vec_environment.step(action)
            total_rew += reward
            obs = next_obs

    total_success = 0
    iterate = (trange(num_episodes) if progress else range(num_episodes))

    for _ in iterate:
        obs, _ = gym_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            next_obs, reward, done, truncated, _ = gym_env.step(action)
            obs = next_obs

            if done: total_success += 1
            if truncated: break

    return (total_success / num_episodes), (total_rew / num_episodes).item()

########################################
######### First Tier Training ##########
########################################

mt1_reach = MT1("reach-v2", seed=seed)
policy = p()


env = mt1_reach.train_classes["reach-v2"]()
task = random.choice(mt1_reach.train_tasks)
env.set_task(task)
new_goal = np.random.uniform(low=[-0.2, 0.4, 0.05], high=[0.2, 0.8, 0.3])
env._target_pos = new_goal
env.goal = new_goal

reach_vec_env = DummyVecEnv([lambda: env])

reach_test_env_test = mt1_reach.train_classes["reach-v2"]()
task_reach_test = random.choice(mt1_reach.train_tasks)
reach_test_env_test.set_task(task_reach_test)
task_reach_goal = np.random.uniform(low=[-0.2, 0.4, 0.05], high=[0.2, 0.8, 0.3])
reach_test_env_test._target_pos = task_reach_goal
reach_test_env_test.goal = task_reach_goal

reach_test_vec_env = DummyVecEnv([lambda: reach_test_env_test])

reach_callback = PPOCallback(verbose=1, save_path="reach-v2", eval_env=reach_test_vec_env)

model = PPO(CustomActorCriticPolicy, env, verbose=verbose)
model.learn(training_iterations, callback=reach_callback)

env.close()
# load the best model from reach-v2
model = PPO.load("reach-v2")
success_percentage, total_reward = test_on_env(reach_test_vec_env, reach_test_env_test, model)
print("Total reward:", total_reward)
print("Success percentage:", success_percentage)

############################################
############### Second Tier ################
############################################
mt1_pick_place = MT1("pick-place-v2", seed=seed)
policy = p()

env = mt1_pick_place.train_classes["pick-place-v2"]()
task = random.choice(mt1_pick_place.train_tasks)
env.set_task(task)
new_goal = np.random.uniform(low=[-0.2, 0.4, 0.05], high=[0.2, 0.8, 0.3])
env._target_pos = new_goal
env.goal = new_goal

pick_place_vec_env = DummyVecEnv([lambda: env])

pick_place_env_test = mt1_pick_place.train_classes["pick-place-v2"]()
task_pick_place_test = random.choice(mt1_pick_place.train_tasks)
pick_place_env_test.set_task(task_pick_place_test)
task_pick_place_goal = np.random.uniform(low=[-0.2, 0.4, 0.05], high=[0.2, 0.8, 0.3])
pick_place_env_test._target_pos = task_pick_place_goal
pick_place_env_test.goal = task_pick_place_goal

pick_place_test_vec_env = DummyVecEnv([lambda: pick_place_env_test])

pick_place_callback = PPOCallback(verbose=1, save_path="pick-place-v2", eval_env=pick_place_test_vec_env)

model = next_model(model, pick_place_vec_env)
model.learn(training_iterations, callback=pick_place_callback)

env.close()

model = PPO.load("pick-place-v2")
success_pick_place_percentage, total_pick_place_reward = test_on_env(pick_place_test_vec_env, pick_place_env_test, model)
print("Pick Place Total reward:", total_pick_place_reward)
print("Pick Place Success percentage:", success_pick_place_percentage)
success_reach_percentage, total_reach_reward = test_on_env(reach_test_vec_env, reach_test_env_test, model)
print("Reach Total reward:", total_reach_reward)
print("Reach Success percentage:", success_reach_percentage)


############################################
########### Third Tier Training ############
############################################

mt1_hammer = MT1("hammer-v2", seed=seed)
policy = p()

env = mt1_hammer.train_classes["hammer-v2"]()
task = random.choice(mt1_hammer.train_tasks)
env.set_task(task)
new_goal = np.random.uniform(low=[-0.2, 0.4, 0.05], high=[0.2, 0.8, 0.3])
env._target_pos = new_goal
env.goal = new_goal

hammer_vec_env = DummyVecEnv([lambda: env])

hammer_test_env_test = mt1_hammer.train_classes["hammer-v2"]()
task_hammer_test = random.choice(mt1_hammer.train_tasks)
hammer_test_env_test.set_task(task_hammer_test)
task_hammer_goal = np.random.uniform(low=[-0.2, 0.4, 0.05], high=[0.2, 0.8, 0.3])
hammer_test_env_test._target_pos = task_hammer_goal
hammer_test_env_test.goal = task_hammer_goal

hammer_test_vec_env = DummyVecEnv([lambda: hammer_test_env_test])

hammer_callback = PPOCallback(verbose=1, save_path="hammer-v2", eval_env=hammer_test_vec_env)

env.close()

model = next_model(model, hammer_vec_env)
model.learn(training_iterations, callback=hammer_callback)
env.close()


model = PPO.load("hammer-v2")
success_pick_place_percentage, total_pick_place_reward = test_on_env(pick_place_test_vec_env, pick_place_env_test, model)
print("Pick Place Total reward:", total_pick_place_reward)
print("Pick Place Success percentage:", success_pick_place_percentage)
success_reach_percentage, total_reach_reward = test_on_env(reach_test_vec_env, reach_test_env_test, model)
print("Reach Total reward:", total_reach_reward)
print("Reach Success percentage:", success_reach_percentage)
success_hammer_percentage, total_hammer_reward = test_on_env(hammer_test_vec_env, hammer_test_env_test, model)
print("Hammer Total reward:", total_hammer_reward)
print("Hammer Success percentage:", success_hammer_percentage)
##########################################