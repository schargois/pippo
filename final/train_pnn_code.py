from train_pnn import make_envs, train_tier, evaluate_model, test_on_env
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
    save_path = datetime_str + "-" + task_name

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
        save_path, model, train_vec_env, test_vec_env, tier.get("policy")
    )

    print(f"Evaluating model on {task_name}...")
    logger.info(f"Evaluating model on {task_name}...")
    evaluate_model(save_path, test_vec_env, test_env, label)

    all_test_envs[label] = (test_vec_env, test_env)

    for prev_label, (prev_vec_env, prev_raw_env) in all_test_envs.items():
        if prev_label == label:
            continue
        evaluate_model(
            save_path, prev_vec_env, prev_raw_env, f"{prev_label} (After {label})"
        )

    if i == 0:
        shared_normalizer = train_vec_env
