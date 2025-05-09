import os
import time
import logging
import csv
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from train_pnn import (
    train_tier,
    make_envs,
    test_on_env,
)
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.policies.sawyer_hammer_v2_policy import SawyerHammerV2Policy
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld import MT1

import numpy as np
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    help="Training mode (e.g., ppo, bc+pnn). If not provided, runs all modes.",
)
args = parser.parse_args()

if args.mode:
    mode_list = [args.mode]
else:
    mode_list = ["ppo", "bc+pnn", "ppo+pnn", "bc+ppo+pnn"]


def logprint(msg):
    print(msg)
    logger.info(msg)


seed = 42
np.random.seed(seed)
random.seed(seed)

datetime_str = time.strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    filename=f"eval_run_{datetime_str}.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    filemode="w",
)
logger = logging.getLogger()

os.makedirs("plots", exist_ok=True)
csv_path = f"plots/eval_results_{datetime_str}.csv"
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Mode", "Task", "Context", "SuccessRate", "AvgReward"])

tiers = [
    {
        "name": "reach-v2",
        "label": "Reach",
        "policy": SawyerReachV2Policy(),
        "bc-epochs": 150,
    },
    {
        "name": "pick-place-v2",
        "label": "Pick Place",
        "policy": SawyerPickPlaceV2Policy(),
        "bc-epochs": 250,
    },
    {
        "name": "hammer-v2",
        "label": "Hammer",
        "policy": SawyerHammerV2Policy(),
        "bc-epochs": 350,
    },
]

all_results = {}

for mode in mode_list:
    logprint(f"\n===== Starting Evaluation Mode: {mode} =====")
    model = None
    shared_normalizer = None
    all_test_envs = {}

    for i, tier in enumerate(tiers):
        task_name = tier["name"]
        label = tier["label"]
        expert_policy = tier["policy"]
        bc_epochs = tier["bc-epochs"]

        mode_tag = f"{datetime_str}-{mode.replace('+', '_')}-{task_name}"

        env_class = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{task_name}-goal-observable"]
        mt1 = MT1(task_name, seed=seed)
        train_tasks = mt1.train_tasks[:-10]
        test_tasks = mt1.train_tasks[-10:]

        train_vec_env, test_vec_env, train_env, test_env = make_envs(
            env_class,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            render=False,
            normalizer_source=shared_normalizer,
        )

        model = train_tier(
            save_path=mode_tag,
            model=model,
            vec_env=train_vec_env,
            test_vec_env=test_vec_env,
            bc_policy=expert_policy if "bc" in mode else None,
            train_ppo="ppo" in mode,
            use_pnn="pnn" in mode,
            bc_epochs=bc_epochs,
        )

        logprint(f"Evaluating model on {task_name}...")
        success, reward = test_on_env(test_vec_env, test_env, model)
        logprint(f"{mode} {label} - Success: {success}, Reward: {reward}")
        with open(csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow([mode, label, "Current", success, reward])

        all_results[(mode, label)] = (test_vec_env, test_env)

        for prev_label, (prev_vec_env, prev_env) in all_test_envs.items():
            if prev_label == label:
                continue
            logprint(f"Evaluating on previous task: {prev_label} (After {label})")
            success, reward = test_on_env(prev_vec_env, prev_env, model)
            logprint(
                f"{mode} {prev_label} (After {label}) - Success: {success}, Reward: {reward}"
            )
            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow(
                    [mode, prev_label, f"After {label}", success, reward]
                )

        all_test_envs[label] = (test_vec_env, test_env)
        if i == 0:
            shared_normalizer = train_vec_env

logprint("\nFinal evaluation file complete. All combinations have been run.")
logprint(f"Results saved to {csv_path}")

# Summary plots
summary_df = pd.read_csv(csv_path)

for metric in ["SuccessRate", "AvgReward"]:
    pivot = summary_df[summary_df.Context == "Current"].pivot(
        index="Task", columns="Mode", values=metric
    )
    pivot.plot(kind="bar", title=f"{metric} by Mode and Task")
    plt.ylabel(metric)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"plots/summary_{metric}_{datetime_str}.png")
    plt.close()

# Transfer (backward) plots
for metric in ["SuccessRate", "AvgReward"]:
    transfer_df = summary_df[summary_df.Context.str.startswith("After")]
    if not transfer_df.empty:
        transfer_df["ContextLabel"] = (
            transfer_df["Task"] + " (" + transfer_df["Context"] + ")"
        )
        pivot = transfer_df.pivot(index="ContextLabel", columns="Mode", values=metric)
        pivot.plot(kind="bar", title=f"{metric} on Previous Tasks")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"plots/transfer_{metric}_{datetime_str}.png")
        plt.close()

logprint("Summary and transfer plots saved.")
