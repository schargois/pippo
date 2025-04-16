import sys
import os
import time
import random
import numpy as np

os.environ["MUJOCO_GL"] = "glfw"

repo_path = os.path.abspath("./Metaworld")
sys.path.insert(0, repo_path)
from metaworld import MT1
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy as p

mt1 = MT1("pick-place-v2", seed=42)
policy = p()

env = mt1.train_classes["pick-place-v2"](render_mode="human")

for i in range(5):
    print(f"\n--- Run #{i + 1} ---")

    task = random.choice(mt1.train_tasks)
    env.set_task(task)

    # Optional: Set a new goal position (for more variation)
    new_goal = np.random.uniform(low=[-0.2, 0.4, 0.05], high=[0.2, 0.8, 0.3])
    env._target_pos = new_goal
    env.goal = new_goal

    obs, info = env.reset()
    done = False

    while not done:
        a = policy.get_action(obs)
        obs, _, _, _, info = env.step(a)
        env.render()
        time.sleep(0.01)
        done = int(info["success"]) == 1
    for j in range(20):
        a = policy.get_action(obs)
        obs, _, _, _, info = env.step(a)
        env.render()
        time.sleep(0.01)

env.close()
