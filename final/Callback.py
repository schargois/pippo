import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
import datetime
import os

class PPOCallback(BaseCallback):
    def __init__(self, verbose=0, save_path='default', eval_env=None):
        super(PPOCallback, self).__init__(verbose)
        self.rewards = []

        self.save_freq = 1024
        self.min_reward = -np.inf
        self.actor = None
        self.eval_env = eval_env

        self.save_path = save_path
        self.eval_steps = []
        self.eval_rewards = []

    def _init_callback(self) -> None:
        pass

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.actor = PPOActor(model=self.model)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        episode_info = self.model.ep_info_buffer
        rewards = [ep_info['r'] for ep_info in episode_info]
        mean_rewards = np.mean(rewards)
        self.rewards.append(mean_rewards)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.eval_env is None:
            return True

        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps != 0:
            mean_reward = evaluate_policy(self.actor, environment=self.eval_env, num_episodes=20)
            print(f'evaluating {self.num_timesteps=}, {mean_reward=}=======')

            self.eval_steps.append(self.num_timesteps)
            self.eval_rewards.append(mean_reward)
            if mean_reward > self.min_reward:
                self.min_reward = mean_reward
                self.model.save(self.save_path)
                print(f'model saved on eval reward: {self.min_reward}')

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print(f'model saved on eval reward: {self.min_reward}')

        plt.plot(self.eval_steps, self.eval_rewards, c='red')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Rewards over Episodes')

        directory = os.path.join('plots')

        os.makedirs(directory, exist_ok=True)
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{self.save_path}.png"
        plt.savefig(os.path.join(directory, filename))
        plt.close()


def evaluate_policy(actor, environment, num_episodes=100, progress=True):
    """
        Returns the mean trajectory reward of rolling out `actor` on `environment.

        Parameters
        - actor: PPOActor instance, defined in Part 1.
        - environment: classstable_baselines3.common.vec_env.VecEnv instance.
        - num_episodes: Total number of trajectories to collect and average over.
    """

    total_rew = 0
    iterate = (trange(num_episodes) if progress else range(num_episodes))

    for _ in iterate:
        obs = environment.reset()
        done = False

        while not done:
            action = actor.select_action(obs)
            next_obs, reward, done, info = environment.step(action)
            total_rew += reward
            obs = next_obs

    return (total_rew / num_episodes).item()


def success_rate(actor, environment, num_episodes=100, progress=True):
    """
        Returns the percentage of successful trajectories of `actor` on `environment`.

        Parameters
        - actor: PPOActor instance, defined in Part 1.
        - environment: Gymnasium environment.
        - num_episodes: Total number of trajectories to collect and average over.
    """

    total_success = 0
    iterate = (trange(num_episodes) if progress else range(num_episodes))

    for _ in iterate:
        obs, info = environment.reset()
        done = False

        while not done:
            action = actor.select_action(obs)
            next_obs, reward, done, truncated, info = environment.step(action)
            obs = next_obs

            if done: total_success += 1
            if truncated: break

    return (total_success / num_episodes)

class PPOActor():
    def __init__(self, ckpt: str=None, environment: VecEnv=None, model=None):
        '''
          Requires environment to be a 1-vectorized environment

          The `ckpt` is a .zip file path that leads to the checkpoint you want
          to use for this particular actor.

          If the `model` variable is provided, then this constructor will store
          that as the internal representing model instead of loading one from the
          checkpoint path
        '''
        assert ckpt is not None or model is not None

        if model is not None:
            self.model = model
            return

        # TODO: Load checkpoint
        self.model = PPO.load(ckpt, env=environment)
        # END TODO

    def select_action(self, obs):
        '''Gives the action prediction of this particular actor'''

        # TODO: Select action
        return self.model.predict(obs)[0]
        # END TODO