import numpy as np
from time import time
from datetime import timedelta
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt

from utils import play_mp4

class Trainer:
    def __init__(self, env, env_test, algo, seed=0, num_steps=10**6, eval_interval=10**4, num_eval_episodes=3, model_dir='best_model'):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.seed = seed
        self.returns = {'step': [], 'return': []}
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.model_dir = model_dir
        self.best_mean_return = -float('inf')

    def train(self):
        self.start_time = time()
        t = 0
        state, _ = self.env.reset(seed=self.seed)
        for steps in range(1, self.num_steps + 1):
            state, t = self.algo.step(self.env, state, t, steps)
            if self.algo.is_update(steps):
                self.algo.update()
            if steps % self.eval_interval == 0:
                self.evaluate(steps)

    def evaluate(self, steps):
        returns = []
        for _ in range(self.num_eval_episodes):
            state, _ = self.env_test.reset(seed=2**31 - 1 - self.seed)
            episode_return = 0.0
            done = False
            while not done:
                action = self.algo.exploit(state)
                state, reward, terminated, truncated, _ = self.env_test.step(action)
                done = terminated or truncated
                episode_return += reward
            returns.append(episode_return)

        mean_return = np.mean(returns)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)
        print(f'Num steps: {steps:<6}   Return: {mean_return:<5.1f}   Time: {self.time}')

        if mean_return > self.best_mean_return:
            self.best_mean_return = mean_return
            print(f"最高スコアを記録: {self.best_mean_return:.1f}")
            self.algo.save_models(self.model_dir)

    def visualize(self, folder="final_agent_videos"):
        vis_env = gym.make('CarRacing-v3', render_mode='rgb_array')
        vis_env = RecordVideo(vis_env, folder, episode_trigger=lambda e: True)
        state, _ = vis_env.reset(seed=self.seed)
        done = False
        while not done:
            action = self.algo.exploit(state)
            state, _, terminated, truncated, _ = vis_env.step(action)
            done = terminated or truncated
        vis_env.close()
        return play_mp4(folder)

    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Steps', fontsize=18)
        plt.ylabel('Return', fontsize=18)
        plt.yticks(np.arange(-100, 1001, 100))
        plt.title(f'Learning Curve for {self.env.unwrapped.spec.id}', fontsize=20)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
