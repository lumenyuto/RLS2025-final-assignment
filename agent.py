from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from torch import nn

from models import CNN, PPOActor, PPOCritic
from utils import preprocess_state, RunningMeanStd, evaluate_lop_pi

def calculate_advantage(values, rewards, dones, next_values, gamma=0.995, lambd=0.997):
    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.empty_like(rewards)
    advantages[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        advantages[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * advantages[t + 1]
    targets = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return targets, advantages

class RolloutBuffer:
    def __init__(self, buffer_size, state_shape, action_shape, device):
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.uint8, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float32, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float32, device=device)
        self.log_pis = torch.empty((buffer_size, 1), dtype=torch.float32, device=device)
        self._p = 0
        self.buffer_size = buffer_size

    def append(self, state, action, reward, done, log_pi):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self._p = (self._p + 1) % self.buffer_size

    def get(self):
        assert self._p == 0, 'Buffer needs to be full before training.'
        return self.states, self.actions, self.rewards, self.dones, self.log_pis

class Algorithm(ABC):
    @abstractmethod
    def explore(self, state):
        pass
    @abstractmethod
    def exploit(self, state):
        pass
    @abstractmethod
    def is_update(self, steps):
        pass
    @abstractmethod
    def step(self, env, state, t, steps):
        pass
    @abstractmethod
    def update(self):
        pass

class PPO(Algorithm):
    def __init__(self, state_shape, action_shape, device, seed=0,
                 batch_size=256, gamma=0.99, lr=3e-5,
                 rollout_length=2048, num_updates=10, clip_eps=0.2, lambd=0.95,
                 coef_ent=0.01, max_grad_norm=0.5, reward_scaling=True):
        super().__init__()
        torch.manual_seed(seed)
        if device.type == 'cuda': torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        self.buffer = RolloutBuffer(rollout_length, state_shape, action_shape, device)
        self.device = device

        # モデルの構築
        dummy_state = torch.zeros(1, state_shape[2], state_shape[0], state_shape[1], device=device)
        self.cnn = CNN(input_channels=state_shape[2]).to(device)
        cnn_output_dim = self.cnn(dummy_state).shape[-1]
        self.actor = PPOActor(cnn_output_dim, action_shape).to(device)
        self.critic = PPOCritic(cnn_output_dim).to(device)
        self.optim = torch.optim.Adam(list(self.cnn.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        self.batch_size = batch_size
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.num_updates = num_updates
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.reward_scaling=reward_scaling
        if self.reward_scaling:
            self.reward_rms = RunningMeanStd(shape=(1,))

        self.off_course_threshold = 100
        self.off_course_counter = 0


    def explore(self, state):
        state_tensor = preprocess_state(state, self.device)
        with torch.no_grad():
            features = self.cnn(state_tensor)
            action, log_pi = self.actor.sample(features)
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        state_tensor = preprocess_state(state, self.device)
        with torch.no_grad():
            features = self.cnn(state_tensor)
            action = self.actor(features)
        return action.cpu().numpy()[0]

    def is_update(self, steps):
        return steps % self.rollout_length == 0

    def step(self, env, state, t, steps):
        t += 1
        action, log_pi = self.explore(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if reward < 0:
            self.off_course_counter += 1
        else:
            self.off_course_counter = 0

        if self.off_course_counter >= self.off_course_threshold:
            done = True
            reward = -100

        done_masked = done if not truncated else False
        self.buffer.append(state, action, reward, done_masked, log_pi)

        if done:
            t = 0
            next_state, _ = env.reset()
            self.off_course_counter = 0

        return next_state, t

    def update(self):
        states, actions, rewards, dones, log_pis_old = self.buffer.get()
        processed_states = states.float().permute(0, 3, 1, 2) / 255.0

        if self.reward_scaling:
            self.reward_rms.update(rewards.cpu().numpy())

            scaled_rewards = rewards / torch.sqrt(torch.tensor(self.reward_rms.var, device=self.device, dtype=torch.float32) + 1e-8)
            scaled_rewards = torch.clamp(scaled_rewards, -10.0, 10.0)
        else:
            scaled_rewards = rewards

        with torch.no_grad():
            features = self.cnn(processed_states)
            values = self.critic(features)
            next_features = self.cnn(torch.roll(processed_states, -1, 0))
            next_values = self.critic(next_features)

        targets, advantages = calculate_advantage(values, scaled_rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.num_updates):
            indices = np.random.permutation(self.rollout_length)
            for start in range(0, self.rollout_length, self.batch_size):
                idx = indices[start:start+self.batch_size]
                features_batch = self.cnn(processed_states[idx])

                # Critic損失
                loss_critic = (self.critic(features_batch) - targets[idx]).pow(2).mean()

                # Actor損失
                log_pis = evaluate_lop_pi(self.actor.fc(features_batch), self.actor.log_stds, actions[idx])
                mean_entropy = -log_pis.mean()
                ratios = (log_pis - log_pis_old[idx]).exp()
                loss_actor1 = -ratios * advantages[idx]
                loss_actor2 = -torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[idx]
                loss_actor = torch.max(loss_actor1, loss_actor2).mean() - self.coef_ent * mean_entropy

                # 合計損失
                total_loss = loss_actor + 0.5 * loss_critic

                self.optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.cnn.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optim.step()

    def save_models(self, save_dir):
        """モデルの重みを保存する"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.cnn.state_dict(), os.path.join(save_dir, 'cnn.pth'))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'critic.pth'))
        print(f"モデルを {save_dir} に保存しました。")

    def load_models(self, load_dir):
        """モデルの重みを読み込む"""
        self.cnn.load_state_dict(torch.load(os.path.join(load_dir, 'cnn.pth')))
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(load_dir, 'critic.pth')))
        print(f"モデルを {load_dir} から読み込みました。")
