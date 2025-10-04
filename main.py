import argparse
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from agent import PPO
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='train or demo')
    parser.add_argument('--model_dir', default='best_ppo', type=str, help='directory to save/load model')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_steps', default=5*10**5, type=int)
    parser.add_argument('--eval_interval', default=2048, type=int)
    parser.add_argument('--value_clipping', action='store_true', help='Enable reward clipping')
    args = parser.parse_args()

    suffix ='_value_clipping' if args.value_clipping else ''
    model_dir = f'{args.model_dir}{suffix}'

    env = gym.make('CarRacing-v3', render_mode='rgb_array', max_episode_steps=1000)
    env_test = gym.make('CarRacing-v3', render_mode='rgb_array', max_episode_steps=1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    algo = PPO(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        seed=args.seed,
        device=device,
        value_clipping=args.value_clipping
    )

    if args.mode == 'train':
        trainer = Trainer(
            env=env,
            env_test=env_test,
            algo=algo,
            seed=args.seed,
            num_steps=args.num_steps,
            eval_interval=args.eval_interval,
            model_dir=model_dir
        )
        trainer.train()
        trainer.plot()

    elif args.mode == 'demo':
        algo.load_models(model_dir)
        video_folder = "demo_videos"
        env_demo = RecordVideo(env, video_folder, episode_trigger=lambda e: True)
        state, _ = env_demo.reset(seed=1)
        done = False
        while not done:
            action = algo.exploit(state)
            state, _, terminated, truncated, _ = env_demo.step(action)
            done = terminated or truncated
        env_demo.close()

    env.close()
    env_test.close()

if __name__ == '__main__':
    main()
