import os
import glob
from base64 import b64encode
import math
import torch
import numpy as np
from IPython.display import HTML

def play_mp4(folder_path="videos"):
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    if not video_files:
        print(f"動画ファイルが'{folder_path}' に見つかりません。")
        return
    latest_file = max(video_files, key=os.path.getctime)
    with open(latest_file, 'rb') as f:
        video_data = f.read()
    video_url = f"data:video/mp4;base64,{b64encode(video_data).decode()}"
    return HTML(f"""<video width=600 controls><source src="{video_url}" type="video/mp4"></video>""")

def preprocess_state(state, device):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
    return state_tensor.unsqueeze(0)

def calculate_log_pi(log_stds, noises, actions):
    log_pis = -0.5 * math.log(2 * math.pi) * log_stds.size(-1) - log_stds.sum(dim=-1, keepdim=True) - (0.5 * noises.pow(2)).sum(dim=-1, keepdim=True)
    log_pis -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    return log_pis

def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(means)
    us = means + noises * stds
    actions = torch.tanh(us)
    log_pis = calculate_log_pi(log_stds, noises, actions)
    return actions, log_pis

def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)
