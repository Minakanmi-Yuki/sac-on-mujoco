import sys
import os
import gymnasium as gym
from runner import Runner
import torch
import argparse
from config import get_config
import math
import torch
import random
import numpy as np


def parser_args(args, parser):
    parser.add_argument('--env_name', type=str, default='Ant-v5')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load_checkpoint', type=int, default=10000)
    all_args = parser.parse_known_args(args)[0]

    return all_args

def make_env(all_args):
    env_name = all_args.env_name
    env = gym.make(env_name, render_mode='rgb_array')
    return env


def main(args):
    parser = get_config()
    all_args = parser_args(args, parser)
    seed = all_args.seed
    all_args.use_wandb = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_env = make_env(all_args)
    eval_env = make_env(all_args)

    config = {
        'all_args': all_args,
        'env': train_env, 
        'eval_env': eval_env,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    runner = Runner(config)
    
    runner.load(checkpoint=int(2200000))
    frames = runner.render()
    frames[0].save(
        f"./gif/{all_args.env_name}_animation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=50,  # 可根据环境步长调整（如0.01s/步对应100ms）
        loop=0
    )

if __name__ == '__main__':
    main(sys.argv[1:])