import argparse

def get_config():
    parser = argparse.ArgumentParser(
        description='soft-actor-critic', formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--max_timestep', type=int, default=int(3e6))
    parser.add_argument('--prep_step', type=int, default=int(1e4))
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--target_entropy', type=float)
    parser.add_argument('--buffer_maxlen', type=int, default=int(1e6))
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--alpha_lr', type=float, default=3e-4)

    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--eval_round', type=int, default=20)
    parser.add_argument('--use_wandb', type=bool, default=True)

    return parser
