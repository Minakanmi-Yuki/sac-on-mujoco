import torch
import gymnasium as gym
from algorithms.trainer import SACTrainer
from PIL import Image
import wandb
import os

class Runner(object):
    def __init__(self, config):
        
        self.all_args = config['all_args']
        self.env = config['env']
        self.eval_env = config['eval_env']
        self.device = config['device']

        self.env_name = self.all_args.env_name
        self.max_timestep = self.all_args.max_timestep
        self.prep_step = self.all_args.prep_step
        self.batch_size = self.all_args.batch_size

        # trainer parameters
        self.hidden_dim = self.all_args.hidden_dim
        self.gamma = self.all_args.gamma
        self.tau = self.all_args.tau
        
        self.buffer_maxlen = self.all_args.buffer_maxlen
        self.actor_lr = self.all_args.actor_lr
        self.critic_lr = self.all_args.critic_lr
        self.alpha_lr = self.all_args.alpha_lr
        
        self.act_space = self.env.action_space
        self.action_dim = self.act_space.shape[0]
        self.action_bound = self.act_space.high
        self.state_dim = self.env.observation_space.shape[0]
        self.target_entropy = torch.tensor(-self.act_space.shape[0]).to(self.device)

        self.use_wandb = self.all_args.use_wandb
        self.eval_interval = self.all_args.eval_interval
        self.eval_round = self.all_args.eval_round
        self.log_interval = self.all_args.log_interval
        self.save_interval = self.all_args.save_interval

        self.trainer = SACTrainer(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            action_bound=self.action_bound,
            device=self.device,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            alpha_lr=self.alpha_lr,
            gamma=self.gamma,
            tau=self.tau,
            target_entropy=self.target_entropy,
            buffer_maxlen=self.buffer_maxlen,
        )

        if self.use_wandb:
            wandb.init(
                project='sac-on-mujoco',
                name=f'sac-{self.env_name}',
                config={
                    'batch_size': self.batch_size,
                    'hidden_dim': self.hidden_dim,
                    'gamma': self.gamma,
                    'tau': self.tau,
                    'buffer_maxlen': self.buffer_maxlen,
                    'actor_lr': self.actor_lr,
                    'critic_lr': self.critic_lr,
                    'alpha_lr': self.alpha_lr,
                    'target_entropy': self.target_entropy
                }
            )
    
    # 训练开始前，使用随机动作进行交互，保证buffer数据充足
    def warmup(self):
        obs, info = self.env.reset()
        done = False
        for timestep in range(self.prep_step):
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.trainer.buffer.add(obs, action, reward, next_obs, done)    
            obs = next_obs
            if done:
                obs, info = self.env.reset()
                done=False

    def load(self, checkpoint):
        load_dir = f'./agents/{self.env_name}/'
        self.trainer.actor.load_state_dict(torch.load(load_dir+f'actor-checkpoints-{checkpoint}'))
        self.trainer.q1.load_state_dict(torch.load(load_dir+f'q1-checkpoints-{checkpoint}'))
        self.trainer.q2.load_state_dict(torch.load(load_dir+f'q2-checkpoints-{checkpoint}'))
        self.trainer.log_alpha = torch.load(load_dir+f'log_alpha-checkpoints-{checkpoint}')

    def render(self):
        obs, info = self.eval_env.reset()
        done = False
        frames = []
        while not done:
            action = self.trainer.take_action(obs)
            next_obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            obs = next_obs
            frame = self.eval_env.render()
            frames.append(Image.fromarray(frame))
        return frames

    def eval(self):
        avg_reward = 0
        for round in range(self.eval_round):
            obs, info = self.eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.trainer.take_action(obs)
                next_obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                obs = next_obs
            avg_reward += total_reward
        return {'eval_reward': avg_reward / self.eval_round}

    # 使用trainer与环境交互
    def train(self):
        self.warmup()
        obs, info = self.env.reset()
        total_reward = 0
        for timestep in range(self.prep_step, self.max_timestep):
            action = self.trainer.take_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.trainer.buffer.add(obs, action, reward, next_obs, done)
            total_reward += reward
            obs = next_obs
            train_log = self.trainer.update(self.batch_size)
            if done:
                obs, info = self.env.reset()
                done=False
                print(f'Timestep: {timestep}, total reward: {total_reward}')
                if self.use_wandb:
                    wandb.log({'train_reward': total_reward}, step=timestep)
                total_reward=0
            
            if (timestep+1) % self.log_interval == 0 and self.use_wandb:
                wandb.log(train_log, step=timestep)
            
            if (timestep+1) % self.save_interval == 0:
                save_dir = f'./agents/{self.env_name}/'
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                torch.save(self.trainer.actor.state_dict(), save_dir+f'actor-checkpoints-{timestep+1}')
                torch.save(self.trainer.q1.state_dict(), save_dir+f'q1-checkpoints-{timestep+1}')
                torch.save(self.trainer.q2.state_dict(), save_dir+f'q2-checkpoints-{timestep+1}')
                torch.save([self.trainer.log_alpha], save_dir+f'log_alpha-checkpoints-{timestep+1}')
            
            if (timestep+1) % self.eval_interval == 0:
                eval_log = self.eval()
                if self.use_wandb:
                    wandb.log(eval_log, step=timestep)

        wandb.finish()
                
                
                
                

                
