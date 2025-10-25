import torch
import torch.nn.functional as F
import numpy as np
from algorithms.critic import QValueNetContinuous
from algorithms.actor import PolicyNetContinuous
from utils import ReplayBuffer

class SACTrainer():
    def __init__(self, state_dim, hidden_dim, action_dim, device, actor_lr, critic_lr, 
                 alpha_lr, gamma, tau, target_entropy, buffer_maxlen,
                 action_bound=None):

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy

        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound=action_bound, device=device)
        self.q1 = QValueNetContinuous(state_dim, hidden_dim, action_dim, device=device)
        self.q2 = QValueNetContinuous(state_dim, hidden_dim, action_dim, device=device)
        self.target_q1 = QValueNetContinuous(state_dim, hidden_dim, action_dim, device=device)
        self.target_q2 = QValueNetContinuous(state_dim, hidden_dim, action_dim, device=device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float).to(device)
        self.log_alpha.requires_grad = True  

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.buffer = ReplayBuffer(buffer_maxlen)
    
    def calc_td_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor.evaluate(next_states)
        entropy = -log_prob
        target_q1 = self.target_q1(next_states, next_actions)
        target_q2 = self.target_q2(next_states, next_actions)
        next_values = torch.min(target_q1, target_q2) + torch.exp(self.log_alpha) * entropy
        td_target = rewards + self.gamma * next_values * (1 - dones)

        return td_target
    
    def soft_update(self):
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.actor.act(state)[0]

    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1,1).to(self.device)

        td_target = self.calc_td_target(rewards, next_states, dones)
        q1_loss = F.mse_loss(self.q1(states, actions), td_target.detach())
        q2_loss = F.mse_loss(self.q2(states, actions), td_target.detach())
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        
        hat_actions, log_prob = self.actor.evaluate(states)
        entropy = -log_prob
        q1_val = self.q1(states, hat_actions)
        q2_val = self.q2(states, hat_actions)
        actor_loss = torch.mean(-torch.exp(self.log_alpha) * entropy - torch.min(q1_val, q2_val))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * torch.exp(self.log_alpha))
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update()

        train_log = {
            'actor_loss': actor_loss.item(),
            'q1_loss': q1_loss.item(), 
            'q2_loss': q2_loss.item(),
            'log_alpha_loss': alpha_loss.item() 
        }

        return train_log