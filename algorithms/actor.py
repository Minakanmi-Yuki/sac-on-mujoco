import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, device):
        super(PolicyNetContinuous, self).__init__()

        self.device = device

        self.fc1 = nn.Linear(state_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.mu = nn.Linear(hidden_dim, action_dim).to(device)
        self.std = nn.Linear(hidden_dim, action_dim).to(device)
        self.action_bound = torch.tensor(action_bound, dtype=torch.float).to(device)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        std = F.softplus(self.std(x)) # softplus保证标准差为正

        return mu, std
    
    def evaluate(self, states):
        mu, std = self.forward(states)
        std = torch.clamp(std, min=np.exp(-20.0), max=np.exp(2.0))
        normal = Normal(mu, std)
        noise = Normal(0, 1)
        epsilon = noise.sample().to(self.device)
        
        actions = mu + std * epsilon
        log_prob = normal.log_prob(actions).sum(dim=1, keepdim=True)
        actions = torch.tanh(actions)
        log_prob = log_prob - torch.log(1-actions.pow(2)+1e-7).sum(dim=1, keepdim=True)
        # print(actions.shape, self.action_bound.shape)
        actions = actions * self.action_bound

        return actions, log_prob 
    
    def act(self, state):
        with torch.no_grad():
            mu, std = self.forward(states=state)
            normal = Normal(mu, std)
            action = normal.sample()
            action = torch.tanh(action).detach().cpu().numpy()

            return action

class PolicyNetDiscrete(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device, *args, **kwargs):
        super(PolicyNetDiscrete, self).__init__(*args, **kwargs)

        self.device = device
        self.fc1 = nn.Linear(state_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc3 = nn.Linear(hidden_dim, action_dim).to(device)
    
    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)
    
    def evaluate(self, states):
        logits = self.forward(states)
        dist = Categorical(logits)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        return actions, log_prob

    def act(self, states):
        logits = self.forward(states)
        dist = Categorical(logits)
        actions = dist.sample()
        return actions