import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class QValueNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(QValueNetContinuous, self).__init__()
        
        self.device = device
        
        self.fc1 = nn.Linear(state_dim+action_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc3 = nn.Linear(hidden_dim, 1).to(device)
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)