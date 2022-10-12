import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
class ActionVAE(nn.Module):
    def __init__(self,state_dim,action_dim,latent_dim) -> None:
        super(ActionVAE,self).__init__()
        
        self.statedim = state_dim
        self.actiondim = action_dim
        self.latentdim = latent_dim
        self.Encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16)
        )
        self.muencoder = nn.Sequential(
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,4)
        )
        self.sigmaencoder = deepcopy(self.muencoder)