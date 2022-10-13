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
            nn.Linear(8,latent_dim)
        )
        self.sigmaencoder = deepcopy(self.muencoder)
        self.Decoder = nn.Sequential(
            nn.Linear(latent_dim + state_dim,16),
            nn.ReLU(),
            nn.Linear(16,action_dim),
            nn.Tanh()
        )

    def forward(self,state,action):
        if isinstance(state,np.ndarray):
            state = torch.from_numpy(state).cuda().to(torch.float32)
        if isinstance(action,np.ndarray):
            action = torch.from_numpy(action).cuda().to(torch.float32)
        feature = torch.concat([state,action],-1)
        feature = self.Encoder(feature)
        mu = self.muencoder(feature)
        sigma = self.sigmaencoder(feature)
        noise = torch.randn_like(mu)
        sigma = torch.exp(torch.mul(0.5,sigma))
        feature = sigma * noise + mu
        return self.Decoder(torch.concat([state,feature],-1)),mu,sigma
    
    def deduction(self,state,latentvariable=None):
        if isinstance(state,np.ndarray):
            state = torch.from_numpy(state)
        if len(state.shape) == 1:
            state = state.unsqueeze(0).cuda()
        if latentvariable == None:
            latentvariable = torch.randn_like(state)
        feature = torch.concat([state,feature],-1)
        return self.Decoder(feature)

if __name__ == "__main__":
    from random import choice
    # obs = choice()