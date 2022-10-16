import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
class Actor(nn.Module):
    def __init__(self,state_dim,latent_action_dim,range = 1) -> None:
        super(Actor,self).__init__()
        '''
        map state into latent action
        '''
        self.Actorencoder = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,latent_action_dim),
            nn.Tanh()
        )
        self.range = range

    def forward(self,state:torch.Tensor):
        if isinstance(state,np.ndarray):
            state = torch.from_numpy(state).cuda().to(torch.float32)
        return self.range * self.Actorencoder(state)



class Critic(nn.Module):
    def __init__(self,state_dim,action_dim) -> None:
        super(Critic,self).__init__()
        self.Stateactionvaluenetleft = nn.Sequential(
            nn.Linear(state_dim + action_dim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        self.Stateactionvaluenetright = deepcopy(self.Stateactionvaluenetleft)
        self.Statevaluenet = nn.Sequential(
            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    
    def forward(self,state:torch.Tensor,action:torch.Tensor):
        if isinstance(state,np.ndarray):
            state = torch.from_numpy(state).cuda().to(torch.float32)
        if isinstance(action,np.ndarray):
            action = torch.from_numpy(action).cuda().to(torch.float32)
        
        feature = torch.concat([state,action],-1)
        return self.Stateactionvaluenetleft(feature),self.Stateactionvaluenetright(feature),self.Statevaluenet(state)

class SingleCritic(nn.Module):
    def __init__(self,state_dim,action_dim) -> None:
        super(SingleCritic,self).__init__()
        self.Criticnet = nn.Sequential(
            nn.Linear(state_dim + action_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )        
    
    def forward(self,state:torch.Tensor,action:torch.Tensor):
        if isinstance(state,np.ndarray):
            state = torch.from_numpy(state).cuda().to(torch.float32)
        if isinstance(action,np.ndarray):
            action = torch.from_numpy(action).cuda().to(torch.float32)
        
        feature = torch.concat([state,action],-1)
        return self.Criticnet(feature)
        # return self.Stateactionvaluenetleft(feature),self.Stateactionvaluenetright(feature),self.Statevaluenet(state)
