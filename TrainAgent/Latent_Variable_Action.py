import torch

from model.ActionVariationAutoEncoder import ActionVAE
from model.Actor_Critic import Actor,Critic,SingleCritic
from replaybuffer.static_dataset import Maze2d

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import d4rl
import gym
import torch.nn.functional as F
from copy import deepcopy

class LAPOAgent(object):
    def __init__(self,envname = "maze2d-large-v1",latent_dim = 3,load_from_pretrainVAE = True,modelpath = "model/CVAEpretrain",EPOCH=128,tau = 0.001,gamma = 0.99) -> None:
        self.env = gym.make(envname)
        self.state_dim = len(self.env.observation_space.sample())
        self.action_dim = len(self.env.action_space.sample())
        self.latent_dim = latent_dim
        self.Actor = Actor(state_dim=self.state_dim,latent_action_dim=self.latent_dim,range=self.env.action_space.high[0]).cuda()
        self.Critic = SingleCritic(state_dim=self.state_dim,action_dim=self.action_dim)
        self.optimActor = torch.optim.Adam(self.Actor.parameters(),lr = 0.0001)
        self.optimCirtic = torch.optim.Adam(self.Critic.parameters(),lr = 0.001)
        if load_from_pretrainVAE:
            self.Conditional_Action_Variation_AutoEncoder = torch.load(modelpath)
        else:
            self.Conditional_Action_Variation_AutoEncoder = ActionVAE(state_dim=self.state_dim,action_dim=self.action_dim,latent_dim=self.latent_dim).cuda()
        self.staticdataset = Maze2d(envname)
        self.loader = DataLoader(self.staticdataset,batch_size=64)
        self.load_from_pretrain = load_from_pretrainVAE
        self.testenv = deepcopy(self.env)
        self.TargetCritic = deepcopy(self.Critic)
        self.TargetActor = deepcopy(self.Actor)
        self.epoch = EPOCH
        self.validationepisode = 16
        self.tau = tau
        self.gamma = gamma
        # self.rewardindex = 1
        self.writer = SummaryWriter('./logs/LAPO')

    def _softupdate(self):
        for target,param in zip(self.TargetActor.parameters(),self.Actor.parameters()):
            target.data.copy_(
                (1 - self.tau) * target + self.tau * param
            )
        for target,param in zip(self.TargetCritic.parameters(),self.Critic.parameters()):
            target.data.copy_(
                (1 - self.tau) * target + self.tau * param
            )
    
    def getactionfromstate(self,state,noise = True):
        latent_action = self.Actor(state)
        return self.Conditional_Action_Variation_AutoEncoder.deduction(state,latent_action)

    def validate(self):
        reward = 0
        
        for ep in range(self.validationepisode):
            done = False
            state = self.testenv.reset()
            while done == False:
                # action = self.Actor(state).cpu().detach().numpy()
                action = self.getactionfromstate(state,noise=False).cpu().detach().numpy()
                ns,r,done,_ = self.testenv.step(action)
                reward += r
                state = ns
        return reward/self.validationepisode
    def learn(self):
        from tqdm import tqdm
        _id = 1
        for epoch in range(self.epoch):
            for state,action,reward,nextstate in tqdm(self.loader):
                self.state = state.cuda()
                self.action = action.cuda()
                self.reward = reward.cuda()
                self.nextstate = nextstate.cuda()
                self.CriticUpdate()
                self.ActorUpdate()
                if _id % 32 == 0:
                    reward = self.validate()
                    self.writer.add_scalar("LAPO/reward",reward,_id)
                    # self.rewardindex += 1
                _id += 1

    def CriticUpdate(self):
        '''
        map the state into action of latent space
        then map the latent space into real actionspace
        '''
        # self.latent_action = self.Actor(self.state)
        with torch.no_grad():
            nextactions = self.getactionfromstate(self.nextstate)
            nextvalues = self.TargetCritic(self.nextstate,nextactions).squeeze()
            nextvalues = self.gamma * nextvalues + self.reward
        self.optimCirtic.zero_grad()
        currentvalues = self.Critic(self.state,self.action).squeeze()
        loss = F.mse_loss(currentvalues,nextactions)
        loss.backward()
        self.optimCirtic.step()
    
    def ActorUpdate(self):
        self.optimActor.zero_grad()
        currentvalues = self.Critic(self.state,self.getactionfromstate(self.state)).squeeze()
        currentvalue = -torch.mean(currentvalues)
        currentvalue.backward()
        self.optimActor.step()

