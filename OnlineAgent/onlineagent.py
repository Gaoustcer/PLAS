import torch
import torch.nn as nn

from model.Actor_Critic import Actor,Critic
from Policy.BasePolicy import BasePolicy
import gym
import d4rl
from copy import deepcopy
import numpy as np
import multiprocessing as mp
from replaybuffer.replay import buffer
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from noise.OUnoise import OrnsteinUhlenbeckActionNoise

def f(state):
    torch.from_numpy(state).cuda()
    return np.random.random(2)

class OnlineAgent(BasePolicy):
    def __init__(self, envname: str, parallelnumber: int, EPOCH=128,nabla = 0.75,gamma = 0.98,lr = 0.0001) -> None:
        super().__init__(envname, parallelnumber, EPOCH)
        self.env = gym.make(envname)
        self.state_dim = len(self.env.observation_space.sample())
        self.action_dim = len(self.env.action_space.sample())
        self.actornet = Actor(state_dim=len(self.env.observation_space.sample()),latent_action_dim=len(self.env.action_space.sample())).cuda()
        self.criticnet = Critic(state_dim=self.state_dim,action_dim=self.action_dim).cuda()
        self.lr = lr
        self.gamma = gamma
        self.optimizeractor = torch.optim.Adam(self.actornet.parameters(),lr = self.lr)
        self.optimizercritic = torch.optim.Adam(self.criticnet.parameters(),lr = self.lr*10,weight_decay=1e-2)
        self.targetactornet = deepcopy(self.actornet)
        self.targetcriticnet = deepcopy(self.criticnet)
        self.range = -self.env.action_space.low[0]
        self.sigma = 0.1
        self.validatetime = 16
        self.tau = 0.001
        self.epoch = 128
        # self.collectsteps = 1024
        self.env = gym.make(envname)
        self.sampletime = 64
        self.sampleepi = 4
        self.episteps = 1024
        self.nabla = nabla
        self.buffer = buffer(state_dim=self.state_dim,action_dim=self.action_dim,max_size=int(1e6))
        self.writer = SummaryWriter("./logs/DDPGDueling")
        self.losstriger = 1
        self.testenv = gym.make(envname)
        self.noiseadd = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim),sigma=0.2)
        self.rewardindex = 1
    def action(self,state,noise=True):
        action = self.actornet(state).cpu().detach().numpy()
        if noise == False:
            return action
        action += self.noiseadd.noise()
        return action
        return np.clip(np.random.normal(action,self.sigma),a_min=-self.range,a_max=self.range)

    def learnaneps(self):
        _id = 0
        from tqdm import tqdm
        for _ in tqdm(range(self.sampleepi)):
            done = False
            state = self.env.reset()
            while done == False:
                action = self.action(state)
                ns,r,done,_ = self.env.step(action)
                _id += 1
                self.buffer.push_memory(state,action,r,ns)
                self.learnanapoch()
                # if id % 8 == 0:
                self._softupdate()
                if _id % 128 == 0:
                    reward = self.policyvalidate()
                    self.writer.add_scalar('reward',reward,self.rewardindex)
                    self.rewardindex += 1
                state = ns
    
    def epslearn(self):
        from tqdm import tqdm
        done = True
        for step in tqdm(range(self.episteps)):
            if done:
                state = self.env.reset()
                done = False
            action = self.action(state)
            ns,r,done,_ = self.env.step(action)
            self.buffer.push_memory(state,action,r,ns)
            self.learnanapoch()
            if step % 16 == 0:
                self._softupdate()
            state = ns 

    def paramupdate(self):
        self.collect()
        for epoch in range(self.epoch):
            # self.learnaneps()
            self.epslearn()
            reward = self.policyvalidate()
            self.writer.add_scalar('reward',reward,self.rewardindex)
            self.rewardindex += 1

    def collect(self,learn = True):
        
        for _ in range(self.sampleepi):
            done = False
            state = self.env.reset()
            id = 0
            while done == False:
                action = self.action(state)
                ns,r,done,_ = self.env.step(action)
                self.buffer.push_memory(state,action,r,ns)
                if learn:
                    self.learnanapoch()
                    if id % 8 == 0:
                        self._softupdate()
                id += 1
                state = ns

    def _softupdate(self):
        for target,param in zip(self.targetactornet.parameters(),self.actornet.parameters()):
            target.data.copy_(
                (1 - self.tau) * target + self.tau * param
            )
        for target,param in zip(self.targetcriticnet.parameters(),self.criticnet.parameters()):
            target.data.copy_(
                (1 - self.tau) * target + self.tau * param
            )
    def policyvalidate(self):
        # threads = [self.pool.apply(p.valid,args=(self.validatetime,self.actornet)) for p in self.parallellist]
        # # return super().policyvaldate()
        # results = [p.get() for p in threads]
        # print("result is",results)
        # result = self.parallellist[0].valid(self.validatetime,self.actornet)
        # print("result is",result)
        reward = 0
        for _ in range(self.validatetime):
            done = False
            state = self.testenv.reset()
            while done == False:
                action = self.action(state,noise=False)
                ns,r,done,_ = self.testenv.step(action)
                # reward = self.actornet(
                reward += r
                # self.buffer.push_memory(state,action,r,ns)
                state = ns
        return reward/self.validatetime
    def learn(self):
        from tqdm import tqdm
        self.collect(learn=False)
        for epoch in tqdm(range(self.epoch)):
            self.collect()
            # for _ in range(4):
            # self.learnanapoch()
            # self.learnanapoch()
            # self._softupdate()
            # if epoch % 8 == 0:
            reward = self.policyvalidate()
                
            self.writer.add_scalar("reward",reward,epoch)
    #         pass
    
    def random(self):
        from tqdm import tqdm
        for epoch in tqdm(range(self.epoch//8)):
            reward = self.policyvalidate()
            self.writer.add_scalar("reward/baseline",reward,epoch)

    def learnanapoch(self):
        self.currentexp,self.actionexp,self.reward,self.ns = self.buffer.sample(self.sampletime)
        # print()
        '''
        things = self.buffer.sample(10)
        (10, 4)
        (10, 2)
        (10,)
        (10, 4)
        '''
        self.updateCritic()
        self.updateActor()
        self.losstriger += 1
        # self.sigma /= 2
        # self._softupdate()

        # for t in things:
        #     print(t.shape)
        # pass
    def updateCritic(self):
        self.optimizercritic.zero_grad()
        leftcurrent,rightcurrent,_ = self.criticnet(self.currentexp,self.actionexp)
        current_value = (leftcurrent + rightcurrent)/2
        current_value = current_value.squeeze()
        # print("shape of value",current_value.shape)
        with torch.no_grad():
            leftfuture,rightfuture,_ = self.targetcriticnet(self.ns,self.targetactornet(self.ns))
            expect_value = (1 - self.nabla) * leftfuture + self.nabla * rightfuture
            expect_value = self.gamma * expect_value.squeeze() + torch.from_numpy(self.reward).cuda().to(torch.float32)
        # print("shape of exp",expect_value.shape)
        # exit()
        loss = F.mse_loss(current_value,expect_value,reduction='sum')
        loss.backward()
        self.writer.add_scalar("reward/loss",loss,self.losstriger)
        self.optimizercritic.step()
    
    def updateActor(self):
        self.optimizeractor.zero_grad()
        leftcurrent,rightcurrent,_ = self.criticnet(self.currentexp,self.actornet(self.currentexp))
        current_value = (leftcurrent + rightcurrent)/2
        expectationreward = -torch.sum(current_value)
        expectationreward.backward()
        self.writer.add_scalar('reward/value',expectationreward,self.losstriger)
        self.optimizeractor.step()
