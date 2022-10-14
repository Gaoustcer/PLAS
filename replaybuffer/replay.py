from collections import namedtuple
from random import choices
import numpy as np
import gym
import d4rl

class buffer(object):
    def __init__(self,state_dim,action_dim,max_size = 1024) -> None:
        self.TRANSITION = namedtuple('replaybuffer','current_state action reward next_state')
        self.memory = self.TRANSITION(np.zeros((max_size,state_dim)),np.zeros((max_size,action_dim)),np.zeros(max_size),np.zeros((max_size,state_dim)))
        self.index = 0
        self.maxsize = max_size
        # self.size = 0
        self.size = 0
        self.full = False
    def _nextindex(self):
        self.index += 1
        self.index %= self.maxsize
    
    def push_memory(self,currentstate,action,reward,nextstate):
        # if self.size < self.maxsize:
        # self.memory.current_state.append(currentstate)
        # self.memory.action.append(action)
        # self.memory.reward.append(reward)
        # self.memory.next_state.append(nextstate)
        # # self.size += 1
        # self._nextindex()
        # else:
        self.memory.current_state[self.index] = currentstate
        self.memory.action[self.index] = action
        self.memory.reward[self.index] = reward
        self.memory.next_state[self.index] = nextstate
        self._nextindex()
        self.size += 1
        #     self.full = True

    def sample(self,n):
        assert n <= self.size
        # index_list = choices(range(),k=n)
        index = np.random.choice(min(self.size,self.maxsize),n)

        # current_state = []
        # action = []
        # next_state = []
        # reward = []
        # for index in index_list:
        #     current_state.append(self.memory.current_state[index])
        #     action.append(self.memory.action[index])
        #     next_state.append(self.memory.next_state[index])
        #     reward.append(self.memory.reward[index])
        # return np.array(current_state),np.array(action),np.array(reward).astype(float),np.array(next_state)
        return self.memory.current_state[index],self.memory.action[index],self.memory.reward[index],self.memory.next_state[index]

class parallel(object):
    def __init__(self,envname) -> None:
        # super().__init__()
        self.env = gym.make(envname)
    
    def parallelcollect(self,n:int,actionfunc):
        # env = gym.make(envname)
        '''
        n is the number of transition it wants to collect
        p is the policy
        '''
        observationlist = []
        rewardlist = []
        actionlist = []
        nextobservationlist = []
        donelist = []
        count = 0
        while True:
            state = self.env.reset()
            done = False
            while done == False:
                a = actionfunc(state)
                ns,reward,done,_ = self.env.step(a)
                observationlist.append(state)
                rewardlist.append(reward)
                donelist.append(done)
                actionlist.append(a)
                nextobservationlist.append(ns)
                state = ns
                count += 1
                if count > n:
                    return np.vstack(observationlist),np.vstack(actionlist),np.vstack(rewardlist),np.vstack(donelist),np.vstack(nextobservationlist)

    def valid(self,num_epsil,net):
        reward = 0
        for _ in range(num_epsil):
            state = self.env.reset()
            done = False
            while done == False:
                action = net(state).cpu().detach().numpy()
                ns,r,done,_ = self.env.step(action)
                state = ns
                reward += r
        return reward/num_epsil

if __name__ == "__main__":
    memorybuffer = buffer()
    import gym
    import d4rl
    env = gym.make('maze2d-umaze-v1')
    done = False
    while True:
        cs = env.reset()
        done = False
        while done == False:
            a = env.action_space.sample()
            ns,r,done,_ = env.step(a)
            memorybuffer.push_memory(cs,a,r,ns)
        if memorybuffer.full == True:
            break
    current_state,action,reward,next_state = memorybuffer.sample(64)
    import torch
    current_state = torch.from_numpy(current_state)
    actions = torch.from_numpy(action)
    # print(current_state)
    # print(actions.shape)
    # print(actions)
    exit()
    M = 128
    N = 64
    from time import time
    start = time()
    for _ in range(M):
        memorybuffer.sample(N)
    end = time()
    print("Time cost is",end - start)
    

    