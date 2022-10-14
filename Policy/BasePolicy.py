from replaybuffer.replay import parallel
import multiprocessing as mp


class BasePolicy(object):
    def __init__(self,envname:str,parallelnumber:int,EPOCH = 128) -> None:
        # pass
        self.epoch = EPOCH
        self.parallelnum = parallelnumber
        self.envname = envname
        # self.pool = mp.Pool(self.parallelnum)
        
        # self.parallellist = [parallel(envname) for _ in range(self.parallelnum)]
    
    def policyvalidate(self):
        '''
        validation of the process at the end of the iteration of the epoch
        '''
        raise NotImplementedError
    
    def action(self,state):
        '''
        make action from given state
        '''
        raise NotImplementedError
    
    def learn(self):
        raise NotImplementedError

    
    
    
