import d4rl
import gym
from torch.utils.data import Dataset
import numpy as np
class Maze2d(Dataset):
    def __init__(self,envname,load_from_file = False) -> None:
        super(Maze2d,self).__init__()
        if load_from_file == False:
            self.env = gym.make(envname)
            self.dataset = d4rl.qlearning_dataset(self.env)
            self.len = len(self.dataset['actions'])
        else:
            self.dataset = np.load('singleexperience.npz')
            self.len = self.dataset['actions'].shape[0]

    def __len__(self):
        return self.len
        # return len(self.dataset['actions'])
    
    def __getitem__(self, index):
        # print(self.dataset.keys())
        return self.dataset['observations'][index],self.dataset['actions'][index],self.dataset['rewards'][index],self.dataset['next_observations'][index],self.dataset['terminals'][index]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    loader = DataLoader(Maze2d(),batch_size=32)
    for element in loader:
        for e in element:
            print(e.shape)
        exit()
        # return super().__getitem__(index)