import torch

from model.ActionVariationAutoEncoder import ActionVAE
from model.Actor_Critic import Actor,Critic
from replaybuffer.static_dataset import Maze2d

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class LAPOAgent(object):
    def __init__(self,envname = "maze2d-large-v1",latent_dim = 4) -> None:
        self.Actor = Actor()