from model.ActionVariationAutoEncoder import ActionVAE
from replaybuffer.static_dataset import Maze2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from tqdm import tqdm
class TrainCVAE(object):
    def __init__(self) -> None:
        self.dataset = Maze2d()
        self.dataloader = DataLoader(self.dataset,batch_size=64)
        self.EPOCH = 8
        self.CVAE = torch.load('model/CVAEpretrain').cuda()
        # self.CVAE = ActionVAE(state_dim=self.dataset.env.observation_space.shape[0],action_dim=self.dataset.env.action_space.shape[0],latent_dim=3).cuda()
        self.optim = torch.optim.Adam(self.CVAE.parameters(),lr = 0.0001)
        self.writer = SummaryWriter("./logs/trainVAE")
        self.index = 0
        self.validateindex = 0
    
    

    def validation(self):
        # for epoch in range(self.EPOCH):
        for states,actions,_,_,_ in tqdm(self.dataloader):
            self.states = states.cuda()
            self.actions = actions.cuda()
            pred_actions = self.CVAE.deduction(self.states)
            loss = F.mse_loss(pred_actions,self.actions)
            self.writer.add_scalar("validation",loss,self.validateindex)
            self.validateindex += 1

        pass
    def trainanepoch(self):
        recon_actions,mu,logvar = self.CVAE(self.states,self.actions)
        actionloss = F.mse_loss(recon_actions,self.actions,reduction='sum')
        Kldiv = -0.5 * torch.sum(1 + logvar - torch.pow(mu,2) - torch.exp(logvar))
        # print("Recon_actions",recon_actions.shape)
        # print("origin actions",self.actions.shape)
        self.optim.zero_grad()
        loss = actionloss + Kldiv
        self.writer.add_scalar("actionloss",actionloss,self.index)
        self.writer.add_scalar("Totalloss",loss,self.index)
        self.index += 1
        loss.backward()
        self.optim.step()

    def train(self):
        for epoch in range(self.EPOCH):
            for states,actions,_,_,_ in tqdm(self.dataloader):
                self.states = states.cuda()
                self.actions = actions.cuda()
                self.trainanepoch()

    