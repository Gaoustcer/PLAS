import torch
from model.ActionVariationAutoEncoder import ActionVAE
from model.Actor_Critic import Actor
VAE = ActionVAE(state_dim=4,action_dim=2,latent_dim=3).cuda()
states = torch.rand(4)
latent_action = Actor(state_dim=4,latent_action_dim=3).cuda()(states.cuda())
print("latent_action is",latent_action.shape)
print(VAE.deduction(states,latent_action))