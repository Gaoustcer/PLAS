from CVAE.Condition_VAE import TrainCVAE
import torch

if __name__ == "__main__":
    CVAE = TrainCVAE()
    # CVAE.train()
    
    # torch.save(CVAE,"model/CVAEpretrain")
    CVAE.validation()