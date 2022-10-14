from OnlineAgent.onlineagent import OnlineAgent
import torch
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    Online = OnlineAgent(envname='maze2d-umaze-v1',parallelnumber=4)
    # Online.policyvalidate()
    Online.learn()