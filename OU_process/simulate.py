import matplotlib.pyplot as plt

import numpy as np

N = 8
M = 1024 * 8
mu = 1
dt = 0.01
def U_Oprocess():
    xlist = []
    theta = np.random.rand()
    sigma = np.random.rand()
    x = np.random.rand()
    
    for _ in range(M):
        xlist.append(x)
        dx = theta * (mu - x) * dt + sigma *np.random.normal(0,sigma*dt)
        x = x + dx
    plt.scatter(x = range(M),y = xlist,c=np.random.random(3),s=0.01)

if __name__ == "__main__":
    for _ in range(N):
        U_Oprocess()
    plt.savefig("UO.png")
