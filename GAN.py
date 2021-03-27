import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


def artist_work():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + a - 1
    paintings = torch.FloatTensor(paintings)
    return paintings


G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS)
)

D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)

plt.ion()

for step in range(10000):
    real_works = artist_work()
    ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)
    fake_works = G(ideas)
    prob0 = D(fake_works)   # D wants to decrease prob0, so G want to increase prob0
    loss_G = torch.mean(torch.log(1 - prob0))

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    prob1 = D(real_works)   # D wants to increase prob1
    prob0 = D(fake_works.detach())
    loss_D = - torch.mean(torch.log(prob1) + torch.log(1 - prob0))
    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    if (step % 50 == 0):
        plt.cla()
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='r', lw=3, label='lower bound')
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='b', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], fake_works.data.numpy()[0], c='g', lw=3, label='Generated painting')
        plt.text(0.6, 0.5, 'loss_G: %.2f' % loss_G, fontdict={'size': 13})
        plt.text(0.6, 0.7, 'loss_D: %.2f' % loss_D, fontdict={'size': 13})
        plt.legend(loc='best')
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()
