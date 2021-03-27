import torch
from torch import nn
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

EPOCH = 0
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

plt.ion()
fig, ax = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
view_data = train_data.data[:N_TEST_IMG].view(-1, 28 * 28) / 255
for i in range(N_TEST_IMG):
    ax[0][i].imshow(np.reshape(view_data.numpy()[i], (28, 28)), cmap='gray')
    ax[0][i].set_xticks(())
    ax[0][i].set_yticks(())


for epoch in range(EPOCH):
    # break
    for step, (x, y) in enumerate(train_loader):
        b_x = x.view(-1, 28 * 28)
        b_y = x.view(-1, 28 * 28)

        encoded, decoded = autoencoder(b_x)
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch: ', epoch, '| step: ', step, ' | loss: ', loss.item())
            _, pred = autoencoder(view_data)
            # break
            for i in range(N_TEST_IMG):
                ax[1][i].clear()
                ax[1][i].imshow(np.reshape(pred.detach().numpy()[i], (28, 28)), cmap='gray')
                ax[1][i].set_xticks(())
                ax[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.01)


plt.ioff()
plt.show()


# visualize in 3D plot
view_data = train_data.data[:200].view(-1, 28*28).type(torch.FloatTensor) / 255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.targets[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))
    # print(int(255*s/9))
    # print(s)
    # print(c)
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()

