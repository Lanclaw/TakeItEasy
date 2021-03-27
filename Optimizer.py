import torch
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))


LR = 0.01
EPOCH = 12
BATCH_SIZE = 32


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_hidden = torch.nn.Linear(1, 10)
        self.predict = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.n_hidden(x))
        x = self.predict(x)
        return x

net_SGD = Net()
net_Momentum = Net()
net_Adam = Net()
net_CMSprop = Net()
nets = [net_SGD, net_Momentum, net_Adam, net_CMSprop]

optim_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
optim_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
optim_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optim_CMSprop = torch.optim.RMSprop(net_CMSprop.parameters(), lr=LR, alpha=0.9)
optims = [optim_SGD, optim_Momentum, optim_Adam, optim_CMSprop]

loss_func = torch.nn.MSELoss()

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)
loss_his = [[], [], [], []]


for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net, opt, loss_h in zip(nets, optims, loss_his):
            pred = net(b_x)
            loss = loss_func(pred, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_h.append(loss.data.item())

labels = ['SGD', 'Momentum', 'Adam', 'CMSprop']

for i, loss in enumerate(loss_his):
    plt.plot(loss, lw=0.7)
plt.legend(loc='best', labels = labels)
plt.xlabel('step')
plt.ylabel('loss')
plt.ylim(0, 0.2)
plt.show()

