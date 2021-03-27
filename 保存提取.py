import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.linspace(-2, 2, 100)
x = torch.unsqueeze(x, dim=1)
y = x.pow(2) + 0.5 * torch.rand(x.size())

x = Variable(x)
y = Variable(y)

net1 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

optimizer = torch.optim.SGD(net1.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

for t in range(100):
    pred = net1(x)
    loss = loss_func(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(net1, 'net1.pkl')
torch.save(net1.state_dict(), 'net1_params.pkl')

plt.subplot(131)
plt.scatter(x.data.numpy(), y.data.numpy(), s=0.2)
plt.plot(x.data.numpy(), pred.data.numpy(), c='red', lw=2)



def restore_net():
    net2 = torch.load('net1.pkl')
    pred = net2(x)
    plt.subplot(132)
    plt.scatter(x.data.numpy(), y.data.numpy(), s=0.2)
    plt.plot(x.data.numpy(), pred.data.numpy(), c='red', lw=2)



def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('net1_params.pkl'))
    pred = net3(x)
    plt.subplot(133)
    plt.scatter(x.data.numpy(), y.data.numpy(), s=0.2)
    plt.plot(x.data.numpy(), pred.data.numpy(), c='red', lw=2)
    plt.show()

restore_net()
restore_params()

