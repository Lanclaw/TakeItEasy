import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x0 = torch.normal(1, 1, (100, 2))
y0 = torch.zeros(100)
x1 = torch.normal(-1, 1, (100, 2))
y1 = torch.ones(100)
x = torch.cat((x0, x1), dim=0).type(torch.FloatTensor)
y = torch.cat((y0, y1), dim=0).type(torch.LongTensor)

x = Variable(x)
y = Variable(y)

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x
# net = Net(2, 10, 2)
net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)



optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()


plt.ion()
for t in range(100):
    out = net(x)

    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        target = y.data.numpy()
        pred = torch.max(F.softmax(out), dim=1)[1].data.numpy()
        accuracy = sum(pred == target) / len(target)
        plt.scatter(x[:, 0].data.numpy(), x[:, 1].data.numpy(), c=pred, s=50, lw=0)
        plt.text(0, 0, 'accuracy: %.4f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.5)

plt.ioff()
plt.show()


