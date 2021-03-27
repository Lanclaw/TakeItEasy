import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)  # class0 x data(tensor), shape=(100, 2)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), dim=0).type(torch.FloatTensor)  # FLoatTensor = 32-bit floating
y = torch.cat((y0, y1), dim=0).type(torch.LongTensor)   # LongTensor = 64-bit integer

x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(2, 10, 2)
print(net)

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
        prediction = torch.max(F.softmax(out), dim=1)[1]
        pred_y = prediction.data.numpy()
        target = y.data.numpy()
        plt.scatter(x[:, 0].data.numpy(), x[:, 1].data.numpy(), c=pred_y, s=20, lw=0)
        accuracy = sum(pred_y == target) / len(y)
        plt.text(0, 0, 'Accuracy = %.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.5)

plt.ioff()
plt.show()

