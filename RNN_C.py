import torch
import torchvision
import torch.utils.data as Data
from torch import nn
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = test_data.data[:2000] / 255
test_y = test_data.targets[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(loader):
        x = x.view(-1, 28, 28)
        out = rnn(x)
        loss = loss_func(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            out = rnn(test_x)
            pred = torch.max(out, dim=1)[1].numpy()
            accuracy = sum(pred == test_y.numpy()) / len(test_y)
            print('Epoch: ', epoch, '| step: ', step, '| loss: ', loss.item(), '| accuracy: ', accuracy)


# prediction = rnn(test_x[:10])
# pred = torch.max(prediction, dim=1)[1].numpy()
# real = test_y[:10]
# print('pred:', pred, 'real:', real)


