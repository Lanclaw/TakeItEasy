import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable
import torch
from torch import nn
import torch.nn.functional as F
import time

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),   # 把每一个pixel的值压缩到（0， 1）,
    download=DOWNLOAD_MNIST                    # for step, (b_x, b_y) in enumerate(train_loader): 中才压缩，之前不压缩
)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    download=DOWNLOAD_MNIST
)

# print(train_data.data.size())
# print(train_data.targets.size())
# print(test_data.data.size())
# print(test_data.targets.size())
#
# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.title('%d' % train_data.targets[0])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# cuda
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000].cuda() / 255
test_y = test_data.targets[:2000].cuda()


class CNN(nn.Module):           # (1, 28, 28)
    def __init__(self):
        super(CNN, self).__init__()        # w = [(w + 2 * p_w) - k_w + s_w] / s_w  (floor)
        self.conv1 = nn.Sequential(  # h = [(h + 2 * p_h) - k_h + s_h] / s_h  (floor)
            nn.Conv2d(             # (16, 28, 28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2        # 左右两侧分别填充2列或者上下分别填充2行
            ),
            nn.ReLU(),          # (16, 28, 28)
            nn.MaxPool2d(kernel_size=2)     # 默认stride和kernel_size相同   (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),     # (32, 14, 14)
            nn.ReLU(),                      # (32, 14, 14)
            nn.MaxPool2d(2)                 # (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)   # (batch_size, 16, 14, 14)
        x = self.conv2(x)   # (batch_size, 32, 7, 7)
        x = x.view(x.size(0), -1)   # (batch_size, 32 * 7 * 7)
        x = self.out(x)
        return x


cnn = CNN().cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

start = time.time()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        # cuda
        b_x = b_x.cuda()
        b_y = b_y.cuda()

        out = cnn(b_x)
        loss = loss_func(out, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            pred = cnn(test_x)
            net_softmax = nn.Softmax(dim=1)
            pred = torch.max(net_softmax(pred), dim=1)[1]
            accuracy = sum(pred == test_y) / len(test_y)
            print('Epoch:', epoch, ' | step:', step, ' | loss:', loss.item(), ' | acc:', accuracy)

end = time.time()

pred = cnn(test_x[:10])
net_softmax = nn.Softmax(dim=1)
pred = torch.max(net_softmax(pred), dim=1)[1]
print('prediction number:', pred)
print('real number:', test_y[:10])
print('time:', end - start)



