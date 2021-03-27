import torch
import numpy as np

np_data = np.arange(6).reshape((2,3))
print(np_data)
torch_data = torch.from_numpy(np_data)  # numpy -> tensor
print(torch_data)
tensor2array = torch_data.numpy()  # tensor -> numpy
print(tensor2array)

# abs
data = [-1, -2, -3, 5]  # list
tensor = torch.FloatTensor(data)    # 32bit
print(tensor)
print(np.abs(data))  # data is list, 'list' object has no attribute 'abs'!
print(tensor.abs())
print(np.mean(data))
print(tensor.mean())

data = [[1, 2], [3, 4]]  # list has no 'abs , has no 'dot'

tensor = torch.FloatTensor(data)  # 32bit  可用于list (array)
data = np.array(data)
print(
    '\nnumpy', np.matmul(data, data),
    '\nnumpy', data.dot(data),
    '\ntorch', torch.mm(tensor, tensor)
)

data = [1, 2, 3, 4]
tensor = torch.FloatTensor(data)
print(tensor.dot(tensor))

from torch.autograd import Variable  #  神经网络里的参数都是Variable变量的形式

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad = True)

print(tensor)
print(variable)

t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)

print(t_out)
print(v_out)

v_out.backward()
print(variable.grad)
# v_out = 1/4 * sum(var * var)
# d_v_out / d_var = 1/4 * 2 * var = var / 2
print(variable.data)  # variable.data is tensor
print(variable.data.numpy())  # numpy() 只能用在tensor上

import torch.nn.functional as F  # activation function
from torch.autograd import variable
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200)  # tensor shape(200)
x = Variable(x)
x_np = x.data.numpy()  # tensor 无法被plt识别, 必须转成numpy

y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.5, 1.5))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.5, 1.5))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.5, 5))
plt.legend(loc='best')

plt.show()