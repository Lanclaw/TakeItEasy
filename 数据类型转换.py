import torch
import numpy as np

x = [[1, 2], [3, 4], [5, 6]]    # list
print('list:', x)
x = np.array(x)  # list -> numpy
print('numpy:', x)
x = x.tolist()  # numpy -> list
print('list:', x)

x = torch.Tensor(x)  # list -> tensor     or torch.FloatTensor(x)  | numpy -> tensor
print('tensor:', x)
x = x.numpy().tolist()  # tensor -> list
print('list:', x)

x = np.array(x)
x = torch.from_numpy(x)     # numpy -> tensor
print('tensor:', x)
x = x.numpy()               # tensor -> numpy
print('numpy:', x)

