import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)   # r_out:(B_s, t_s, h_s), h_state:(num_layers, b_s, h_s)
        out = []
        for time_step in range(TIME_STEP):
            out.append(self.out(r_out[:, time_step, :]))    # one of out(list) : (b_s, output_size) out: (time_step, b_s, output_size)
        return torch.stack(out, dim=1), h_state     # torch.stack(out, dim=1): (b_s, time_step, output_size)


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None
plt.ion()

for step in range(100):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)

    x_np = np.cos(steps)
    y_np = np.sin(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    pred, h_state = rnn(x, h_state)
    h_state = h_state.data      # important!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! to drop grad
    loss = loss_func(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Step: ', step, '| loss: ', loss)
    plt.plot(steps, pred.detach().numpy().flatten(), 'r-')
    plt.plot(steps, y_np, 'b-')
    plt.draw()
    plt.pause(0.2)


plt.ioff()
plt.show()
