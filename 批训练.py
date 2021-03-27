import torch
import torch.utils.data as Data

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)

BATCH_SIZE = 5
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader, 1):
        print('epoch:', epoch, ' | step:', step, '| batch_x:', batch_x, '| batch_y:', batch_y)


