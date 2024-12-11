import torch
import torch.nn as nn
import torch.optim as optim
import math

xs = torch.linspace(0, 2*math.pi, 20)
ys = torch.sin(xs)
xs = xs.unsqueeze(1)
ys = ys.unsqueeze(1)

model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
    pred = model(xs) 
    loss = criterion(pred, ys)
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss: {loss.item()}")
