import torch
import torch.nn as nn
import torch.optim as optim
import math

# xs = torch.linspace(0, 2*math.pi, 20)
# ys = torch.sin(xs)
# xs = xs.unsqueeze(1)
# ys = ys.unsqueeze(1)

xs = torch.arange(50, dtype=torch.float32) / 50.0    # shape [50]
ys = (2.0 * torch.arange(50, dtype=torch.float32)) / 50.0  # shape [50]

xs = xs.unsqueeze(1)  # shape now [50, 1]
ys = ys.unsqueeze(1) 

# model = nn.Sequential(
#     nn.Linear(1, 1),
#     nn.ReLU(),
#     nn.Linear(1, 10),
#     nn.ReLU(),
#     nn.Linear(1, 1)
# )

# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# for epoch in range(1000):
#     pred = model(xs) 
#     loss = criterion(pred, ys) # 1.94378e-07
#     optimizer.zero_grad() 
#     loss.backward()
#     optimizer.step()
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch} Loss: {loss.item()}")

model = nn.Sequential(
    nn.Linear(1, 1),
    nn.ReLU(),
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1) 
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    pred = model(xs) 
    loss = criterion(pred, ys)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss: {loss.item()}")
