import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50,50),
            nn.Tanh(),
            nn.Linear(50,50),
            nn.Tanh(),
            nn.Linear(50,1)
        )

    def forward(self,x,t):
        input = torch.cat([x,t],dim=1)
        return self.net(input)


model = PINN()

alpha = 0.01
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# collocation points
N = 1000
x = torch.rand(N,1,requires_grad=True)
t = torch.rand(N,1,requires_grad=True)

# training loop
for epoch in range(500):

    optimizer.zero_grad()

    u = model(x,t)

    # derivatives
    u_t = torch.autograd.grad(u,t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

    u_x = torch.autograd.grad(u,x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x,x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True)[0]

    # PDE residual
    f = u_t - alpha*u_xx

    loss_pde = torch.mean(f**2)

    # initial condition
    x_ic = torch.rand(N,1)
    t_ic = torch.zeros(N,1)
    u_ic = torch.sin(np.pi*x_ic)

    pred_ic = model(x_ic,t_ic)

    loss_ic = torch.mean((pred_ic-u_ic)**2)

    loss = loss_pde + loss_ic

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(epoch, loss.item())
import matplotlib.pyplot as plt
import numpy as np
import torch

alpha = 0.01

# analytical solution
def true_solution(x,t):
    return np.sin(np.pi*x)*np.exp(-alpha*np.pi**2*t)

x = np.linspace(0,1,100)
x_torch = torch.tensor(x,dtype=torch.float32).reshape(-1,1)

time_points = [0.0,0.25,0.5,0.75,1.0]

plt.figure(figsize=(8,6))

for t in time_points:

    t_torch = torch.ones_like(x_torch)*t

    with torch.no_grad():
        pred = model(x_torch,t_torch).numpy()

    true = true_solution(x,t)

    plt.plot(x,true,'--',label=f"True t={t}")
    plt.plot(x,pred,label=f"PINN t={t}")

plt.xlabel("x")
plt.ylabel("Temperature u(x,t)")
plt.title("1D Heat Equation: True vs PINN Prediction")
plt.legend()
plt.show()
        
