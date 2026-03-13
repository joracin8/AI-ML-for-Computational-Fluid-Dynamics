

# 1D Heat Diffusion using Physics-Informed Neural Networks (PINNs)

This project demonstrates how **Physics-Informed Neural Networks (PINNs)** can be used to solve the **1D heat diffusion equation**, a fundamental partial differential equation in heat transfer and computational physics.

Instead of using traditional numerical methods (Finite Difference / Finite Volume / Finite Element), a neural network is trained to learn the solution while enforcing the governing physics.

---

## Governing Equation

The 1D heat diffusion equation is:

du/dt = α * d²u/dx²

Where:

- u(x,t) → temperature
- x → spatial coordinate
- t → time
- α → thermal diffusivity

---

## Initial Condition

The initial temperature distribution is:

u(x,0) = sin(πx)

---

## Boundary Conditions

Dirichlet boundary conditions are applied:

u(0,t) = 0  
u(1,t) = 0

---

## PINN Methodology

A neural network is trained to approximate the solution:

Input:
(x , t)

Output:
u(x,t)

The training minimizes three loss components.

### 1. PDE Residual Loss
Ensures the neural network satisfies the heat equation.

### 2. Initial Condition Loss
Enforces the initial temperature distribution.

### 3. Boundary Condition Loss
Ensures the solution satisfies the boundary conditions.

Total loss function:

Loss = Loss_PDE + Loss_IC + Loss_BC

Automatic differentiation is used to compute derivatives of the neural network output.

---

## Implementation

The implementation uses:

- Python
- PyTorch
- Automatic differentiation
- Fully connected neural networks
- Adam optimizer for training

The neural network learns the temperature field across the spatio-temporal domain.

---

## Results

After training, the predicted solution from the PINN model is compared with the **analytical solution** of the heat equation at different time steps.

The visualization shows:

- True analytical solution
- PINN predicted solution
- Multiple time snapshots

This comparison verifies that the neural network correctly learns the underlying physics.

---


<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/c393af23-58f8-4ff8-a63e-94ba6d687f27" />

                         @ 500 Epoch, the loss function is not converged yet 
