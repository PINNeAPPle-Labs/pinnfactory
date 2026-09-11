# PINNeAPPle - PINNFactory

A lightweight framework for building **Physics-Informed Neural Networks (PINNs)** with symbolic PDE definitions using **SymPy** and automatic differentiation in **PyTorch**.  

It provides:
- Flexible neural architectures (`NeuralNetwork` class).  
- Wrapper for inverse parameter estimation (`PINN`).  
- Factory for PDE-driven loss generation from symbolic equations (`PINNFactory`).  

---

## Installation

```bash
pip install pinnfactory
```

Or from source:

```bash
git clone https://github.com/PINNeAPPle-Labs/pinnfactory.git
cd pinnfactory
pip install -e ".[examples]"
```

PyPI: https://pypi.org/project/pinnfactory/

### Requirements
- torch
- sympy
- matplotlib (only for `examples/`)
- numpy (only for `examples/`)

---

## Quick Start

### 1. Define your neural network
```python
from pinnfactory import NeuralNetwork, PINN

# Example: 1 input, 1 output, 3 hidden layers with 20 neurons each
net = NeuralNetwork(num_inputs=1, num_outputs=1, num_layers=3, num_neurons=20)
pinn = PINN(net)
```

### 2. Define PDEs and conditions symbolically
```python
from pinnfactory import PINNFactory

# PDE: u_xx + u = 0  (example)
pde_residuals = ["Derivative(u(x), (x,2)) + u(x)"]

# Boundary conditions: u(0) = 0, u(pi) = 0
conditions = [
    {"equation": "u(0)"},
    {"equation": "u(pi)"}
]

factory = PINNFactory(
    pde_residuals=pde_residuals,
    conditions=conditions,
    independent_vars=["x"],
    dependent_vars=["u"]
)

loss_fn = factory.generate_loss_function()
```

### 3. Training loop
```python
import torch

optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

# Example training batch
x = torch.linspace(0, 3.14, 100).view(-1, 1).requires_grad_(True)

for epoch in range(1000):
    optimizer.zero_grad()
    loss, loss_components = loss_fn(pinn, {"collocation": (x,)})
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Total Loss: {loss.item():.6f}")
```

---

## Roadmap
- [ ] Add support for system of PDEs.  
- [ ] Implement collocation sampling utilities.  
- [ ] Add GPU support for large-scale PDE solving.  
- [ ] Integrate with visualization tools for PINN training.
- [ ] Add more NN Architectures.

---

## License
Apache License 2.0.  
You may use, modify, and distribute this project under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
