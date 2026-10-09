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

Without installing the package, the same lists are in `requirements.txt` (library) and `requirements-examples.txt`
(examples):

```bash
pip install -r requirements.txt -r requirements-examples.txt
```

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

**Research verified 2026-09-13**: two established lightweight PINN
libraries already solve pinnfactory's two most concrete open roadmap
items, and are worth using as reference API shapes rather than
designing either from a blank page.
[DeepXDE](https://github.com/lululxvi/deepxde) (verified by opening the
GitHub repo and its
[forward-PDE demo docs](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward.html))
ships six collocation-sampling strategies out of the box -- uniform,
pseudorandom, Latin hypercube, Halton, Hammersley, and Sobol sequences,
plus residual-based adaptive resampling during training -- across five
backends (TensorFlow 1/2, PyTorch, JAX, PaddlePaddle), and its own demo
gallery includes real systems of coupled equations (a simple ODE
system, Lotka-Volterra, and Kovasznay flow's coupled u/v/pressure
fields).
[NeuroDiffEq](https://github.com/NeuroDiffGym/neurodiffeq) (verified by
opening the GitHub repo; MIT-licensed, used by multiple research groups
including Harvard IACS) solves systems of coupled ODEs/PDEs with
multiple dependent variables today -- its own docs solve the two-variable
Lotka-Volterra system -- via `Generator1D`/`Generator2D`/`Generator3D`
classes that control the number, distribution, and bounding domain of
sampled points, including generator concatenation for composing more
complex domains.

Concretely, for this repo: `PINNFactory` already accepts a list of
`pde_residuals` and multiple `dependent_vars` sharing one network's
split output (see `pinn_generator.py`'s
`_compute_derivatives_and_values`), which is a real starting point, not
a blank slate -- extending that to fully-coupled systems (residuals that
reference more than one dependent variable's derivatives jointly, the
actual open item) is the natural next step rather than a rearchitecture.
For collocation sampling, the Quick Start example above builds its
training points with a bare `torch.linspace(...).requires_grad_(True)`
call; a small `pinnfactory.sampling` module offering
uniform/Latin-hypercube/Sobol generators -- modeled on NeuroDiffEq's
`GeneratorND` classes or backed directly by `scipy.stats.qmc` (already a
SciPy-stack dependency, no new hard dependency needed) -- would satisfy
this roadmap item with a small, self-contained addition.

---

## License
Apache License 2.0.  
You may use, modify, and distribute this project under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
