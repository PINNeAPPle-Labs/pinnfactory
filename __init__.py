"""
pinnfactory — build Physics-Informed Neural Networks (PINNs) from symbolic
PDE definitions using SymPy, with automatic differentiation in PyTorch.

Quick start
-----------
>>> from pinnfactory import NeuralNetwork, PINN, PINNFactory
"""

from .pinn_generator import NeuralNetwork, PINN, PINNFactory

__all__ = ["NeuralNetwork", "PINN", "PINNFactory"]

__version__ = "0.1.0"
