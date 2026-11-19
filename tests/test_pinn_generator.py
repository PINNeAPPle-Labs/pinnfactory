"""Tests for the symbolic-to-loss generator.

The package is on PyPI, so the point of these is that a published version has something
checking it: that a factory builds a loss from real SymPy strings, that the loss is a
genuine differentiable function of the network's parameters, and that the data term is the
mean squared error it claims to be rather than something that merely looks like one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinn_generator import NeuralNetwork, PINN, PINNFactory  # noqa: E402


def _heat_factory(**kwargs) -> PINNFactory:
    return PINNFactory(
        pde_residuals=["Derivative(u(t, x), t) - 0.1 * Derivative(u(t, x), x, 2)"],
        conditions=[{"equation": "u(t, x) - 0"}],
        independent_vars=["t", "x"],
        dependent_vars=["u"],
        **kwargs,
    )


def _model(seed: int = 0, **kwargs) -> PINN:
    torch.manual_seed(seed)
    return PINN(NeuralNetwork(num_inputs=2, num_outputs=1, num_layers=2, num_neurons=8), **kwargs)


def test_a_factory_builds_a_loss_and_reports_its_components():
    loss_fn = _heat_factory().generate_loss_function()
    t = torch.rand(16, 1)
    x = torch.rand(16, 1)

    total, components = loss_fn(_model(), {"collocation": (t, x), "conditions": [(t, x)]})

    assert torch.is_tensor(total)
    assert torch.isfinite(total)
    assert set(components) >= {"pde", "conditions", "total"}
    assert components["pde"] >= 0 and components["conditions"] >= 0
    assert total.item() == pytest.approx(components["total"])


def test_the_loss_is_differentiable_with_respect_to_the_network():
    loss_fn = _heat_factory().generate_loss_function()
    model = _model()
    t, x = torch.rand(32, 1), torch.rand(32, 1)

    total, _ = loss_fn(model, {"collocation": (t, x)})
    total.backward()

    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients, "the network has no trainable parameters"
    assert any(g is not None and torch.any(g != 0) for g in gradients), \
        "the loss produced no gradient anywhere, so it is not training anything"


def test_the_data_term_is_the_mean_squared_error():
    """Checked against a hand computation, not against the value the code happens to return."""
    loss_fn = PINNFactory(
        pde_residuals=[], conditions=[], independent_vars=["t", "x"], dependent_vars=["u"],
    ).generate_loss_function()
    model = _model()
    t, x = torch.rand(20, 1), torch.rand(20, 1)
    truth = torch.rand(20, 1)

    total, components = loss_fn(model, {"data": ((t, x), truth)})

    with torch.no_grad():
        expected = torch.mean((model(t, x) - truth) ** 2)
    assert components["data"] == pytest.approx(expected.item(), rel=1e-6)
    assert total.item() == pytest.approx(expected.item(), rel=1e-6)


def test_inverse_parameters_are_registered_and_receive_a_gradient():
    """An inverse problem is only solvable if the unknown is a parameter the loss touches."""
    model = _model(inverse_params_names=["alpha"], initial_guesses={"alpha": 0.3})
    assert "alpha" in model.inverse_params
    assert model.inverse_params["alpha"].item() == pytest.approx(0.3)

    factory = PINNFactory(
        pde_residuals=["Derivative(u(t, x), t) - alpha * Derivative(u(t, x), x, 2)"],
        conditions=[{"equation": "u(t, x) - 0"}],
        independent_vars=["t", "x"],
        dependent_vars=["u"],
        inverse_params=["alpha"],
    )
    t, x = torch.rand(16, 1), torch.rand(16, 1)

    total, _ = factory.generate_loss_function()(model, {"collocation": (t, x)})
    total.backward()

    gradient = model.inverse_params["alpha"].grad
    assert gradient is not None, "the inverse parameter never entered the loss"


def test_a_loss_weight_scales_only_its_own_term():
    maker = lambda w: PINNFactory(  # noqa: E731
        pde_residuals=[], conditions=[], independent_vars=["t", "x"], dependent_vars=["u"],
        loss_weights={"data": w},
    ).generate_loss_function()
    model = _model()
    t, x, truth = torch.rand(8, 1), torch.rand(8, 1), torch.rand(8, 1)

    plain, _ = maker(1.0)(model, {"data": ((t, x), truth)})
    doubled, _ = maker(2.0)(model, {"data": ((t, x), truth)})

    assert doubled.item() == pytest.approx(2 * plain.item(), rel=1e-6)
