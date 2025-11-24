import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.models.KAN import KAN


@pytest.fixture
def seed():
    return 42

@pytest.fixture
def layer_dims():
    return [2, 3, 1]  # Example: 2 input nodes, 3 hidden nodes, 1 output node

@pytest.fixture
def req_params():
    return {'k': 3, 'G': 5, 'grid_range': (-1, 1), 'grid_e': 0.05}

@pytest.fixture
def sample_input(seed):
    key = jax.random.key(seed)
    return jax.random.uniform(key, shape=(10, 2))


# Test: Initialization
def test_kan_initialization(seed, layer_dims, req_params):
    
    model = KAN(layer_dims=layer_dims, layer_type="base", required_parameters=req_params, seed=seed)

    assert len(model.layers) == len(layer_dims) - 1, "Incorrect number of layers initialized"


def test_kan_invalid_layer_type(seed, layer_dims, req_params):
    
    with pytest.raises(ValueError, match="Unknown layer type"):
        KAN(layer_dims=layer_dims, layer_type="unknown", required_parameters=req_params, seed=seed)


# Test: Forward Pass
def test_kan_forward_pass(seed, layer_dims, req_params, sample_input):
    
    model = KAN(layer_dims=layer_dims, layer_type="base", required_parameters=req_params, seed=seed)
    output = model(sample_input)

    assert output.shape == (10, 1), "Forward pass output shape is incorrect"


def test_kan_forward_pass_minimal_case(seed, req_params):
    
    model = KAN(layer_dims=[1, 1], layer_type="base", required_parameters=req_params, seed=seed)
    minimal_input = jnp.array([[0.5]])
    output = model(minimal_input)

    assert output.shape == (1, 1), "Forward pass for minimal case returned incorrect shape"


# Test: Grid Updates
def test_kan_update_grids(seed, layer_dims, req_params, sample_input):
    
    model = KAN(layer_dims=layer_dims, layer_type="base", required_parameters=req_params, seed=seed)
    initial_grids = [layer.grid.item for layer in model.layers]

    model.update_grids(sample_input, G_new=8)

    updated_grids = [layer.grid.item for layer in model.layers]
    
    assert all(
        ig.shape != ug.shape for ig, ug in zip(initial_grids, updated_grids)
    ), "Grid update did not modify layer grids"


def test_kan_update_grids_edge_case(seed, req_params):
    
    model = KAN(layer_dims=[1, 1], layer_type="base", required_parameters=req_params, seed=seed)
    edge_case_input = jnp.array([[0.5]])
    model.update_grids(edge_case_input, G_new=10)

    assert True, "Grid update failed for edge case input"


# Test: Layer Properties
def test_kan_layer_properties(seed, layer_dims, req_params):
    
    model = KAN(layer_dims=layer_dims, layer_type="base", required_parameters=req_params, seed=seed)
    
    for layer in model.layers:
        assert hasattr(layer, 'grid'), "Layer grid property missing"
        assert hasattr(layer, 'c_basis'), "Layer basis coefficients missing"


# Test: End-to-End Integration
def test_kan_integration(seed, sample_input):
    
    layer_dims = [2, 5, 3, 1]  # More complex network
    reqd_params = {'k': 4, 'G': 6, 'grid_range': (-2, 2), 'grid_e': 0.1}
    model = KAN(layer_dims=layer_dims, layer_type="base", required_parameters=reqd_params, seed=seed)

    output = model(sample_input)
    
    assert output.shape == (10, 1), "End-to-end integration test failed"


# Test: Different Layer Types
def test_kan_spline_layer(seed):
    """Test KAN with spline layer type."""
    model = KAN([2, 4, 1], 'spline', {'k': 3, 'G': 5}, seed)
    x = jnp.array([[0.5, 0.5]])
    output = model(x)
    
    assert output.shape == (1, 1), "Spline KAN output shape incorrect"


def test_kan_chebyshev_layer(seed):
    """Test KAN with Chebyshev layer type."""
    model = KAN([2, 4, 1], 'chebyshev', {'D': 5}, seed)
    x = jnp.array([[0.5, 0.5]])
    output = model(x)
    
    assert output.shape == (1, 1), "Chebyshev KAN output shape incorrect"


def test_kan_fourier_layer(seed):
    """Test KAN with Fourier layer type."""
    model = KAN([2, 4, 1], 'fourier', {'D': 4}, seed)
    x = jnp.array([[0.5, 0.5]])
    output = model(x)
    
    assert output.shape == (1, 1), "Fourier KAN output shape incorrect"


def test_kan_legendre_layer(seed):
    """Test KAN with Legendre layer type."""
    model = KAN([2, 4, 1], 'legendre', {'D': 5}, seed)
    x = jnp.array([[0.5, 0.5]])
    output = model(x)
    
    assert output.shape == (1, 1), "Legendre KAN output shape incorrect"


def test_kan_rbf_layer(seed):
    """Test KAN with RBF layer type."""
    model = KAN([2, 4, 1], 'rbf', {'D': 5, 'kernel': {'type': 'gaussian'}}, seed)
    x = jnp.array([[0.5, 0.5]])
    output = model(x)
    
    assert output.shape == (1, 1), "RBF KAN output shape incorrect"


def test_kan_sine_layer(seed):
    """Test KAN with Sine layer type."""
    model = KAN([2, 4, 1], 'sine', {'D': 5}, seed)
    x = jnp.array([[0.5, 0.5]])
    output = model(x)
    
    assert output.shape == (1, 1), "Sine KAN output shape incorrect"


# Test: Output Validity
def test_kan_output_finite(seed, sample_input):
    """Test that KAN outputs are finite."""
    model = KAN([2, 8, 1], 'spline', {'k': 3, 'G': 5}, seed)
    output = model(sample_input)
    
    assert jnp.all(jnp.isfinite(output)), "KAN output contains non-finite values"


def test_kan_deterministic_output(seed):
    """Test that KAN produces deterministic outputs for same input."""
    model = KAN([2, 4, 1], 'spline', {'k': 3, 'G': 5}, seed)
    x = jnp.array([[0.5, 0.5]])
    
    output1 = model(x)
    output2 = model(x)
    
    assert jnp.allclose(output1, output2), "KAN should produce deterministic outputs"


# Test: Deep Networks
def test_kan_deep_network(seed):
    """Test KAN with many layers."""
    layer_dims = [2, 8, 8, 8, 8, 1]
    model = KAN(layer_dims, 'spline', {'k': 3, 'G': 5}, seed)
    x = jnp.array([[0.5, 0.5]])
    output = model(x)
    
    assert output.shape == (1, 1), "Deep KAN output shape incorrect"
    assert len(model.layers) == 5, "Deep KAN should have 5 layers"


# Test: Wide Networks
def test_kan_wide_network(seed):
    """Test KAN with wide hidden layers."""
    layer_dims = [3, 50, 50, 2]
    model = KAN(layer_dims, 'spline', {'k': 3, 'G': 5}, seed)
    x = jnp.array([[0.5, 0.5, 0.5]])
    output = model(x)
    
    assert output.shape == (1, 2), "Wide KAN output shape incorrect"


# Test: Batch Processing
def test_kan_batch_processing(seed):
    """Test KAN with various batch sizes."""
    model = KAN([2, 8, 1], 'spline', {'k': 3, 'G': 5}, seed)
    
    # Single sample
    x1 = jnp.array([[0.5, 0.5]])
    out1 = model(x1)
    assert out1.shape == (1, 1)
    
    # Small batch
    x10 = jax.random.uniform(jax.random.key(seed), (10, 2))
    out10 = model(x10)
    assert out10.shape == (10, 1)
    
    # Large batch
    x100 = jax.random.uniform(jax.random.key(seed), (100, 2))
    out100 = model(x100)
    assert out100.shape == (100, 1)
