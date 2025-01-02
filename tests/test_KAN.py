import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.KAN import KAN


@pytest.fixture
def rng():
    return nnx.Rngs(42)


@pytest.fixture
def layer_dims():
    return [2, 3, 1]  # Example: 2 input nodes, 3 hidden nodes, 1 output node


@pytest.fixture
def req_params():
    return {'k': 3, 'G': 3, 'grid_range': (-1, 1), 'grid_e': 0.05}


@pytest.fixture
def sample_input(rng):
    return jax.random.uniform(rng.params(), shape=(10, 2))


# Test: Initialization
def test_kan_initialization(rng, layer_dims, req_params):
    
    model = KAN(layer_dims=layer_dims, layer_type="base", required_parameters=req_params, rngs=rng)

    assert len(model.layers) == len(layer_dims) - 1, "Incorrect number of layers initialized"
    assert hasattr(model, 'biases'), "Biases not initialized"
    assert model.add_bias, "Default add_bias value should be True"


def test_kan_invalid_layer_type(rng, layer_dims, req_params):
    
    with pytest.raises(ValueError, match="Unknown layer type"):
        KAN(layer_dims=layer_dims, layer_type="unknown", required_parameters=req_params, rngs=rng)


# Test: Forward Pass
def test_kan_forward_pass(rng, layer_dims, req_params, sample_input):
    
    model = KAN(layer_dims=layer_dims, layer_type="base", required_parameters=req_params, rngs=rng)
    output = model(sample_input)

    assert output.shape == (10, 1), "Forward pass output shape is incorrect"


def test_kan_forward_pass_minimal_case(rng, req_params):
    
    model = KAN(layer_dims=[1, 1], layer_type="base", required_parameters=req_params, rngs=rng)
    minimal_input = jnp.array([[0.5]])
    output = model(minimal_input)

    assert output.shape == (1, 1), "Forward pass for minimal case returned incorrect shape"


# Test: Grid Updates
def test_kan_update_grids(rng, layer_dims, req_params, sample_input):
    
    model = KAN(layer_dims=layer_dims, layer_type="base", required_parameters=req_params, rngs=rng)
    initial_grids = [layer.grid.item for layer in model.layers]

    model.update_grids(sample_input, G_new=5)

    updated_grids = [layer.grid.item for layer in model.layers]
    
    assert all(
        ig.shape != ug.shape for ig, ug in zip(initial_grids, updated_grids)
    ), "Grid update did not modify layer grids"


def test_kan_update_grids_edge_case(rng, req_params):
    
    model = KAN(layer_dims=[1, 1], layer_type="base", required_parameters=req_params, rngs=rng)
    edge_case_input = jnp.array([[0.5]])
    model.update_grids(edge_case_input, G_new=10)

    assert True, "Grid update failed for edge case input"


# Test: Layer Properties
def test_kan_layer_properties(rng, layer_dims, req_params):
    
    model = KAN(layer_dims=layer_dims, layer_type="base", required_parameters=req_params, rngs=rng)
    
    for layer in model.layers:
        assert hasattr(layer, 'grid'), "Layer grid property missing"
        assert hasattr(layer, 'c_basis'), "Layer basis coefficients missing"


# Test: End-to-End Integration
def test_kan_integration(rng, sample_input):
    
    layer_dims = [2, 5, 3, 1]  # More complex network
    req_params = {'k': 4, 'G': 6, 'grid_range': (-2, 2), 'grid_e': 0.1}
    model = KAN(layer_dims=layer_dims, layer_type="base", required_parameters=req_params, rngs=rng)

    output = model(sample_input)
    
    assert output.shape == (10, 1), "End-to-end integration test failed"
