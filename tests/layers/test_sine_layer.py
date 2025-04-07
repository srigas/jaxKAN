import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.layers.Sine import SineLayer


@pytest.fixture
def seed():
    return 42

@pytest.fixture
def x(seed):
    key = jax.random.key(seed)
    return jax.random.uniform(key, shape=(10, 2))

@pytest.fixture
def model_params():
    return {
        "n_in": 2,
        "n_out": 5,
        "D": 4,
        "external_weights": True,
        "add_bias": True
    }

@pytest.fixture
def sine_layer(seed, model_params):
    return SineLayer(**model_params, seed=seed)


# Tests
def test_sine_layer_initialization(sine_layer, model_params):
    
    assert sine_layer.n_in == model_params["n_in"], "SineLayer n_in not set correctly"
    assert sine_layer.n_out == model_params["n_out"], "SineLayer n_out not set correctly"
    assert sine_layer.c_basis.value.shape == (
        model_params["n_out"],
        model_params["n_in"],
        model_params["D"]
    ), "SineLayer c_basis shape incorrect"
    assert sine_layer.c_ext.value.shape == (
        model_params["n_out"],
        model_params["n_in"]
    ), "SineLayer c_ext shape incorrect"


def test_sine_layer_forward_pass(sine_layer, x, model_params):
    
    y = sine_layer(x)
    batch = x.shape[0]

    assert y.shape == (batch, model_params["n_out"]), "SineLayer forward pass returned incorrect shape"


def test_sine_layer_degree_update(sine_layer, x):
    
    D_new = 8
        
    initial_degree = sine_layer.D
    sine_layer.update_grid(x, D_new)
    updated_degree = sine_layer.D

    assert initial_degree != updated_degree, "SineLayer degree update failed"
    assert updated_degree == D_new, "SineLayer degree update did not set the new degree correctly"


@pytest.mark.parametrize("n_in, n_out, D", [
    (2, 5, 3),
    (1, 1, 1),
    (3, 10, 5),
])
def test_sine_layer_varied_configs(seed, n_in, n_out, D):

    key = jax.random.key(seed)
    
    x = jax.random.uniform(key, shape=(10, n_in))
    
    layer = SineLayer(n_in=n_in, n_out=n_out, D=D, seed=seed)
    
    y = layer(x)
    
    assert y.shape == (x.shape[0], n_out), f"Forward pass of SineLayer returned incorrect shape for n_in={n_in}, n_out={n_out}, D={D}"
