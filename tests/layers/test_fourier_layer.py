import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.layers.Fourier import FourierLayer


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
        "D": 8,
        "smooth_init": True
    }

@pytest.fixture
def fourier_layer(seed, model_params):
    return FourierLayer(**model_params, seed=seed)


# Tests
def test_fourier_layer_initialization(fourier_layer, model_params):
    
    assert fourier_layer.n_in == model_params["n_in"], "FourierLayer n_in not set correctly"
    assert fourier_layer.n_out == model_params["n_out"], "FourierLayer n_out not set correctly"
    assert fourier_layer.c_cos[...].shape == (
        model_params["n_out"],
        model_params["n_in"],
        model_params["D"]
    ), "FourierLayer c_cos shape incorrect"
    assert fourier_layer.c_sin[...].shape == (
        model_params["n_out"],
        model_params["n_in"],
        model_params["D"]
    ), "FourierLayer c_sin shape incorrect"


def test_fourier_layer_forward_pass(seed, x, model_params):
    
    model_params["smooth_init"] = False
    layer = FourierLayer(**model_params, seed=seed)
    y = layer(x)
    batch = x.shape[0]

    assert y.shape == (batch, model_params["n_out"]), "FourierLayer forward pass returned incorrect shape"


def test_fourier_layer_order_update(fourier_layer, x):
    
    D_new = 10
    
    initial_order = fourier_layer.D
    fourier_layer.update_grid(x, D_new)
    updated_order = fourier_layer.D

    assert initial_order != updated_order, "FourierLayer order update failed"
    assert updated_order == D_new, "FourierLayer order update did not set the new order correctly"


@pytest.mark.parametrize("n_in, n_out, D", [
    (2, 5, 4),
    (1, 1, 1),
    (3, 10, 6),
])
def test_fourier_layer_varied_configs(seed, n_in, n_out, D):

    key = jax.random.key(seed)
    
    x = jax.random.uniform(key, shape=(10, n_in))
    
    layer = FourierLayer(n_in=n_in, n_out=n_out, D=D, smooth_init=True, seed=seed)
    
    y = layer(x)
    
    assert y.shape == (x.shape[0], n_out), f"Forward pass returned incorrect shape for n_in={n_in}, n_out={n_out}, D={D}"
