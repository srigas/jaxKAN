import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.layers.FourierLayer import FourierLayer


@pytest.fixture
def rng():
    return nnx.Rngs(42)


@pytest.fixture
def x(rng):
    return jax.random.uniform(rng.params(), shape=(10, 2))  # Batch size of 10, n_in=2


@pytest.fixture
def fourier_params():
    return {
        "n_in": 2,
        "n_out": 5,
        "k": 8,
        "smooth_init": True
    }


@pytest.fixture
def fourier_layer(rng, fourier_params):
    return FourierLayer(**fourier_params, rngs=rng)


# Tests
def test_fourier_layer_initialization(fourier_layer, fourier_params):
    
    assert fourier_layer.n_in == fourier_params["n_in"], "FourierLayer n_in not set correctly"
    assert fourier_layer.n_out == fourier_params["n_out"], "FourierLayer n_out not set correctly"
    assert fourier_layer.c_cos.value.shape == (
        fourier_params["n_out"],
        fourier_params["n_in"],
        fourier_params["k"]
    ), "FourierLayer c_cos shape incorrect"
    assert fourier_layer.c_sin.value.shape == (
        fourier_params["n_out"],
        fourier_params["n_in"],
        fourier_params["k"]
    ), "FourierLayer c_sin shape incorrect"


def test_fourier_layer_forward_pass(rng, x, fourier_params):
    
    fourier_params["smooth_init"] = False
    layer = FourierLayer(**fourier_params, rngs=rng)
    y = layer(x)
    batch = x.shape[0]

    assert y.shape == (batch, fourier_params["n_out"]), "FourierLayer forward pass returned incorrect shape"


def test_fourier_layer_order_update(fourier_layer, x):
    
    k_new = 10
    
    initial_order = fourier_layer.k
    fourier_layer.update_grid(x, k_new)
    updated_order = fourier_layer.k

    assert initial_order != updated_order, "FourierLayer order update failed"
    assert updated_order == k_new, "FourierLayer order update did not set the new order correctly"


@pytest.mark.parametrize("n_in, n_out, k", [
    (2, 5, 4),
    (1, 1, 1),
    (3, 10, 6),
])
def test_fourier_layer_varied_configs(rng, n_in, n_out, k):
    
    x = jax.random.uniform(rng.params(), shape=(10, n_in))
    
    layer = FourierLayer(n_in=n_in, n_out=n_out, k=k, smooth_init=True, rngs=rng)
    
    y = layer(x)
    
    assert y.shape == (x.shape[0], n_out), f"Forward pass returned incorrect shape for n_in={n_in}, n_out={n_out}, k={k}"
