import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.layers.BaseLayer import BaseLayer


@pytest.fixture
def rng():
    return nnx.Rngs(42)


@pytest.fixture
def x(rng):
    return jax.random.uniform(rng.params(), shape=(10, 2))  # Batch size of 10, n_in=2


@pytest.fixture
def layer_params():
    return {
        "n_in": 2,
        "n_out": 5,
        "k": 4,
        "G": 3
    }


@pytest.fixture
def layer(rng, layer_params):
    return BaseLayer(**layer_params, rngs=rng)
    

# Tests
def test_base_layer_initialization(layer, layer_params):
    
    assert layer.n_in == layer_params["n_in"], "BaseLayer n_in not set correctly"
    assert layer.n_out == layer_params["n_out"], "BaseLayer n_out not set correctly"
    assert layer.c_basis.value.shape == (layer_params["n_in"] * layer_params["n_out"], layer_params["G"] + layer_params["k"]), "BaseLayer c_basis shape incorrect"
    assert layer.c_spl.value.shape == (layer_params["n_out"], layer_params["n_in"]), "BaseLayer c_spl shape incorrect"
    assert layer.c_res.value.shape == (layer_params["n_out"], layer_params["n_in"]), "BaseLayer c_res shape incorrect"


def test_base_layer_forward_pass(layer, x, layer_params):
    
    y = layer(x)
    batch = x.shape[0]

    assert y.shape == (batch, layer_params["n_out"]), "BaseLayer forward pass returned incorrect shape"


def test_base_layer_grid_update(layer, x, layer_params):
    
    G_new = 5
    
    initial_grid = layer.grid.item
    layer.update_grid(x, G_new)
    updated_grid = layer.grid.item

    assert initial_grid.shape != updated_grid.shape, "BaseLayer grid update failed to change grid shape"
    assert updated_grid.shape == (layer_params["n_in"] * layer_params["n_out"], G_new + 2 * layer_params["k"] + 1), "New BaseLayer grid shape incorrect"


@pytest.mark.parametrize("n_in, n_out, k, G", [
    (2, 5, 3, 3),
    (1, 1, 1, 1),
    (3, 10, 4, 5),
])
def test_base_layer_varied_configs(rng, n_in, n_out, k, G):
    
    x = jax.random.uniform(rng.params(), shape=(10, n_in))
    
    layer = BaseLayer(n_in=n_in, n_out=n_out, k=k, G=G, rngs=rng)
    
    y = layer(x)
    
    assert y.shape == (x.shape[0], n_out), f"Forward pass returned incorrect shape for n_in={n_in}, n_out={n_out}, k={k}, G={G}"
