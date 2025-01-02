import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.layers.ModifiedChebyLayer import ModifiedChebyLayer


@pytest.fixture
def rng():
    return nnx.Rngs(42)


@pytest.fixture
def x(rng):
    return jax.random.uniform(rng.params(), shape=(10, 2))  # Batch size of 10, n_in=2


@pytest.fixture
def modcheby_params():
    return {
        "n_in": 2,
        "n_out": 5,
        "k": 5
    }


@pytest.fixture
def modcheby_layer(rng, modcheby_params):
    return ModifiedChebyLayer(**modcheby_params, rngs=rng)


# Tests
def test_modcheby_layer_initialization(modcheby_layer, modcheby_params):
    
    assert modcheby_layer.n_in == modcheby_params["n_in"], "ModifiedChebyLayer n_in not set correctly"
    assert modcheby_layer.n_out == modcheby_params["n_out"], "ModifiedChebyLayer n_out not set correctly"
    assert modcheby_layer.c_basis.value.shape == (
        modcheby_params["n_out"],
        modcheby_params["n_in"],
        modcheby_params["k"] + 1
    ), "ModifiedChebyLayer c_basis shape incorrect"
    assert modcheby_layer.c_act.value.shape == (
        modcheby_params["n_out"],
        modcheby_params["n_in"]
    ), "ModifiedChebyLayer c_act shape incorrect"


def test_modcheby_layer_forward_pass(modcheby_layer, x, modcheby_params):
    
    y = modcheby_layer(x)
    batch = x.shape[0]

    assert y.shape == (batch, modcheby_params["n_out"]), "ModifiedChebyLayer forward pass returned incorrect shape"


def test_modcheby_layer_degree_update(modcheby_layer, x):
    
    k_new = 7
    
    initial_degree = modcheby_layer.k
    modcheby_layer.update_grid(x, k_new)
    updated_degree = modcheby_layer.k

    assert initial_degree != updated_degree, "ModifiedChebyLayer degree update failed"
    assert updated_degree == k_new, "ModifiedChebyLayer degree update did not set the new degree correctly"


@pytest.mark.parametrize("n_in, n_out, k", [
    (2, 5, 3),
    (1, 1, 1),
    (3, 10, 5),
])
def test_modcheby_layer_varied_configs(rng, n_in, n_out, k):
    
    x = jax.random.uniform(rng.params(), shape=(10, n_in))
    
    layer = ModifiedChebyLayer(n_in=n_in, n_out=n_out, k=k, rngs=rng)
    
    y = layer(x)
    
    assert y.shape == (x.shape[0], n_out), f"Forward pass returned incorrect shape for n_in={n_in}, n_out={n_out}, k={k}"
