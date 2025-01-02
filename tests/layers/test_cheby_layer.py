import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.layers.ChebyLayer import ChebyLayer


@pytest.fixture
def rng():
    return nnx.Rngs(42)


@pytest.fixture
def x(rng):
    return jax.random.uniform(rng.params(), shape=(10, 2))  # Batch size of 10, n_in=2


@pytest.fixture
def cheby_params():
    return {
        "n_in": 2,
        "n_out": 5,
        "k": 5
    }


@pytest.fixture
def cheby_layer(rng, cheby_params):
    return ChebyLayer(**cheby_params, rngs=rng)


# Tests
def test_cheby_layer_initialization(cheby_layer, cheby_params):
    
    assert cheby_layer.n_in == cheby_params["n_in"], "ChebyLayer n_in not set correctly"
    assert cheby_layer.n_out == cheby_params["n_out"], "ChebyLayer n_out not set correctly"
    assert cheby_layer.c_basis.value.shape == (
        cheby_params["n_out"],
        cheby_params["n_in"],
        cheby_params["k"] + 1
    ), "ChebyLayer c_basis shape incorrect"
    assert cheby_layer.c_act.value.shape == (
        cheby_params["n_out"],
        cheby_params["n_in"]
    ), "ChebyLayer c_act shape incorrect"


def test_cheby_layer_forward_pass(cheby_layer, x, cheby_params):
    
    y = cheby_layer(x)
    batch = x.shape[0]

    assert y.shape == (batch, cheby_params["n_out"]), "ChebyLayer forward pass returned incorrect shape"


def test_cheby_layer_degree_update(cheby_layer, x):
    
    k_new = 7
    
    initial_degree = cheby_layer.k
    cheby_layer.update_grid(x, k_new)
    updated_degree = cheby_layer.k

    assert initial_degree != updated_degree, "ChebyLayer degree update failed"
    assert updated_degree == k_new, "ChebyLayer degree update did not set the new degree correctly"


@pytest.mark.parametrize("n_in, n_out, k", [
    (2, 5, 3),
    (1, 1, 1),
    (3, 10, 5),
])
def test_cheby_layer_varied_configs(rng, n_in, n_out, k):
    
    x = jax.random.uniform(rng.params(), shape=(10, n_in))
    
    layer = ChebyLayer(n_in=n_in, n_out=n_out, k=k, rngs=rng)
    
    y = layer(x)
    
    assert y.shape == (x.shape[0], n_out), f"Forward pass returned incorrect shape for n_in={n_in}, n_out={n_out}, k={k}"
