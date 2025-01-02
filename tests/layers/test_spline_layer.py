import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.layers.SplineLayer import SplineLayer


@pytest.fixture
def rng():
    return nnx.Rngs(42)


@pytest.fixture
def x(rng):
    return jax.random.uniform(rng.params(), shape=(10, 2))  # Batch size of 10, n_in=2


@pytest.fixture
def spline_params():
    return {
        "n_in": 2,
        "n_out": 5,
        "k": 4,
        "G": 3
    }


@pytest.fixture
def spline_layer(rng, spline_params):
    return SplineLayer(**spline_params, rngs=rng)


# Tests
def test_spline_layer_initialization(spline_layer, spline_params):
    
    assert spline_layer.n_in == spline_params["n_in"], "SplineLayer n_in not set correctly"
    assert spline_layer.n_out == spline_params["n_out"], "SplineLayer n_out not set correctly"
    assert spline_layer.c_basis.value.shape == (
        spline_params["n_out"],
        spline_params["n_in"],
        spline_params["G"] + spline_params["k"]
    ), "SplineLayer c_basis shape incorrect"
    assert spline_layer.c_spl.value.shape == (
        spline_params["n_out"],
        spline_params["n_in"]
    ), "SplineLayer c_spl shape incorrect"
    assert spline_layer.c_res.value.shape == (
        spline_params["n_out"],
        spline_params["n_in"]
    ), "SplineLayer c_res shape incorrect"


def test_spline_layer_forward_pass(spline_layer, x, spline_params):
    
    y = spline_layer(x)
    batch = x.shape[0]

    assert y.shape == (batch, spline_params["n_out"]), "SplineLayer forward pass returned incorrect shape"


def test_spline_layer_grid_update(spline_layer, x):
    
    G_new = 5
    
    initial_grid = spline_layer.grid.item
    spline_layer.update_grid(x, G_new)
    updated_grid = spline_layer.grid.item

    assert initial_grid.shape != updated_grid.shape, "SplineLayer grid update failed to change grid shape"
    assert updated_grid.shape == (
        spline_layer.n_in,
        G_new + 2 * spline_layer.k + 1
    ), "New SplineLayer grid shape incorrect"


@pytest.mark.parametrize("n_in, n_out, k, G", [
    (2, 5, 3, 3),
    (1, 1, 1, 1),
    (3, 10, 4, 5),
])
def test_spline_layer_varied_configs(rng, n_in, n_out, k, G):
    
    x = jax.random.uniform(rng.params(), shape=(10, n_in))
    
    layer = SplineLayer(n_in=n_in, n_out=n_out, k=k, G=G, rngs=rng)
    
    y = layer(x)
    
    assert y.shape == (x.shape[0], n_out), f"Forward pass returned incorrect shape for n_in={n_in}, n_out={n_out}, k={k}, G={G}"
