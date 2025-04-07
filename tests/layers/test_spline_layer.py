import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.layers.Spline import BaseLayer, SplineLayer

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
        "k": 4,
        "G": 5,
        "external_weights": True,
        "residual": nnx.silu
    }

@pytest.fixture
def base_layer(seed, model_params):
    return BaseLayer(**model_params, seed=seed)

@pytest.fixture
def spline_layer(seed, model_params):
    return SplineLayer(**model_params, seed=seed)


# Spline Layer
def test_spline_layer_initialization(spline_layer, model_params):
    
    assert spline_layer.n_in == model_params["n_in"], "SplineLayer n_in not set correctly"
    assert spline_layer.n_out == model_params["n_out"], "SplineLayer n_out not set correctly"
    assert spline_layer.c_basis.value.shape == (
        model_params["n_out"],
        model_params["n_in"],
        model_params["G"] + model_params["k"]
    ), "SplineLayer c_basis shape incorrect"
    assert spline_layer.c_spl.value.shape == (
        model_params["n_out"],
        model_params["n_in"]
    ), "SplineLayer c_spl shape incorrect"
    assert spline_layer.c_res.value.shape == (
        model_params["n_out"],
        model_params["n_in"]
    ), "SplineLayer c_res shape incorrect"


def test_spline_layer_forward_pass(spline_layer, x, model_params):
    
    y = spline_layer(x)
    batch = x.shape[0]

    assert y.shape == (batch, model_params["n_out"]), "SplineLayer forward pass returned incorrect shape"


def test_spline_layer_grid_update(spline_layer, x):
    
    G_new = 10
    
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
def test_spline_layer_varied_configs(seed, n_in, n_out, k, G):

    key = jax.random.key(seed)
    
    x = jax.random.uniform(key, shape=(10, n_in))
    
    layer = SplineLayer(n_in=n_in, n_out=n_out, k=k, G=G, seed=seed)
    
    y = layer(x)
    
    assert y.shape == (x.shape[0], n_out), f"Forward pass returned incorrect shape for n_in={n_in}, n_out={n_out}, k={k}, G={G}"


# Base Layer
def test_base_layer_initialization(base_layer, model_params):
    
    assert base_layer.n_in == model_params["n_in"], "BaseLayer n_in not set correctly"
    assert base_layer.n_out == model_params["n_out"], "BaseLayer n_out not set correctly"
    assert base_layer.c_basis.value.shape == (
        model_params["n_out"] * model_params["n_in"],
        model_params["G"] + model_params["k"]
    ), "BaseLayer c_basis shape incorrect"
    assert base_layer.c_spl.value.shape == (
        model_params["n_out"],
        model_params["n_in"]
    ), "BaseLayer c_spl shape incorrect"
    assert base_layer.c_res.value.shape == (
        model_params["n_out"],
        model_params["n_in"]
    ), "BaseLayer c_res shape incorrect"


def test_base_layer_forward_pass(base_layer, x, model_params):
    
    y = base_layer(x)
    batch = x.shape[0]

    assert y.shape == (batch, model_params["n_out"]), "BaseLayer forward pass returned incorrect shape"


def test_base_layer_grid_update(base_layer, x):
    
    G_new = 10
    
    initial_grid = base_layer.grid.item
    base_layer.update_grid(x, G_new)
    updated_grid = base_layer.grid.item

    assert initial_grid.shape != updated_grid.shape, "BaseLayer grid update failed to change grid shape"
    assert updated_grid.shape == (
        base_layer.n_in * base_layer.n_out,
        G_new + 2 * base_layer.k + 1
    ), "New BaseLayer grid shape incorrect"


@pytest.mark.parametrize("n_in, n_out, k, G", [
    (2, 5, 3, 3),
    (1, 1, 1, 1),
    (3, 10, 4, 5),
])
def test_spline_layer_varied_configs(seed, n_in, n_out, k, G):

    key = jax.random.key(seed)
    
    x = jax.random.uniform(key, shape=(10, n_in))
    
    layer = BaseLayer(n_in=n_in, n_out=n_out, k=k, G=G, seed=seed)
    
    y = layer(x)
    
    assert y.shape == (x.shape[0], n_out), f"Forward pass returned incorrect shape for n_in={n_in}, n_out={n_out}, k={k}, G={G}"
