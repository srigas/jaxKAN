import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.layers.RBF import RBFLayer


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
def gaussian_layer(seed, model_params):
    return RBFLayer(**model_params, kernel={"type": "gaussian"}, seed=seed)


# Tests
def test_rbf_layer_initialization(gaussian_layer, model_params):

    for model in [gaussian_layer]:
    
        assert model.n_in == model_params["n_in"], f"{model} n_in not set correctly"
        assert model.n_out == model_params["n_out"], f"{model} n_out not set correctly"
        assert model.c_basis.value.shape == (
            model_params["n_out"],
            model_params["n_in"],
            model_params["D"]
        ), f"{model} c_basis shape incorrect"
        assert model.c_ext.value.shape == (
            model_params["n_out"],
            model_params["n_in"]
        ), f"{model} c_ext shape incorrect"


def test_rbf_layer_forward_pass(gaussian_layer, x, model_params):

    for model in [gaussian_layer]:
    
        y = model(x)
        batch = x.shape[0]
    
        assert y.shape == (batch, model_params["n_out"]), f"{model} forward pass returned incorrect shape"


def test_rbf_layer_degree_update(gaussian_layer, x):
    
    D_new = 8

    for model in [gaussian_layer]:
        
        initial_degree = model.D
        model.update_grid(x, D_new)
        updated_degree = model.grid.D
    
        assert initial_degree != updated_degree, f"{model} degree update failed"
        assert updated_degree == D_new, f"{model} degree update did not set the new degree correctly"


@pytest.mark.parametrize("n_in, n_out, D", [
    (2, 5, 3),
    (1, 1, 1),
    (3, 10, 5),
])
def test_rbf_layer_varied_configs(seed, n_in, n_out, D):

    key = jax.random.key(seed)
    
    x = jax.random.uniform(key, shape=(10, n_in))

    for kernel_type in ["gaussian"]:
    
        layer = RBFLayer(n_in=n_in, n_out=n_out, D=D, kernel={"type": kernel_type}, seed=seed)
        
        y = layer(x)
        
        assert y.shape == (x.shape[0], n_out), f"Forward pass of {kernel_type} RBF layer returned incorrect shape for n_in={n_in}, n_out={n_out}, D={D}"
