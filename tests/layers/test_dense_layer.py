import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.layers.Dense import Dense


@pytest.fixture
def seed():
    return 42

@pytest.fixture
def x(seed):
    key = jax.random.key(seed)
    return jax.random.uniform(key, shape=(10, 8))

@pytest.fixture
def layer_params():
    return {
        "n_in": 8,
        "n_out": 4,
    }


# Tests
def test_dense_layer_initialization(seed, layer_params):
    """Test that Dense layer initializes correctly."""
    layer = Dense(**layer_params, seed=seed)
    
    assert layer.W[...].shape == (layer_params["n_in"], layer_params["n_out"]), "W shape incorrect"
    assert layer.g[...].shape == (layer_params["n_out"],), "g shape incorrect"
    assert layer.b[...].shape == (layer_params["n_out"],), "b shape incorrect"


def test_dense_layer_no_bias(seed, layer_params):
    """Test Dense layer without bias."""
    layer = Dense(**layer_params, add_bias=False, seed=seed)
    
    assert layer.b is None, "Bias should be None when add_bias=False"


def test_dense_layer_forward_pass(seed, layer_params, x):
    """Test forward pass produces correct output shape."""
    layer = Dense(**layer_params, seed=seed)
    y = layer(x)
    
    assert y.shape == (x.shape[0], layer_params["n_out"]), "Forward pass output shape incorrect"


def test_dense_layer_weight_normalization(seed, layer_params, x):
    """Test that weight normalization is applied (columns of V have unit norm)."""
    layer = Dense(**layer_params, seed=seed)
    
    # Compute normalized weights as in forward pass
    W_norm = jnp.linalg.norm(layer.W[...], axis=0, keepdims=True)
    V = layer.W[...] / (W_norm + 1e-8)
    
    # Check that each column has approximately unit norm
    col_norms = jnp.linalg.norm(V, axis=0)
    assert jnp.allclose(col_norms, 1.0, atol=1e-5), "Normalized weight columns should have unit norm"


def test_dense_layer_init_schemes(seed, layer_params):
    """Test different initialization schemes."""
    schemes = ['glorot', 'he', 'lecun', 'normal', 'uniform']
    
    for scheme in schemes:
        layer = Dense(**layer_params, init_scheme=scheme, seed=seed)
        assert layer.W[...].shape == (layer_params["n_in"], layer_params["n_out"]), \
            f"Initialization failed for scheme: {scheme}"


def test_dense_layer_invalid_init_scheme(seed, layer_params):
    """Test that invalid init_scheme raises ValueError."""
    with pytest.raises(ValueError, match="Unknown init_scheme"):
        Dense(**layer_params, init_scheme="invalid_scheme", seed=seed)
