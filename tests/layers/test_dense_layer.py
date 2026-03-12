import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.layers.Dense import DenseLayer


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
    """Test that DenseLayer initializes correctly."""
    layer = DenseLayer(**layer_params, seed=seed)
    
    assert layer.v[...].shape == (layer_params["n_in"], layer_params["n_out"]), "v shape incorrect"
    assert layer.g[...].shape == (layer_params["n_out"],), "g shape incorrect"
    assert layer.b[...].shape == (layer_params["n_out"],), "b shape incorrect"


def test_dense_layer_no_bias(seed, layer_params):
    """Test DenseLayer without bias."""
    layer = DenseLayer(**layer_params, add_bias=False, seed=seed)
    
    assert layer.b is None, "Bias should be None when add_bias=False"


def test_dense_layer_forward_pass(seed, layer_params, x):
    """Test forward pass produces correct output shape."""
    layer = DenseLayer(**layer_params, seed=seed)
    y = layer(x)
    
    assert y.shape == (x.shape[0], layer_params["n_out"]), "Forward pass output shape incorrect"


def test_dense_layer_weight_normalization(seed, layer_params, x):
    """Test that the RWF kernel is correctly reconstructed as g * v."""
    layer = DenseLayer(**layer_params, seed=seed)
    
    # Reconstruct kernel as in forward pass
    expected = jnp.dot(x, layer.g[...] * layer.v[...]) + layer.b[...]
    actual = layer(x)
    
    assert jnp.allclose(actual, expected, atol=1e-5), "RWF kernel reconstruction mismatch"


def test_dense_layer_init_schemes(seed, layer_params):
    """Test different RWF configurations."""
    rwf_configs = [
        {"mean": 1.0, "std": 0.1},
        {"mean": 0.5, "std": 0.2},
        {"mean": 2.0, "std": 0.05},
        {"mean": 0.0, "std": 0.3},
        {"mean": 1.5, "std": 0.15},
    ]
    
    for rwf in rwf_configs:
        layer = DenseLayer(**layer_params, RWF=rwf, seed=seed)
        assert layer.v[...].shape == (layer_params["n_in"], layer_params["n_out"]), \
            f"Initialization failed for RWF config: {rwf}"


def test_dense_layer_invalid_init_scheme(seed, layer_params):
    """Test that an incomplete RWF dict raises KeyError."""
    with pytest.raises(KeyError):
        DenseLayer(**layer_params, RWF={"mean": 1.0}, seed=seed)  # missing 'std'
