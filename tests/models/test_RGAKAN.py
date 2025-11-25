import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.models.RGAKAN import RGAKAN, RGABlock


@pytest.fixture
def seed():
    return 42

@pytest.fixture
def sample_input(seed):
    key = jax.random.key(seed)
    return jax.random.uniform(key, shape=(10, 2))


# RGABlock Tests
def test_rga_block_initialization(seed):
    """Test RGABlock initialization."""
    block = RGABlock(n_in=64, n_out=64, n_hidden=64, D=5, flavor='exact', seed=seed)
    
    assert block.InputLayer is not None
    assert block.OutputLayer is not None
    assert hasattr(block, 'alpha')
    assert hasattr(block, 'beta')


def test_rga_block_forward(seed):
    """Test RGABlock forward pass."""
    block = RGABlock(n_in=64, n_out=64, n_hidden=64, D=5, seed=seed)
    
    x = jnp.ones((32, 64))
    u = jnp.ones((32, 64))
    v = jnp.zeros((32, 64))
    
    output = block(x, u, v)
    
    assert output.shape == (32, 64), "Output shape should match n_out"
    assert jnp.all(jnp.isfinite(output)), "Output should be finite"


# RGAKAN Tests
def test_rgakan_initialization(seed):
    """Test RGAKAN initialization."""
    model = RGAKAN(n_in=2, n_out=1, n_hidden=64, num_blocks=4, D=5, seed=seed)
    
    assert len(model.blocks) == 4, "Should have 4 blocks"
    assert model.U is not None
    assert model.V is not None


def test_rgakan_with_rff_embedder(seed):
    """Test RGAKAN with RFF embedding."""
    model = RGAKAN(n_in=2, n_out=1, n_hidden=64, num_blocks=2, 
                   rff_std=1.0, D=5, seed=seed)
    
    x = jnp.array([[1.0, 0.5], [2.0, 1.0]])
    output = model(x)
    
    assert output.shape == (2, 1), "Output shape should be (batch, n_out)"
    assert model.FE is not None, "RFFEmbedder should be initialized"


def test_rgakan_with_sine_basis(seed):
    """Test RGAKAN with sine basis layer."""
    model = RGAKAN(n_in=2, n_out=1, n_hidden=64, num_blocks=2, 
                   sine_D=3, D=5, seed=seed)
    
    x = jnp.array([[1.0, 0.5], [2.0, 1.0]])
    output = model(x)
    
    assert output.shape == (2, 1), "Output shape should be (batch, n_out)"
    assert model.SineBasis is not None, "SineLayer should be initialized"
