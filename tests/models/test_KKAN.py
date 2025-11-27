import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.models.KKAN import ChebyshevEmbedding, InnerBlock, OuterBlock, KKAN


@pytest.fixture
def seed():
    return 42

@pytest.fixture
def sample_input(seed):
    key = jax.random.key(seed)
    return jax.random.uniform(key, shape=(10, 2), minval=-1.0, maxval=1.0)


# ChebyshevEmbedding Tests
def test_chebyshev_embedding_initialization():
    """Test ChebyshevEmbedding initializes correctly."""
    D_e = 5
    embedding = ChebyshevEmbedding(D_e=D_e)
    
    assert embedding.D_e == D_e, "D_e not set correctly"
    assert embedding.C[...].shape == (D_e + 1,), "C shape incorrect"
    # Check initialization: c_i = 1/(i+1)
    expected_C = jnp.array([1.0 / (i + 1) for i in range(D_e + 1)])
    assert jnp.allclose(embedding.C[...], expected_C), "C initialization incorrect"


def test_chebyshev_embedding_forward_pass():
    """Test ChebyshevEmbedding forward pass."""
    D_e = 5
    embedding = ChebyshevEmbedding(D_e=D_e)
    
    x = jnp.array([[0.5], [-0.3], [0.8]])  # (3, 1)
    y = embedding(x)
    
    assert y.shape == (3, D_e + 1), "Output shape incorrect"


def test_chebyshev_embedding_1d_input():
    """Test ChebyshevEmbedding handles 1D input."""
    D_e = 3
    embedding = ChebyshevEmbedding(D_e=D_e)
    
    x = jnp.array([0.5, -0.3, 0.8])  # (3,) - 1D input
    y = embedding(x)
    
    assert y.shape == (3, D_e + 1), "1D input handling incorrect"


# InnerBlock Tests
def test_inner_block_initialization(seed):
    """Test InnerBlock initializes correctly."""
    inner = InnerBlock(D_e=5, H=16, L=2, m=8, activation='tanh', seed=seed)
    
    assert inner.input_embedding is not None
    assert inner.input_layer is not None
    assert len(inner.hidden_layers) == 2, "Hidden layers count incorrect"
    assert inner.output_embedding is not None
    assert inner.output_layer is not None


def test_inner_block_forward_pass(seed):
    """Test InnerBlock forward pass."""
    inner = InnerBlock(D_e=5, H=16, L=2, m=8, seed=seed)
    
    x_p = jnp.array([[0.5], [-0.3], [0.8]])  # (3, 1)
    y = inner(x_p)
    
    assert y.shape == (3, 8), "InnerBlock output shape incorrect"


# OuterBlock Tests
def test_outer_block_initialization(seed):
    """Test OuterBlock initializes correctly."""
    outer = OuterBlock(m=10, n_out=1, layer_type='sine', seed=seed)
    
    assert outer.layer is not None


def test_outer_block_forward_pass(seed):
    """Test OuterBlock forward pass."""
    outer = OuterBlock(m=10, n_out=1, layer_type='chebyshev', 
                       layer_params={'D': 5}, seed=seed)
    
    xi = jax.random.uniform(jax.random.key(seed), (5, 10), minval=-1.0, maxval=1.0)
    y = outer(xi)
    
    assert y.shape == (5, 1), "OuterBlock output shape incorrect"


# KKAN Tests
def test_kkan_initialization(seed):
    """Test KKAN model initializes correctly."""
    model = KKAN(n_in=2, n_out=1, m=8, D_e=5, H=16, L=2, seed=seed)
    
    assert model.n_in == 2, "n_in not set correctly"
    assert len(model.inner_blocks) == 2, "Should have one inner block per input dimension"
    assert model.outer_block is not None


def test_kkan_forward_pass(seed, sample_input):
    """Test KKAN forward pass."""
    model = KKAN(n_in=2, n_out=1, m=8, D_e=5, H=16, L=2, seed=seed)
    y = model(sample_input)
    
    assert y.shape == (10, 1), "KKAN output shape incorrect"


def test_kkan_different_outer_layers(seed, sample_input):
    """Test KKAN with different outer layer types."""
    layer_types = ['chebyshev', 'sine', 'legendre']
    
    for layer_type in layer_types:
        model = KKAN(n_in=2, n_out=1, m=8, D_e=3, H=8, L=1,
                     outer_layer_type=layer_type,
                     outer_layer_params={'D': 3},
                     seed=seed)
        y = model(sample_input)
        
        assert y.shape == (10, 1), f"KKAN with {layer_type} outer layer failed"
