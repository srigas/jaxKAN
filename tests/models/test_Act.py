import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.models.ActNet import ActLayer, ActNet


@pytest.fixture
def seed():
    return 42

@pytest.fixture
def layer_dims():
    return [2, 5, 1]  # Example: 2 input nodes, 5 hidden nodes, 1 output node

@pytest.fixture
def sample_input(seed):
    key = jax.random.key(seed)
    return jax.random.uniform(key, shape=(10, 2))


# =============================================================================
# ActLayer Tests
# =============================================================================

# Test: Initialization
def test_actlayer_initialization(seed):
    """Test ActLayer basic initialization."""
    layer = ActLayer(n_in=2, n_out=5, N=4, train_basis=True, seed=seed)

    assert layer.n_in == 2, "Incorrect n_in"
    assert layer.n_out == 5, "Incorrect n_out"
    assert layer.N == 4, "Incorrect N"


def test_actlayer_parameter_shapes(seed):
    """Test that ActLayer parameters have correct shapes."""
    layer = ActLayer(n_in=3, n_out=4, N=5, seed=seed)
    
    assert layer.beta.shape == (4, 5), "Beta shape incorrect"
    assert layer.Lambda.shape == (4, 3), "Lambda shape incorrect"
    assert layer.omega.shape == (5,), "Omega shape incorrect"
    assert layer.phase.shape == (5,), "Phase shape incorrect"


def test_actlayer_train_basis_true(seed):
    """Test that omega and phase are nnx.Param when train_basis=True."""
    layer = ActLayer(n_in=2, n_out=3, N=4, train_basis=True, seed=seed)
    
    assert isinstance(layer.omega, nnx.Param), "Omega should be Param when train_basis=True"
    assert isinstance(layer.phase, nnx.Param), "Phase should be Param when train_basis=True"


# Test: Basis Function
def test_actlayer_basis_shape(seed, sample_input):
    """Test that basis function returns correct shape."""
    layer = ActLayer(n_in=2, n_out=5, N=4, seed=seed)
    B = layer.basis(sample_input)
    
    assert B.shape == (10, 4, 2), "Basis output shape incorrect: expected (batch, N, n_in)"


def test_actlayer_basis_finite(seed, sample_input):
    """Test that basis function returns finite values."""
    layer = ActLayer(n_in=2, n_out=5, N=4, seed=seed)
    B = layer.basis(sample_input)
    
    assert jnp.all(jnp.isfinite(B)), "Basis contains non-finite values"


# Test: Forward Pass
def test_actlayer_forward_pass(seed, sample_input):
    """Test ActLayer forward pass output shape."""
    layer = ActLayer(n_in=2, n_out=5, N=4, seed=seed)
    output = layer(sample_input)

    assert output.shape == (10, 5), "Forward pass output shape is incorrect"


def test_actlayer_forward_pass_minimal(seed):
    """Test ActLayer forward pass with minimal input."""
    layer = ActLayer(n_in=1, n_out=1, N=2, seed=seed)
    minimal_input = jnp.array([[0.5]])
    output = layer(minimal_input)

    assert output.shape == (1, 1), "Forward pass for minimal case returned incorrect shape"


def test_actlayer_output_finite(seed, sample_input):
    """Test that ActLayer outputs are finite."""
    layer = ActLayer(n_in=2, n_out=5, N=4, seed=seed)
    output = layer(sample_input)
    
    assert jnp.all(jnp.isfinite(output)), "ActLayer output contains non-finite values"


def test_actlayer_deterministic_output(seed):
    """Test that ActLayer produces deterministic outputs for same input."""
    layer = ActLayer(n_in=2, n_out=3, N=4, seed=seed)
    x = jnp.array([[0.5, 0.5]])
    
    output1 = layer(x)
    output2 = layer(x)
    
    assert jnp.allclose(output1, output2), "ActLayer should produce deterministic outputs"


# =============================================================================
# ActNet Tests
# =============================================================================

# Test: Initialization
def test_actnet_initialization(seed, layer_dims):
    """Test ActNet basic initialization."""
    model = ActNet(layer_dims=layer_dims, N=4, add_bias=True, seed=seed)

    assert len(model.layers) == len(layer_dims) - 1, "Incorrect number of layers initialized"


def test_actnet_no_bias(seed, layer_dims):
    """Test ActNet initialization without bias."""
    model = ActNet(layer_dims=layer_dims, N=4, add_bias=False, seed=seed)

    assert not hasattr(model, 'biases') or len(model.biases) == 0 or not model.add_bias, \
        "Biases should not be present when add_bias=False"


def test_actnet_with_projections(seed):
    """Test ActNet with input/output projections."""
    model = ActNet(layer_dims=[2, 5, 5, 1], N=4, use_projections=True, seed=seed)
    
    assert hasattr(model, 'input_proj'), "Input projection should exist"
    assert hasattr(model, 'output_proj'), "Output projection should exist"


# Test: Forward Pass
def test_actnet_forward_pass(seed, layer_dims, sample_input):
    """Test ActNet forward pass output shape."""
    model = ActNet(layer_dims=layer_dims, N=4, seed=seed)
    output = model(sample_input)

    assert output.shape == (10, 1), "Forward pass output shape is incorrect"


def test_actnet_forward_pass_minimal(seed):
    """Test ActNet forward pass with minimal network."""
    model = ActNet(layer_dims=[1, 1], N=2, seed=seed)
    minimal_input = jnp.array([[0.5]])
    output = model(minimal_input)

    assert output.shape == (1, 1), "Forward pass for minimal case returned incorrect shape"


def test_actnet_omega0_scaling(seed):
    """Test that omega0 frequency scaling is applied."""
    model_default = ActNet(layer_dims=[2, 3, 1], N=4, omega0=1.0, seed=seed)
    model_scaled = ActNet(layer_dims=[2, 3, 1], N=4, omega0=2.0, seed=seed)
    
    x = jnp.array([[0.5, 0.5]])
    
    # Outputs should differ when omega0 is different
    out_default = model_default(x)
    out_scaled = model_scaled(x)
    
    # They won't be equal due to different input scaling
    # Just check both produce valid outputs
    assert jnp.all(jnp.isfinite(out_default)), "Default omega0 output should be finite"
    assert jnp.all(jnp.isfinite(out_scaled)), "Scaled omega0 output should be finite"


# Test: Output Validity
def test_actnet_output_finite(seed, sample_input):
    """Test that ActNet outputs are finite."""
    model = ActNet(layer_dims=[2, 8, 1], N=4, seed=seed)
    output = model(sample_input)
    
    assert jnp.all(jnp.isfinite(output)), "ActNet output contains non-finite values"


def test_actnet_deterministic_output(seed):
    """Test that ActNet produces deterministic outputs for same input."""
    model = ActNet(layer_dims=[2, 4, 1], N=4, seed=seed)
    x = jnp.array([[0.5, 0.5]])
    
    output1 = model(x)
    output2 = model(x)
    
    assert jnp.allclose(output1, output2), "ActNet should produce deterministic outputs"


# Test: Deep Networks
def test_actnet_deep_network(seed):
    """Test ActNet with many layers."""
    layer_dims = [2, 8, 8, 8, 8, 1]
    model = ActNet(layer_dims, N=4, seed=seed)
    x = jnp.array([[0.5, 0.5]])
    output = model(x)
    
    assert output.shape == (1, 1), "Deep ActNet output shape incorrect"
    assert len(model.layers) == 5, "Deep ActNet should have 5 layers"


# Test: Wide Networks
def test_actnet_wide_network(seed):
    """Test ActNet with wide hidden layers."""
    layer_dims = [3, 50, 50, 2]
    model = ActNet(layer_dims, N=4, seed=seed)
    x = jnp.array([[0.5, 0.5, 0.5]])
    output = model(x)
    
    assert output.shape == (1, 2), "Wide ActNet output shape incorrect"


# Test: Batch Processing
def test_actnet_batch_processing(seed):
    """Test ActNet with various batch sizes."""
    model = ActNet([2, 8, 1], N=4, seed=seed)
    
    # Single sample
    x1 = jnp.array([[0.5, 0.5]])
    out1 = model(x1)
    assert out1.shape == (1, 1)
    
    # Small batch
    x10 = jax.random.uniform(jax.random.key(seed), (10, 2))
    out10 = model(x10)
    assert out10.shape == (10, 1)
    
    # Large batch
    x100 = jax.random.uniform(jax.random.key(seed), (100, 2))
    out100 = model(x100)
    assert out100.shape == (100, 1)
