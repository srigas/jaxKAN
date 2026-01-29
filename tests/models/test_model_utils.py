import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.models.KAN import KAN
from jaxkan.models.utils import count_params, get_frob, batched_frob, get_complexity, PeriodEmbedder, RFFEmbedder, get_activation


@pytest.fixture
def simple_model():
    """Create a simple KAN model for testing."""
    return KAN([2, 8, 1], 'spline', {'k': 3, 'G': 5}, 42)


@pytest.fixture
def small_model():
    """Create a very small KAN model for testing."""
    return KAN([2, 3, 1], 'spline', {'k': 2, 'G': 3}, 0)


def test_count_params_returns_int(simple_model):
    """Test that count_params returns an integer."""
    num_params = count_params(simple_model)
    
    assert isinstance(num_params, int), "count_params should return int"
    assert num_params > 0, "Model should have positive number of parameters"


def test_count_params_small_model(small_model):
    """Test count_params with a small model."""
    num_params = count_params(small_model)
    
    assert isinstance(num_params, int)
    assert num_params > 0


def test_count_params_different_architectures():
    """Test count_params with different model architectures."""
    model1 = KAN([2, 4, 1], 'spline', {'k': 3, 'G': 3}, 42)
    model2 = KAN([2, 8, 8, 1], 'spline', {'k': 3, 'G': 5}, 42)
    model3 = KAN([3, 10, 1], 'spline', {'k': 4, 'G': 4}, 42)
    
    params1 = count_params(model1)
    params2 = count_params(model2)
    params3 = count_params(model3)
    
    # Larger networks should generally have more parameters
    assert params2 > params1, "Larger network should have more parameters"
    assert all(p > 0 for p in [params1, params2, params3])


def test_get_frob_single_point(simple_model):
    """Test get_frob with a single point."""
    x = jnp.array([0.5, 0.3])
    frob_sq = get_frob(simple_model, x)
    
    assert isinstance(frob_sq, jnp.ndarray) or isinstance(frob_sq, float)
    assert frob_sq >= 0, "Squared Frobenius norm should be non-negative"
    assert jnp.isfinite(frob_sq), "Frobenius norm should be finite"


def test_get_frob_2d_input(simple_model):
    """Test get_frob with 2D input (batch size 1)."""
    x = jnp.array([[0.5, 0.3]])
    frob_sq = get_frob(simple_model, x)
    
    assert frob_sq >= 0
    assert jnp.isfinite(frob_sq)


def test_get_frob_different_points(simple_model):
    """Test get_frob at different input points."""
    x1 = jnp.array([0.1, 0.2])
    x2 = jnp.array([0.8, 0.9])
    
    frob1 = get_frob(simple_model, x1)
    frob2 = get_frob(simple_model, x2)
    
    # Both should be valid
    assert jnp.isfinite(frob1) and frob1 >= 0
    assert jnp.isfinite(frob2) and frob2 >= 0


def test_get_frob_zero_point(simple_model):
    """Test get_frob at origin."""
    x = jnp.array([0.0, 0.0])
    frob_sq = get_frob(simple_model, x)
    
    assert jnp.isfinite(frob_sq)
    assert frob_sq >= 0


def test_batched_frob_multiple_points(simple_model):
    """Test batched_frob with multiple points."""
    X = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    frob_values = batched_frob(simple_model, X)
    
    assert frob_values.shape == (3,), "Should return one value per input point"
    assert jnp.all(frob_values >= 0), "All Frobenius norms should be non-negative"
    assert jnp.all(jnp.isfinite(frob_values)), "All values should be finite"


def test_batched_frob_single_point(simple_model):
    """Test batched_frob with single point."""
    X = jnp.array([[0.5, 0.5]])
    frob_values = batched_frob(simple_model, X)
    
    assert frob_values.shape == (1,)
    assert frob_values[0] >= 0


def test_get_complexity_pde_only(simple_model):
    """Test get_complexity with PDE collocs only."""
    pde_collocs = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    
    complexity = get_complexity(simple_model, pde_collocs)
    
    assert isinstance(complexity, jnp.ndarray) or isinstance(complexity, float)
    assert complexity >= 0, "Complexity should be non-negative"
    assert jnp.isfinite(complexity), "Complexity should be finite"


def test_get_complexity_with_bc(simple_model):
    """Test get_complexity with both PDE and BC collocs."""
    pde_collocs = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    bc_collocs = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    
    complexity = get_complexity(simple_model, pde_collocs, bc_collocs)
    
    assert complexity >= 0
    assert jnp.isfinite(complexity)


def test_get_complexity_single_point(simple_model):
    """Test get_complexity with single collocation point."""
    pde_collocs = jnp.array([[0.5, 0.5]])
    
    complexity = get_complexity(simple_model, pde_collocs)
    
    assert complexity >= 0
    assert jnp.isfinite(complexity)


def test_get_complexity_is_average(simple_model):
    """Test that complexity is the average of Frobenius norms."""
    pde_collocs = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    
    complexity = get_complexity(simple_model, pde_collocs)
    
    # Manually compute average
    frob_values = batched_frob(simple_model, pde_collocs)
    expected_complexity = jnp.mean(frob_values)
    
    assert jnp.allclose(complexity, expected_complexity), "Complexity should be mean of Frobenius norms"


def test_get_complexity_with_bc_concatenation(simple_model):
    """Test that complexity correctly combines PDE and BC collocs."""
    pde_collocs = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    bc_collocs = jnp.array([[0.0, 0.0]])
    
    complexity = get_complexity(simple_model, pde_collocs, bc_collocs)
    
    # Manually compute with combined collocs
    combined = jnp.concatenate([pde_collocs, bc_collocs], axis=0)
    expected_complexity = jnp.mean(batched_frob(simple_model, combined))
    
    assert jnp.allclose(complexity, expected_complexity), "Should use combined collocs"


def test_get_complexity_different_models():
    """Test complexity computation with different model types."""
    models = [
        KAN([2, 4, 1], 'spline', {'k': 3, 'G': 3}, 42),
        KAN([2, 8, 1], 'chebyshev', {'D': 5}, 123),
        KAN([2, 6, 1], 'fourier', {'D': 4}, 99)
    ]
    
    pde_collocs = jnp.array([[0.2, 0.3], [0.5, 0.6]])
    
    for model in models:
        complexity = get_complexity(model, pde_collocs)
        assert complexity >= 0, f"Complexity should be non-negative for {type(model)}"
        assert jnp.isfinite(complexity), f"Complexity should be finite for {type(model)}"


def test_count_params_consistency():
    """Test that count_params is consistent across multiple calls."""
    model = KAN([2, 6, 1], 'spline', {'k': 3, 'G': 4}, 42)
    
    params1 = count_params(model)
    params2 = count_params(model)
    
    assert params1 == params2, "count_params should be deterministic"


# PeriodEmbedder Tests
def test_period_embedder_initialization():
    """Test PeriodEmbedder initialization."""
    period_axes = {0: (2.0 * jnp.pi, False), 1: (jnp.pi, True)}
    embedder = PeriodEmbedder(period_axes)
    
    assert hasattr(embedder.axes, '0')
    assert hasattr(embedder.axes, '1')


def test_period_embedder_forward():
    """Test PeriodEmbedder forward pass."""
    period_axes = {1: (jnp.pi, False)}
    embedder = PeriodEmbedder(period_axes)
    
    x = jnp.array([[1.0, 0.5], [2.0, 1.0]])
    y = embedder(x)
    
    # Axis 0 unchanged, axis 1 becomes [cos, sin]
    assert y.shape == (2, 3), "Output shape should be (2, 3)"


def test_period_embedder_trainable():
    """Test PeriodEmbedder with trainable period."""
    period_axes = {0: (2.0, True)}
    embedder = PeriodEmbedder(period_axes)
    
    x = jnp.array([[1.0], [2.0]])
    y = embedder(x)
    
    assert y.shape == (2, 2), "Should produce [cos, sin] for single axis"
    assert jnp.all(jnp.isfinite(y)), "Output should be finite"


def test_period_embedder_multiple_axes():
    """Test PeriodEmbedder with multiple periodic axes."""
    period_axes = {0: (2.0 * jnp.pi, False), 2: (jnp.pi, False)}
    embedder = PeriodEmbedder(period_axes)
    
    x = jnp.array([[1.0, 0.5, 0.3], [2.0, 1.0, 0.6]])
    y = embedder(x)
    
    # Axes 0 and 2 become [cos, sin], axis 1 unchanged
    assert y.shape == (2, 5), "Output shape should be (2, 5)"


# RFFEmbedder Tests
def test_rff_embedder_initialization():
    """Test RFFEmbedder initialization."""
    embedder = RFFEmbedder(std=1.0, n_in=2, embed_dim=256, seed=42)
    
    assert embedder.B[...].shape == (2, 128), "B matrix should be (n_in, embed_dim//2)"


def test_rff_embedder_forward():
    """Test RFFEmbedder forward pass."""
    embedder = RFFEmbedder(std=1.0, n_in=2, embed_dim=256, seed=42)
    
    x = jnp.array([[1.0, 0.5], [2.0, 1.0]])
    y = embedder(x)
    
    assert y.shape == (2, 256), "Output shape should be (batch, embed_dim)"
    assert jnp.all(jnp.isfinite(y)), "Output should be finite"


def test_rff_embedder_deterministic():
    """Test RFFEmbedder produces consistent results with same seed."""
    embedder1 = RFFEmbedder(std=1.0, n_in=2, embed_dim=128, seed=42)
    embedder2 = RFFEmbedder(std=1.0, n_in=2, embed_dim=128, seed=42)
    
    x = jnp.array([[1.0, 0.5]])
    y1 = embedder1(x)
    y2 = embedder2(x)
    
    assert jnp.allclose(y1, y2), "Same seed should produce same initialization and output"


# get_activation tests
def test_get_activation_returns_callable():
    """Test that get_activation returns a callable function."""
    activation = get_activation('tanh')
    
    assert callable(activation), "get_activation should return a callable"


def test_get_activation_common_functions():
    """Test get_activation with common activation functions."""
    activations = ['tanh', 'relu', 'silu', 'gelu', 'sigmoid']
    x = jnp.array([-1.0, 0.0, 1.0])
    
    for name in activations:
        act_fn = get_activation(name)
        y = act_fn(x)
        
        assert y.shape == x.shape, f"Activation {name} changed shape"
        assert jnp.all(jnp.isfinite(y)), f"Activation {name} produced non-finite values"


def test_get_activation_invalid():
    """Test that get_activation raises ValueError for unknown activation."""
    with pytest.raises(ValueError, match="Unknown activation"):
        get_activation('invalid_activation')


def test_get_activation_output_ranges():
    """Test that activations produce expected output ranges."""
    x = jnp.linspace(-3.0, 3.0, 100)
    
    # tanh should be in [-1, 1]
    tanh_out = get_activation('tanh')(x)
    assert jnp.all(tanh_out >= -1.0) and jnp.all(tanh_out <= 1.0), "tanh should be in [-1, 1]"
    
    # sigmoid should be in [0, 1]
    sigmoid_out = get_activation('sigmoid')(x)
    assert jnp.all(sigmoid_out >= 0.0) and jnp.all(sigmoid_out <= 1.0), "sigmoid should be in [0, 1]"
    
    # relu should be >= 0
    relu_out = get_activation('relu')(x)
    assert jnp.all(relu_out >= 0.0), "relu should be >= 0"


# get_optimizer tests
def test_get_adam_import():
    """Test that get_adam can be imported."""
    from jaxkan.models.utils import get_adam
    assert callable(get_adam)


def test_get_adam_constant_lr(simple_model):
    """Test get_adam with constant learning rate."""
    from jaxkan.models.utils import get_adam
    
    optimizer = get_adam(
        learning_rate=1e-3
    )
    
    # Test that optimizer can be initialized
    opt_state = optimizer.init(simple_model)
    assert opt_state is not None


def test_get_adam_exponential_decay(simple_model):
    """Test get_adam with exponential decay schedule."""
    from jaxkan.models.utils import get_adam
    
    optimizer = get_adam(
        learning_rate=1e-3,
        schedule_type='exponential',
        decay_steps=5000,
        decay_rate=0.9
    )
    
    opt_state = optimizer.init(simple_model)
    assert opt_state is not None


def test_get_adam_with_warmup(simple_model):
    """Test get_adam with warmup."""
    from jaxkan.models.utils import get_adam
    
    optimizer = get_adam(
        learning_rate=1e-3,
        schedule_type='exponential',
        decay_steps=5000,
        decay_rate=0.9,
        warmup_steps=1000
    )
    
    opt_state = optimizer.init(simple_model)
    assert opt_state is not None


def test_get_adam_cosine_schedule(simple_model):
    """Test get_adam with cosine annealing."""
    from jaxkan.models.utils import get_adam
    
    optimizer = get_adam(
        learning_rate=1e-3,
        schedule_type='cosine',
        decay_steps=10000
    )
    
    opt_state = optimizer.init(simple_model)
    assert opt_state is not None


def test_get_adam_polynomial_schedule(simple_model):
    """Test get_adam with polynomial decay."""
    from jaxkan.models.utils import get_adam
    
    optimizer = get_adam(
        learning_rate=1e-3,
        schedule_type='polynomial',
        decay_steps=5000,
        decay_rate=2.0
    )
    
    opt_state = optimizer.init(simple_model)
    assert opt_state is not None


def test_get_adam_custom_params(simple_model):
    """Test get_adam with custom Adam parameters."""
    from jaxkan.models.utils import get_adam
    
    optimizer = get_adam(
        learning_rate=1e-3,
        b1=0.95,
        b2=0.999,
        eps=1e-7
    )
    
    opt_state = optimizer.init(simple_model)
    assert opt_state is not None


def test_get_adam_invalid_schedule():
    """Test that get_adam raises error for invalid schedule type."""
    from jaxkan.models.utils import get_adam
    
    with pytest.raises(ValueError, match="Unknown schedule_type"):
        get_adam(
            learning_rate=1e-3,
            schedule_type='invalid_schedule'
        )


def test_get_adam_initialization_with_params(simple_model):
    """Test that optimizer can be initialized with model parameters."""
    from jaxkan.models.utils import get_adam
    import optax
    
    optimizer = get_adam(
        learning_rate=1e-3,
        schedule_type='exponential',
        decay_steps=5000,
        decay_rate=0.9,
        warmup_steps=1000
    )
    
    # Extract parameters for optax (as State object)
    params = nnx.state(simple_model, nnx.Param)
    
    # Initialize optimizer state with parameters
    opt_state = optimizer.init(params)
    
    assert opt_state is not None
    
    # Test that we can perform an update with proper gradients
    # Create dummy gradients with same structure as params
    grads = jax.tree.map(lambda x: jnp.ones_like(x), params)
    
    # Perform update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    
    assert updates is not None
    assert new_opt_state is not None
    
    # Apply updates to parameters
    new_params = optax.apply_updates(params, updates)
    assert new_params is not None


def test_get_adam_warmup_only(simple_model):
    """Test optimizer with warmup and constant learning rate after."""
    from jaxkan.models.utils import get_adam
    
    optimizer = get_adam(
        learning_rate=1e-3,
        warmup_steps=500
    )
    
    opt_state = optimizer.init(simple_model)
    assert opt_state is not None


def test_get_adam_staircase_decay(simple_model):
    """Test optimizer with staircase decay."""
    from jaxkan.models.utils import get_adam
    
    optimizer = get_adam(
        learning_rate=1e-3,
        schedule_type='exponential',
        decay_steps=1000,
        decay_rate=0.5,
        staircase=True
    )
    
    opt_state = optimizer.init(simple_model)
    assert opt_state is not None

