import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.models.KAN import KAN
from jaxkan.models.utils import count_params, get_frob, batched_frob, get_complexity


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
