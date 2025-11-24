import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.models.KAN import KAN
from jaxkan.pikan.utils import model_eval


@pytest.fixture
def simple_model():
    """Create a simple KAN model for testing."""
    return KAN([2, 8, 1], 'spline', {'k': 3, 'G': 5}, 42)


def test_model_eval_shape(simple_model):
    """Test that model_eval returns scalar error."""
    coords = jnp.array([[0.5, 0.5], [0.1, 0.2], [0.8, 0.9]])
    refsol = jnp.array([[0.25], [0.02], [0.72]])
    
    error = model_eval(simple_model, coords, refsol)
    
    assert isinstance(error, jnp.ndarray) or isinstance(error, float)
    assert error.ndim == 0 or (error.ndim == 1 and error.shape[0] == 1), "Error should be scalar"


def test_model_eval_zero_error_perfect_fit(simple_model):
    """Test that model_eval gives near-zero error when prediction matches reference."""
    coords = jnp.array([[0.5, 0.5]])
    
    # Get model prediction
    prediction = simple_model(coords)
    
    # Use prediction as reference solution
    error = model_eval(simple_model, coords, prediction)
    
    assert jnp.allclose(error, 0.0, atol=1e-6), "Error should be near zero for perfect fit"


def test_model_eval_positive_error():
    """Test that model_eval returns positive error."""
    model = KAN([2, 8, 1], 'spline', {'k': 3, 'G': 5}, 42)
    coords = jnp.array([[0.5, 0.5], [0.1, 0.2]])
    refsol = jnp.array([[1.0], [2.0]])
    
    error = model_eval(model, coords, refsol)
    
    assert error >= 0, "Relative L2 error should be non-negative"


def test_model_eval_normalization():
    """Test that model_eval properly normalizes by reference solution norm."""
    model = KAN([2, 4, 1], 'spline', {'k': 3, 'G': 3}, 42)
    coords = jnp.array([[0.5, 0.5]])
    
    # Case 1: Small reference solution
    refsol_small = jnp.array([[0.1]])
    error_small = model_eval(model, coords, refsol_small)
    
    # Case 2: Large reference solution
    refsol_large = jnp.array([[10.0]])
    error_large = model_eval(model, coords, refsol_large)
    
    # Both should be finite and positive
    assert jnp.isfinite(error_small) and error_small >= 0
    assert jnp.isfinite(error_large) and error_large >= 0


def test_model_eval_multiple_points(simple_model):
    """Test model_eval with multiple evaluation points."""
    coords = jnp.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.4],
        [0.5, 0.5]
    ])
    refsol = jnp.array([[0.2], [0.4], [0.6], [0.8], [1.0]])
    
    error = model_eval(simple_model, coords, refsol)
    
    assert jnp.isfinite(error), "Error should be finite for multiple points"
    assert error >= 0, "Error should be non-negative"


def test_model_eval_different_ranges(simple_model):
    """Test model_eval with different coordinate ranges."""
    # Positive coordinates
    coords_pos = jnp.array([[0.5, 0.5], [0.8, 0.9]])
    refsol_pos = jnp.array([[0.5], [0.9]])
    error_pos = model_eval(simple_model, coords_pos, refsol_pos)
    
    # Negative coordinates
    coords_neg = jnp.array([[-0.5, -0.5], [-0.8, -0.9]])
    refsol_neg = jnp.array([[0.5], [0.9]])
    error_neg = model_eval(simple_model, coords_neg, refsol_neg)
    
    # Both should work
    assert jnp.isfinite(error_pos) and error_pos >= 0
    assert jnp.isfinite(error_neg) and error_neg >= 0


def test_model_eval_consistency():
    """Test that model_eval is consistent across multiple calls."""
    model = KAN([2, 6, 1], 'spline', {'k': 3, 'G': 4}, 42)
    coords = jnp.array([[0.3, 0.7], [0.6, 0.4]])
    refsol = jnp.array([[0.5], [0.8]])
    
    error1 = model_eval(model, coords, refsol)
    error2 = model_eval(model, coords, refsol)
    
    assert jnp.allclose(error1, error2), "model_eval should be deterministic"


def test_model_eval_known_case():
    """Test model_eval with a simple known case."""
    # Create very simple model
    model = KAN([2, 3, 1], 'spline', {'k': 2, 'G': 3}, 0)
    
    coords = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    
    # Get predictions
    preds = model(coords)
    
    # Create reference with known offset
    refsol = preds + 0.1
    
    # Compute error
    error = model_eval(model, coords, refsol)
    
    # Error should be related to the offset
    expected_error = jnp.linalg.norm(jnp.array([0.1, 0.1])) / jnp.linalg.norm(refsol)
    
    assert jnp.allclose(error, expected_error, rtol=1e-5), "Error calculation incorrect"


def test_model_eval_reshape_compatibility(simple_model):
    """Test that model_eval handles reshaping correctly."""
    coords = jnp.array([[0.5, 0.5], [0.1, 0.2]])
    
    # Reference solution in same shape as model output
    refsol_matched = jnp.array([[0.25], [0.02]])
    error_matched = model_eval(simple_model, coords, refsol_matched)
    
    assert jnp.isfinite(error_matched), "Should handle matched shapes"
    assert error_matched >= 0


def test_model_eval_single_point(simple_model):
    """Test model_eval with single evaluation point."""
    coords = jnp.array([[0.5, 0.5]])
    refsol = jnp.array([[0.75]])
    
    error = model_eval(simple_model, coords, refsol)
    
    assert jnp.isfinite(error), "Should handle single point"
    assert error >= 0
