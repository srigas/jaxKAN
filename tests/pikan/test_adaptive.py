import pytest

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from jaxkan.models.KAN import KAN
from jaxkan.pikan.adaptive import (
    apply_rba_weights,
    get_causal_weights,
    get_colloc_indices,
    get_rad_indices,
    get_rad_probabilities,
    lr_anneal,
    update_rba_weights,
)


@pytest.fixture
def sample_collocs_pool():
    """Create a sample pool of collocation points."""
    return jnp.array([
        [0.1, 0.2],
        [0.5, 0.3],
        [0.2, 0.8],
        [0.9, 0.1],
        [0.6, 0.7],
        [0.3, 0.4],
        [0.8, 0.6],
        [0.4, 0.9],
        [0.7, 0.5],
        [0.0, 0.0]
    ])


@pytest.fixture
def uniform_weights():
    """Create uniform probability weights."""
    return jnp.ones(10) / 10.0


@pytest.fixture
def non_uniform_weights():
    """Create non-uniform probability weights."""
    weights = jnp.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.01, 0.01])
    return weights / jnp.sum(weights)


def test_get_colloc_indices_shape(sample_collocs_pool, uniform_weights):
    """Test that get_colloc_indices returns correct shape."""
    batch_size = 5
    indices = get_colloc_indices(sample_collocs_pool, batch_size, uniform_weights, seed=42)
    
    assert indices.shape == (batch_size,), "Indices shape incorrect"
    assert jnp.all(indices >= 0), "Indices contain negative values"
    assert jnp.all(indices < sample_collocs_pool.shape[0]), "Indices out of bounds"


def test_get_colloc_indices_no_duplicates(sample_collocs_pool, uniform_weights):
    """Test that get_colloc_indices returns unique indices."""
    batch_size = 5
    indices = get_colloc_indices(sample_collocs_pool, batch_size, uniform_weights, seed=42)
    
    unique_indices = jnp.unique(indices)
    assert len(unique_indices) == batch_size, "Indices contain duplicates"


def test_get_colloc_indices_sorted(sample_collocs_pool, uniform_weights):
    """Test that returned indices correspond to points sorted by first coordinate."""
    batch_size = 5
    indices = get_colloc_indices(sample_collocs_pool, batch_size, uniform_weights, seed=42)
    
    selected_points = sample_collocs_pool[indices]
    first_coords = selected_points[:, 0]
    
    # Check if first coordinates are sorted
    assert jnp.all(first_coords[:-1] <= first_coords[1:]), "Points not sorted by first coordinate"


def test_get_colloc_indices_reproducibility(sample_collocs_pool, uniform_weights):
    """Test that same seed produces same indices."""
    batch_size = 6
    
    indices1 = get_colloc_indices(sample_collocs_pool, batch_size, uniform_weights, seed=123)
    indices2 = get_colloc_indices(sample_collocs_pool, batch_size, uniform_weights, seed=123)
    
    assert jnp.array_equal(indices1, indices2), "Same seed should produce identical indices"


def test_get_colloc_indices_different_seeds(sample_collocs_pool, uniform_weights):
    """Test that different seeds produce different indices."""
    batch_size = 5
    
    indices1 = get_colloc_indices(sample_collocs_pool, batch_size, uniform_weights, seed=42)
    indices2 = get_colloc_indices(sample_collocs_pool, batch_size, uniform_weights, seed=99)
    
    # With high probability, different seeds should give different indices
    assert not jnp.array_equal(indices1, indices2), "Different seeds should produce different indices"


def test_get_colloc_indices_weighted_sampling(sample_collocs_pool, non_uniform_weights):
    """Test weighted sampling with non-uniform probabilities."""
    batch_size = 5
    indices = get_colloc_indices(sample_collocs_pool, batch_size, non_uniform_weights, seed=42)
    
    assert indices.shape == (batch_size,), "Weighted sampling shape incorrect"
    assert jnp.all(indices >= 0) and jnp.all(indices < sample_collocs_pool.shape[0])


def test_update_rba_weights_shape_and_finiteness():
    """Test RBA weight updates preserve shape and finiteness."""
    residuals = jnp.array([[1.0], [-2.0], [0.5]])
    weights = jnp.ones_like(residuals)

    updated = update_rba_weights(residuals, weights, gamma=0.9, eta=0.1)

    assert updated.shape == residuals.shape, "Updated RBA weights shape incorrect"
    assert jnp.all(jnp.isfinite(updated)), "Updated RBA weights contain non-finite values"


def test_apply_rba_weights_stops_weight_gradients():
    """Test that RBA weights are detached from the backward graph."""
    residuals = jnp.array([[1.0], [2.0]])
    weights = jnp.array([[3.0], [4.0]])

    grad_w = jax.grad(lambda w: jnp.sum(apply_rba_weights(residuals, w)))(weights)
    grad_r = jax.grad(lambda r: jnp.sum(apply_rba_weights(r, weights)))(residuals)

    assert jnp.allclose(grad_w, 0.0), "RBA weights should have zero gradient after stop_gradient"
    assert jnp.allclose(grad_r, weights), "Residual gradients should match the detached weights"


def test_get_causal_weights_shape_and_detachment():
    """Test causal weight computation shape and stopped-gradient behavior."""
    losses = jnp.array([1.0, 2.0, 3.0])
    causal_matrix = jnp.triu(jnp.ones((3, 3)), k=1).T

    weights = get_causal_weights(losses, causal_matrix, causal_tol=0.5)
    grad_losses = jax.grad(lambda vals: jnp.sum(get_causal_weights(vals, causal_matrix, 0.5)))(losses)

    assert weights.shape == losses.shape, "Causal weights shape incorrect"
    assert jnp.all(jnp.isfinite(weights)), "Causal weights contain non-finite values"
    assert jnp.allclose(grad_losses, 0.0), "Causal weights should be detached from the backward graph"


def test_get_rad_probabilities_normalized():
    """Test RAD probabilities are positive and normalized."""
    weighted_residuals = jnp.array([[1.0], [2.0], [0.5], [4.0]])

    px = get_rad_probabilities(weighted_residuals, rad_a=1.0, rad_c=1.0)

    assert px.shape == (4,), "RAD probability vector shape incorrect"
    assert jnp.all(px > 0), "RAD probabilities should be positive"
    assert jnp.allclose(jnp.sum(px), 1.0), "RAD probabilities should sum to 1"


def test_get_rad_indices_updates_pool_and_samples(sample_collocs_pool):
    """Test generic RAD resampling helper updates the pool and samples a sorted batch."""
    residuals = jnp.arange(1, sample_collocs_pool.shape[0] + 1, dtype=jnp.float32).reshape(-1, 1)
    pool_weights = jnp.ones_like(residuals)
    old_indices = jnp.array([1, 3, 5])
    batch_weights = jnp.array([[2.0], [3.0], [4.0]])

    indices, updated_pool, px = get_rad_indices(
        collocs_pool=sample_collocs_pool,
        residuals=residuals,
        old_indices=old_indices,
        batch_weights=batch_weights,
        pool_weights=pool_weights,
        batch_size=4,
        rad_a=1.0,
        rad_c=1.0,
        seed=42,
    )

    assert indices.shape == (4,), "RAD indices shape incorrect"
    assert updated_pool.shape == pool_weights.shape, "Updated pool weights shape incorrect"
    assert px.shape == (sample_collocs_pool.shape[0],), "RAD probabilities shape incorrect"
    assert jnp.allclose(updated_pool[old_indices], batch_weights), "Old batch weights were not written back to the pool"

    sampled_points = sample_collocs_pool[indices]
    assert jnp.all(sampled_points[:-1, 0] <= sampled_points[1:, 0]), "RAD indices should be sorted by the first coordinate"


def test_lr_anneal_shape():
    """Test that lr_anneal returns correct shapes."""
    # Create simple gradient structures
    grads_E = {'w': jnp.array([[1.0, 2.0], [3.0, 4.0]])}
    grads_B = {'w': jnp.array([[0.5, 1.0], [1.5, 2.0]])}
    
    λ_E_new, λ_B_new = lr_anneal((grads_E, grads_B), (1.0, 1.0), 0.9)
    
    assert isinstance(λ_E_new, jnp.ndarray) or isinstance(λ_E_new, float)
    assert isinstance(λ_B_new, jnp.ndarray) or isinstance(λ_B_new, float)


def test_lr_anneal_positive_weights():
    """Test that lr_anneal returns positive weights."""
    grads_E = {'w': jnp.array([[1.0, 2.0]])}
    grads_B = {'w': jnp.array([[0.5, 1.0]])}
    
    λ_E_new, λ_B_new = lr_anneal((grads_E, grads_B), (1.0, 1.0), 0.9)
    
    assert λ_E_new > 0, "Equation weight should be positive"
    assert λ_B_new > 0, "Boundary weight should be positive"


def test_lr_anneal_finite_values():
    """Test that lr_anneal returns finite values."""
    grads_E = {'w': jnp.array([[1.0, 2.0], [3.0, 4.0]])}
    grads_B = {'w': jnp.array([[0.5, 1.0], [1.5, 2.0]])}
    
    λ_E_new, λ_B_new = lr_anneal((grads_E, grads_B), (1.0, 1.0), 0.9)
    
    assert jnp.isfinite(λ_E_new), "Equation weight should be finite"
    assert jnp.isfinite(λ_B_new), "Boundary weight should be finite"


def test_lr_anneal_balancing():
    """Test that lr_anneal balances large gradient differences."""
    # Large PDE gradients, small BC gradients
    grads_E = {'w': jnp.array([[10.0, 20.0]])}
    grads_B = {'w': jnp.array([[0.1, 0.2]])}
    
    λ_E_new, λ_B_new = lr_anneal((grads_E, grads_B), (1.0, 1.0), 0.5)
    
    # When PDE gradients are larger, its weight should decrease (and BC weight increase)
    # This is because we want to balance the gradient magnitudes
    assert λ_B_new > λ_E_new, "Should increase BC weight when PDE gradients dominate"


def test_lr_anneal_grad_mixing_effect():
    """Test the effect of grad_mixing parameter."""
    grads_E = {'w': jnp.array([[1.0, 2.0]])}
    grads_B = {'w': jnp.array([[3.0, 4.0]])}
    
    # High mixing (more weight on previous values)
    λ_E_high, λ_B_high = lr_anneal((grads_E, grads_B), (2.0, 3.0), 0.99)
    
    # Low mixing (more weight on new values)
    λ_E_low, λ_B_low = lr_anneal((grads_E, grads_B), (2.0, 3.0), 0.01)
    
    # High mixing should be closer to original values
    assert abs(λ_E_high - 2.0) < abs(λ_E_low - 2.0)
    assert abs(λ_B_high - 3.0) < abs(λ_B_low - 3.0)


def test_lr_anneal_zero_gradients():
    """Test lr_anneal with very small/zero gradients."""
    grads_E = {'w': jnp.array([[1e-10, 1e-10]])}
    grads_B = {'w': jnp.array([[1e-10, 1e-10]])}
    
    λ_E_new, λ_B_new = lr_anneal((grads_E, grads_B), (1.0, 1.0), 0.9)
    
    assert jnp.isfinite(λ_E_new), "Should handle tiny gradients without NaN/Inf"
    assert jnp.isfinite(λ_B_new), "Should handle tiny gradients without NaN/Inf"


def test_lr_anneal_with_real_model():
    """Test lr_anneal with actual model gradients."""
    model = KAN([2, 4, 1], 'spline', {'k': 3, 'G': 3}, 42)
    
    def loss_E(model):
        x = jnp.array([[0.5, 0.5]])
        return jnp.sum(model(x)**2)
    
    def loss_B(model):
        x = jnp.array([[0.0, 0.0]])
        return jnp.sum((model(x) - 1.0)**2)
    
    grads_E = nnx.grad(loss_E)(model)
    grads_B = nnx.grad(loss_B)(model)
    
    λ_E_new, λ_B_new = lr_anneal((grads_E, grads_B), (1.0, 1.0), 0.9)
    
    assert jnp.isfinite(λ_E_new), "Real model gradient annealing should work"
    assert jnp.isfinite(λ_B_new), "Real model gradient annealing should work"
    assert λ_E_new > 0 and λ_B_new > 0, "Weights should be positive"


def test_lr_anneal_single_term():
    """Test generalized lr_anneal with a single loss term."""
    grads = ({'w': jnp.array([[1.0, 2.0]])},)
    lambdas = (2.0,)

    updated = lr_anneal(grads, lambdas, 0.9)

    assert len(updated) == 1, "Single-term annealing should return one weight"
    assert jnp.isfinite(updated[0]), "Single updated weight should be finite"
    assert updated[0] > 0, "Single updated weight should remain positive"


def test_lr_anneal_generalized_multi_term():
    """Test generalized lr_anneal with more than two loss terms."""
    grads = (
        {'w': jnp.array([[10.0, 20.0]])},
        {'w': jnp.array([[1.0, 2.0]])},
        {'w': jnp.array([[0.1, 0.2]])},
    )
    lambdas = (1.0, 1.0, 1.0)

    updated = lr_anneal(grads, lambdas, 0.5)

    assert len(updated) == 3, "Should return one updated weight per gradient term"
    assert all(jnp.isfinite(val) for val in updated), "Updated weights should be finite"
    assert all(val > 0 for val in updated), "Updated weights should remain positive"
    assert updated[2] > updated[1] > updated[0], "Smaller gradient norms should receive larger weights"


def test_lr_anneal_four_terms():
    """Test generalized lr_anneal with four loss terms."""
    grads = (
        {'w': jnp.array([[100.0, 200.0]])},
        {'w': jnp.array([[10.0, 20.0]])},
        {'w': jnp.array([[1.0, 2.0]])},
        {'w': jnp.array([[0.1, 0.2]])},
    )
    lambdas = (1.0, 1.0, 1.0, 1.0)

    updated = lr_anneal(grads, lambdas, 0.5)

    assert len(updated) == 4, "Should return one updated weight per gradient term"
    assert all(jnp.isfinite(val) for val in updated), "Updated weights should be finite"
    assert all(val > 0 for val in updated), "Updated weights should remain positive"
    assert updated[3] > updated[2] > updated[1] > updated[0], "Weights should inversely track gradient norm magnitude"


def test_lr_anneal_generalized_length_mismatch():
    """Test generalized lr_anneal input validation."""
    grads = ({'w': jnp.array([[1.0]])}, {'w': jnp.array([[2.0]])})
    lambdas = (1.0,)

    with pytest.raises(ValueError, match="same number of terms"):
        lr_anneal(grads, lambdas, 0.9)
