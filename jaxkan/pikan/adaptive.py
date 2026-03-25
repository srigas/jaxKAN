import jax
import jax.numpy as jnp

from flax import nnx

import optax


DEFAULT_EPS = 1e-12


def get_colloc_indices(collocs_pool, batch_size, px, seed):
    """
    Sample collocation point indices from a pool based on probability weights and sort by first coordinate (time).
    
    Args:
        collocs_pool (jnp.array):
            Pool of collocation points, shape (N, n_dims).
        batch_size (int):
            Number of points to sample.
        px (jnp.array):
            Probability weights for sampling, shape (N,).
        seed (int):
            Random seed for reproducibility.
    
    Returns:
        sorted_pool_indices (jnp.array):
            Indices of sampled points sorted by first coordinate, shape (batch_size,).
    
    Example:
        >>> collocs_pool = jnp.array([[0.5, 0.1], [0.2, 0.3], [0.8, 0.7]])
        >>> px = jnp.array([0.5, 0.3, 0.2])
        >>> indices = get_colloc_indices(collocs_pool, batch_size=2, px=px, seed=42)
    """
    
    collocs_key = jax.random.PRNGKey(seed)

    X_ids = jax.random.choice(key=collocs_key, a=collocs_pool.shape[0], shape=(batch_size,), replace=False, p=px)

    sorted_batch_order = jnp.argsort(collocs_pool[X_ids, 0])
    
    sorted_pool_indices = X_ids[sorted_batch_order]

    return sorted_pool_indices


@nnx.jit
def update_rba_weights(residuals, weights, gamma, eta, eps=DEFAULT_EPS):
    """
    Update residual-based attention weights from the current residual magnitudes.

    Args:
        residuals (jnp.array):
            Residual values, typically shape (N, 1).
        weights (jnp.array):
            Current RBA weights with the same shape as ``residuals``.
        gamma (float):
            Exponential moving average coefficient.
        eta (float):
            Residual scaling coefficient.
        eps (float):
            Numerical stabilizer for normalization.

    Returns:
        jnp.array:
            Updated RBA weights.
    """

    abs_res = jnp.abs(residuals)
    scale = jnp.max(abs_res)
    return (gamma * weights) + (eta * abs_res / (scale + eps))


@nnx.jit
def apply_rba_weights(residuals, weights):
    """
    Apply RBA weights to residuals while keeping the weights outside the backward graph.

    Args:
        residuals (jnp.array):
            Residual values.
        weights (jnp.array):
            RBA weights with matching shape.

    Returns:
        jnp.array:
            Weighted residuals.
    """

    return jax.lax.stop_gradient(weights) * residuals


@nnx.jit
def get_causal_weights(losses, causal_matrix, causal_tol):
    """
    Compute causal training weights and stop their gradients.

    Args:
        losses (jnp.array):
            Chunk-wise loss values.
        causal_matrix (jnp.array):
            Upper-triangular causal coupling matrix.
        causal_tol (float):
            Exponential weighting coefficient.

    Returns:
        jnp.array:
            Stopped-gradient causal weights.
    """

    return jax.lax.stop_gradient(jnp.exp(-causal_tol * (causal_matrix @ losses)))


@nnx.jit
def get_rad_probabilities(weighted_residuals, rad_a, rad_c, eps=DEFAULT_EPS):
    """
    Convert weighted residuals into normalized RAD sampling probabilities.

    Args:
        weighted_residuals (jnp.array):
            Residuals already multiplied by their current adaptive weights.
        rad_a (float):
            Density-control exponent.
        rad_c (float):
            Baseline offset added to the density.
        eps (float):
            Numerical stabilizer for normalization.

    Returns:
        jnp.array:
            Probability vector of shape (N,).
    """

    abs_weighted = jnp.abs(weighted_residuals)
    density = jnp.power(abs_weighted, rad_a)
    probs = (density / (jnp.mean(density) + eps)) + rad_c
    return (probs / jnp.sum(probs))[:, 0]


@nnx.jit(static_argnames=("batch_size",))
def get_rad_indices(collocs_pool, residuals, old_indices, batch_weights, pool_weights, batch_size, rad_a, rad_c, seed, eps=DEFAULT_EPS):
    """
    Update pool weights and draw a new RAD-resampled batch of collocation indices.

    Args:
        collocs_pool (jnp.array):
            Full collocation pool, shape (N, d).
        residuals (jnp.array):
            Residual values over the full pool, shape (N, 1).
        old_indices (jnp.array):
            Previously active batch indices.
        batch_weights (jnp.array):
            Current adaptive weights for the active batch.
        pool_weights (jnp.array):
            Full pool of adaptive weights.
        batch_size (int):
            Number of indices to sample.
        rad_a (float):
            Density-control exponent.
        rad_c (float):
            Baseline offset added to the density.
        seed (int):
            Sampling seed.
        eps (float):
            Numerical stabilizer.

    Returns:
        tuple[jnp.array, jnp.array, jnp.array]:
            Sampled indices, updated full-pool weights, and normalized probabilities.
    """

    updated_pool = pool_weights.at[old_indices].set(batch_weights)
    weighted_residuals = updated_pool * residuals
    px = get_rad_probabilities(weighted_residuals, rad_a, rad_c, eps)
    sorted_indices = get_colloc_indices(collocs_pool=collocs_pool, batch_size=batch_size, px=px, seed=seed)
    return sorted_indices, updated_pool, px


@nnx.jit
def _lr_anneal_impl(grads, lambdas, grad_mixing):
    norms = jnp.stack(tuple(optax.tree.norm(grad) for grad in grads))
    norm_sum = jnp.sum(norms)
    hats = norm_sum / (norms + 1e-5 * norm_sum)

    return tuple(
        grad_mixing * lambdas[i] + (1.0 - grad_mixing) * hats[i]
        for i in range(len(lambdas))
    )


def lr_anneal(grads, lambdas, grad_mixing):
    """
    Perform the learning rate annealing algorithm introduced in "Understanding and mitigating gradient pathologies in physics-informed neural networks"

    Args:
        grads (tuple | list):
            Tuple/list of gradient pytrees, one per loss term.
        lambdas (tuple | list):
            Tuple/list of current global loss weights, matching ``grads`` in order.
        grad_mixing (float):
            Exponential moving average coefficient.

    Returns:
        tuple:
            Updated global loss weights in the same logical order as the inputs.

    Example:
        >>> λs_new = lr_anneal((grads_E, grads_B, grads_aux), (1.0, 1.0, 1.0), 0.9)
    """

    grads = tuple(grads)
    lambdas = tuple(lambdas)

    if len(grads) != len(lambdas):
        raise ValueError("grads and lambdas must have the same number of terms.")

    if len(grads) == 0:
        raise ValueError("lr_anneal requires at least one gradient/weight pair.")

    return _lr_anneal_impl(grads, lambdas, grad_mixing)

