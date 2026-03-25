import jax
import jax.numpy as jnp

from flax import nnx

import optax


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

