import jax
import jax.numpy as jnp

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
def lr_anneal(grads_E, grads_B, λ_E, λ_B, grad_mixing):
    """
    Perform the learning rate annealing algorithm introduced in "Understanding and mitigating gradient pathologies in physics-informed neural networks"
    
    Args:
        grads_E (pytree):
            Gradients from equation/PDE loss.
        grads_B (pytree):
            Gradients from boundary/initial condition loss.
        λ_E (float):
            Current equation loss weight.
        λ_B (float):
            Current boundary loss weight.
        grad_mixing (float):
            Mixing coefficient for exponential moving average (0 to 1).
    
    Returns:
        λ_E_new (float):
            Updated pde loss weight.
        λ_B_new (float):
            Updated boundary loss weight.
    
    Example:
        >>> λ_E_new, λ_B_new = lr_anneal(grads_E, grads_B, λ_E=1.0, λ_B=1.0, grad_mixing=0.9)
    """
    
    norm_E = optax.global_norm(grads_E)
    norm_B = optax.global_norm(grads_B)
    norm_sum = norm_E + norm_B
                
    λ_E_hat = norm_sum / (norm_E + 1e-5*norm_sum)
    λ_B_hat = norm_sum / (norm_B + 1e-5*norm_sum)
                        
    λ_E_new = grad_mixing*λ_E + (1.0 - grad_mixing)*λ_E_hat
    λ_B_new = grad_mixing*λ_B + (1.0 - grad_mixing)*λ_B_hat

    return λ_E_new, λ_B_new

