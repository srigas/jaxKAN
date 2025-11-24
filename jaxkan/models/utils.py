import jax
import jax.numpy as jnp

import nnx


def count_params(model):
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model (nnx.Module):
            Flax model instance.
    
    Returns:
        total_params (int):
            Total number of trainable parameters in the model.
    
    Example:
        >>> model = KAN([2,8,8,1], 'spline', {'k': 4, 'G': 3}, 42)
        >>> num_params = count_params(model)
        >>> print(f"Model has {num_params} parameters")
    """
    # Extract all parameters of type nnx.Param from the model.
    params = nnx.state(model, nnx.Param)
    
    # Flatten the tree to get individual parameter arrays.
    leaves = jax.tree_util.tree_leaves(params)
    
    # Sum the total number of elements (i.e. the product of the dimensions)
    total_params = sum(np.prod(p.shape) for p in leaves)

    return int(total_params)


def get_frob(model, x):
    """
    Compute the squared Frobenius norm of the model's gradient at a given input point.
    
    Args:
        model (nnx.Module):
            Flax model instance.
        x (jnp.array):
            Input point, shape (d,) or (1, d).
    
    Returns:
        fro_sq (float):
            Squared Frobenius norm of the gradient ||∇f(x)||²_F.
    
    Example:
        >>> model = KAN([2,8,1], 'spline', {}, 42)
        >>> x = jnp.array([0.5, 0.3])
        >>> frob_norm_sq = get_frob(model, x)
    """

    # normalize x to (1, d) for the model
    x = x[None, :] if x.ndim == 1 else x

    def u(t):
        y = model(t).flatten()
        return y[0]
    
    g = jax.grad(u)(x)
    fro_sq = jnp.vdot(g, g)
    
    return fro_sq


# Vectorized version of get_frob for batch processing
batched_frob = nnx.jit(jax.vmap(get_frob, in_axes=(None, 0)))


def get_complexity(model, pde_collocs, bc_collocs=None):
    """
    Compute model complexity as the average squared Frobenius norm of gradients over collocation points.
    
    Args:
        model (nnx.Module):
            Flax model instance.
        pde_collocs (jnp.array):
            Collocation points for PDE/equation domain, shape (N, d).
        bc_collocs (jnp.array, optional):
            Initial/boundary condition collocation points, shape (M, d). If None, only use collocs.
    
    Returns:
        complexity (float):
            Average squared Frobenius norm of gradients: mean(||∇f(x)||²_F).
    
    Example:
        >>> model = KAN([2,8,1], 'spline', {}, 42)
        >>> collocs = jnp.array([[0.5, 0.3], [0.2, 0.7]])
        >>> ic_collocs = jnp.array([[0.0, 0.5]])
        >>> complexity = get_complexity(model, collocs, ic_collocs)
    """
    if ic_collocs is not None:
        combined = jnp.concatenate([collocs, ic_collocs], axis=0)
    else:
        combined = collocs
    
    complexity = jnp.mean(batched_frob(model, combined))
    
    return complexity