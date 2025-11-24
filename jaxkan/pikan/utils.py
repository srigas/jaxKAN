import jax.numpy as jnp


def model_eval(model, coords, refsol):
    """
    Compute the relative L2 error between model predictions and reference solution.
    
    Args:
        model (nnx.Module):
            Flax model instance.
        coords (jnp.array):
            Input coordinates for evaluation, shape (N, n_dims).
        refsol (jnp.array):
            Reference solution values for comparison.
    
    Returns:
        l2err (float):
            Relative L2 error: ||prediction - reference|| / ||reference||
    
    Example:
        >>> model = KAN([2,8,1], 'spline', {}, 42)
        >>> coords = jnp.array([[0.5, 0.5], [0.1, 0.2]])
        >>> refsol = jnp.array([[0.25], [0.02]])
        >>> error = model_eval(model, coords, refsol)
    """
    
    output = model(coords).reshape(refsol.shape)
    l2err = jnp.linalg.norm(output-refsol)/jnp.linalg.norm(refsol)

    return l2err
