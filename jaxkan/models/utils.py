import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx


class PeriodEmbedder(nnx.Module):
    """
    Periodic embedding module that applies trigonometric transformations to specified input axes.

    Attributes:
        axes (nnx.Dict):
            Dictionary storing period values for each axis. Values can be trainable (nnx.Param) or fixed.
    """

    def __init__(self, period_axes: dict):
        """
        Initializes a PeriodEmbedder module.

        Args:
            period_axes (dict):
                Dictionary mapping input axis indices to (period, trainable) tuples.
                The key is the axis index (int), and the value is a tuple where:
                - period (float): The period value for the trigonometric transformation.
                - trainable (bool): If True, period is stored as nnx.Param and can be optimized during training.
                
        Example:
            >>> # Fixed period on axis 0, trainable period on axis 1
            >>> period_axes = {0: (2.0 * jnp.pi, False), 1: (jnp.pi, True)}
            >>> embedder = PeriodEmbedder(period_axes)
        """
        self.axes = nnx.Dict()
        
        for axis, (period_value, trainable) in period_axes.items():
            # Convert axis to string for nnx.Dict compatibility
            key = str(axis)
            if trainable:
                # Store as trainable parameter
                setattr(self.axes, key, nnx.Param(jnp.array(period_value)))
            else:
                # Store as regular value
                setattr(self.axes, key, period_value)
    

    def __call__(self, x):
        """
        Applies periodic embedding to the input.

        Args:
            x (jnp.array):
                Input array, shape (batch, n_in).

        Returns:
            y (jnp.array):
                Embedded output. For each axis with a period, the original feature is replaced
                by cos(period * x) and sin(period * x). Non-periodic axes are passed through unchanged.
                Shape (batch, n_out) where n_out depends on the number of periodic axes.
                
        Example:
            >>> period_axes = {1: (jnp.pi, False)}
            >>> embedder = PeriodEmbedder(period_axes)
            >>> x = jnp.array([[1.0, 0.5], [2.0, 1.0]])
            >>> y = embedder(x)  # Shape: (2, 3) - axis 0 unchanged, axis 1 → [cos, sin]
        """
        y = []
        
        for idx in range(x.shape[-1]):
            key = str(idx)
            if hasattr(self.axes, key):
                period = getattr(self.axes, key)
                    
                cs = jnp.cos(period * x[:, [idx]])
                ss = jnp.sin(period * x[:, [idx]])
                y.extend([cs, ss])
            else:
                y.append(x[:, [idx]])

        y = jnp.hstack(y)

        return y


class RFFEmbedder(nnx.Module):
    """
    Random Fourier Features (RFF) embedding module for nonlinear feature transformation.

    Attributes:
        B (nnx.Param):
            Random projection matrix, shape (n_in, embed_dim//2).
    """

    def __init__(self, std: float = 1.0, n_in: int = 1, embed_dim: int = 256, seed: int = 42):
        """
        Initializes a RFFEmbedder module.

        Args:
            std (float):
                Standard deviation for the normal distribution used to initialize the random projection matrix.
            n_in (int):
                Input dimension.
            embed_dim (int):
                Output embedding dimension. Must be even (actual dimension used is embed_dim//2 for the random matrix).
            seed (int):
                Random seed for reproducible initialization.
                
        Example:
            >>> embedder = RFFEmbedder(std=1.0, n_in=2, embed_dim=256, seed=42)
        """
        rngs = nnx.Rngs(seed)

        # Initialize kernel
        self.B = nnx.Param(nnx.initializers.normal(stddev=std)(
                            rngs.params(), (n_in, embed_dim//2), jnp.float32))
    
    def __call__(self, x):
        """
        Applies Random Fourier Features transformation to the input.

        Args:
            x (jnp.array):
                Input array, shape (batch, n_in).

        Returns:
            y (jnp.array):
                Embedded output using random Fourier features: [cos(xB), sin(xB)].
                Shape (batch, embed_dim).
                
        Example:
            >>> embedder = RFFEmbedder(std=1.0, n_in=2, embed_dim=256, seed=42)
            >>> x = jnp.array([[1.0, 0.5], [2.0, 1.0]])
            >>> y = embedder(x)  # Shape: (2, 256)
        """
        Bx = jnp.dot(x, self.B[...])
        
        y = jnp.concatenate([jnp.cos(Bx), jnp.sin(Bx)], axis=-1)

        return y


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
        >>> pde_collocs = jnp.array([[0.5, 0.3], [0.2, 0.7]])
        >>> bc_collocs = jnp.array([[0.0, 0.5]])
        >>> complexity = get_complexity(model, pde_collocs, bc_collocs)
    """
    if bc_collocs is not None:
        combined = jnp.concatenate([pde_collocs, bc_collocs], axis=0)
    else:
        combined = pde_collocs
    
    complexity = jnp.mean(batched_frob(model, combined))
    
    return complexity