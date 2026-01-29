import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

import optax


def get_activation(activation: str = 'tanh'):
    """
    Returns the corresponding activation function based on user input.

    Args:
        activation (str):
            Name of the activation function. Options include:
            - 'celu': Continuously Differentiable ELU
            - 'elu': Exponential Linear Unit
            - 'gelu': Gaussian Error Linear Unit
            - 'hard_sigmoid': Hard sigmoid
            - 'hard_silu' / 'hard_swish': Hard SiLU
            - 'hard_tanh': Hard hyperbolic tangent
            - 'identity': Identity function (no activation)
            - 'leaky_relu': Leaky ReLU
            - 'log_sigmoid': Log-sigmoid function
            - 'relu': Rectified Linear Unit
            - 'selu': Scaled ELU
            - 'sigmoid': Sigmoid function
            - 'silu' / 'swish': Sigmoid Linear Unit
            - 'soft_sign': Soft sign function
            - 'softplus': Softplus function
            - 'tanh': Hyperbolic tangent (default)

    Returns:
        callable:
            The activation function.
            
    Example:
        >>> act_fn = get_activation('tanh')
        >>> y = act_fn(x)
    """
    activation = activation.lower()
    
    activation_map = {
        'celu': nnx.celu,
        'elu': nnx.elu,
        'gelu': nnx.gelu,
        'hard_sigmoid': nnx.hard_sigmoid,
        'hard_silu': nnx.hard_silu,
        'hard_swish': nnx.hard_silu,
        'hard_tanh': nnx.hard_tanh,
        'leaky_relu': nnx.leaky_relu,
        'log_sigmoid': nnx.log_sigmoid,
        'relu': nnx.relu,
        'selu': nnx.selu,
        'sigmoid': nnx.sigmoid,
        'identity': nnx.identity,
        'silu': nnx.silu,
        'soft_sign': nnx.soft_sign,
        'softplus': nnx.softplus,
        'swish': nnx.swish,
        'tanh': nnx.tanh
    }
    
    if activation not in activation_map:
        raise ValueError(f"Unknown activation: {activation}. Available: {list(activation_map.keys())}")
    
    return activation_map[activation]


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


def get_adam(
    learning_rate: float = 1e-3,
    schedule_type: str = None,
    decay_steps: int = 5000,
    decay_rate: float = 0.9,
    warmup_steps: int = 0,
    staircase: bool = False,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    **schedule_kwargs
):
    """
    Create an Adam optimizer with optional learning rate scheduling and warmup.
    
    Args:
        learning_rate (float):
            Base learning rate. Default is 1e-3.
            
        schedule_type (str, optional):
            Type of learning rate schedule. Options:
            - None: Constant learning rate (default)
            - 'exponential': Exponential decay schedule
            - 'cosine': Cosine annealing schedule
            - 'polynomial': Polynomial decay schedule
            - 'piecewise_constant': Piecewise constant schedule (requires 'boundaries' and 'values' in schedule_kwargs)
            
        decay_steps (int):
            Number of steps for the learning rate decay schedule. Default is 5000.
            Used for exponential, cosine, and polynomial schedules.
            
        decay_rate (float):
            Decay rate for exponential schedule. Default is 0.9.
            For polynomial schedule, this is the 'power' parameter.
            
        warmup_steps (int):
            Number of warmup steps with linear learning rate increase from 0 to learning_rate.
            Default is 0 (no warmup).
            
        staircase (bool):
            If True, decay the learning rate at discrete intervals (staircase function).
            Default is False (smooth decay).
            
        b1 (float):
            Exponential decay rate for first moment. Default is 0.9.
            
        b2 (float):
            Exponential decay rate for second moment. Default is 0.999.
            
        eps (float):
            Small constant for numerical stability. Default is 1e-8.
            
        **schedule_kwargs:
            Additional keyword arguments for specific schedules.
            For piecewise_constant schedule:
                - boundaries (list): List of step boundaries
                - values (list): List of learning rate values (must be len(boundaries) + 1)
    
    Returns:
        optax.GradientTransformation:
            Configured Adam optimizer with learning rate schedule.
    
    Example:
        >>> # Adam with exponential decay and warmup
        >>> optimizer = get_adam(
        ...     learning_rate=1e-3,
        ...     schedule_type='exponential',
        ...     decay_steps=5000,
        ...     decay_rate=0.9,
        ...     warmup_steps=1000,
        ...     b1=0.9,
        ...     b2=0.999
        ... )
        
        >>> # Adam with cosine annealing
        >>> optimizer = get_adam(
        ...     learning_rate=1e-3,
        ...     schedule_type='cosine',
        ...     decay_steps=10000,
        ...     warmup_steps=500
        ... )
        
        >>> # Adam with constant learning rate
        >>> optimizer = get_adam(learning_rate=1e-3)
    """
    import optax
    
    # Create learning rate schedule
    if schedule_type is None:
        # Constant learning rate
        lr_schedule = learning_rate
        
    elif schedule_type == 'exponential':
        # Exponential decay: lr * decay_rate^(step/decay_steps)
        lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase
        )
        
    elif schedule_type == 'cosine':
        # Cosine annealing schedule
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=decay_steps,
            alpha=0.0  # Minimum learning rate as fraction of init_value
        )
        
    elif schedule_type == 'polynomial':
        # Polynomial decay schedule
        lr_schedule = optax.polynomial_schedule(
            init_value=learning_rate,
            end_value=learning_rate * 0.01,  # Decay to 1% of initial value
            power=decay_rate,  # Using decay_rate as the power parameter
            transition_steps=decay_steps
        )
        
    elif schedule_type == 'piecewise_constant':
        # Piecewise constant schedule
        if 'boundaries' not in schedule_kwargs or 'values' not in schedule_kwargs:
            raise ValueError("piecewise_constant schedule requires 'boundaries' and 'values' in schedule_kwargs")
        boundaries = schedule_kwargs.pop('boundaries')
        values = schedule_kwargs.pop('values')
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=values[0],
            boundaries_and_scales={b: v / values[0] for b, v in zip(boundaries, values[1:])}
        )
        
    else:
        raise ValueError(
            f"Unknown schedule_type '{schedule_type}'. "
            f"Options: None, 'exponential', 'cosine', 'polynomial', 'piecewise_constant'"
        )
    
    # Add warmup if requested
    if warmup_steps > 0:
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=warmup_steps
        )
        
        # Join warmup and main schedule
        lr_schedule = optax.join_schedules(
            schedules=[warmup_schedule, lr_schedule],
            boundaries=[warmup_steps]
        )
    
    # Check for any remaining unused kwargs
    if schedule_kwargs:
        print(f"Warning: Unused schedule kwargs: {list(schedule_kwargs.keys())}")
    
    # Create Adam optimizer
    tx = optax.adam(
        learning_rate=lr_schedule,
        b1=b1,
        b2=b2,
        eps=eps
    )
    
    return tx


def get_lbfgs(
    learning_rate: float = None,
    memory_size: int = 10,
    scale_init_precond: bool = True,
    linesearch: any = None
):
    """
    Create an L-BFGS optimizer.
    
    Note: L-BFGS requires special handling when used with Flax NNX. You must pass
    `value`, `value_fn`, and `model` to the optimizer's update method. The `value_fn` 
    should be a function that takes the model and returns the loss value.
    
    Args:
        learning_rate (float, optional):
            Initial learning rate. If None, the optimizer uses its own line search
            to determine the step size. Default is None.
            
        memory_size (int):
            Number of past updates to keep in memory to approximate the Hessian inverse.
            Larger values require more memory but may lead to better convergence.
            Default is 10.
            
        scale_init_precond (bool):
            Whether to use a scaled identity as the initial preconditioner.
            Default is True.
            
        linesearch (optax.GradientTransformation, optional):
            Custom line search transformation. If None, uses the default zoom line search.
            Default is None.
    
    Returns:
        optax.GradientTransformationExtraArgs:
            Configured L-BFGS optimizer.
    """
    
    tx = optax.lbfgs(
        learning_rate=learning_rate,
        memory_size=memory_size,
        scale_init_precond=scale_init_precond,
        linesearch=linesearch
    )
    
    return tx