import jax
import jax.numpy as jnp


@jax.jit
def solve_single_lstsq(A_single, B_single):
    """
    Simulates linalg.lstsq by reformulating the problem AX = B via the normal equations: (A^T A) X = A^T B. This is used instead of linalg.lstsq because it's much faster.

    Args:
        A_single (jnp.array):
            Matrix A of AX = B, shape (M, N).
        B_single (jnp.array):
            Matrix B of AX = B, shape (M, K).
        
    Returns:
        single_solution (jnp.array):
            Matrix X of AX = B, shape (N, K).
            
    Example:
        >>> A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        >>> B = jnp.array([[1.0], [2.0]])
        >>>
        >>> solution = solve_single_lstsq(A, B)
    """
    
    AtA = jnp.dot(A_single.T, A_single)
    AtB = jnp.dot(A_single.T, B_single)
    single_solution = jax.scipy.linalg.solve(AtA, AtB, assume_a='pos')
    
    return single_solution


@jax.jit
def solve_full_lstsq(A_full, B_full):
    """
    Parallelizes the single case, so that the problem can be solved for matrices with dimensions higher than 2. Essentially, solve_single_lstsq and solve_full_lstsq combined are a workaround, because (unlike PyTorch for example), JAX's libraries do not support lstsq for dims > 2.

    Args:
        A_full (jnp.array):
            Matrix A of AX = B, shape (batch, M, N).
        B_full (jnp.array):
            Matrix B of AX = B, shape (batch, M, K).
        
    Returns:
        full_solution (jnp.array):
            Matrix X of AX = B, shape (batch, N, K).
            
    Example:
        >>> A = jnp.array([[[2.0, 1.0], [1.0, 3.0]], [[1.0, 2.0], [2.0, 1.0]]])
        >>> B = jnp.array([[[1.0], [2.0]], [[2.0], [3.0]]])
        >>>
        >>> solution = solve_full_lstsq(A, B)
    """
    
    # Define the solver for (*, ., .) dimensions
    solve_full = jax.vmap(solve_single_lstsq, in_axes=(0,0))
    # Apply it to get back the coefficients
    full_solution = solve_full(A_full, B_full)

    return full_solution
    
    
def interpolate_moments(mu_old, nu_old, new_shape):
    """
    Performs a linear interpolation to assign values to the first and second-order moments of gradients of the c_i basis functions coefficients after grid extension.
    
    Args:
        mu_old (jnp.array):
            First-order moments before extension, shape (n_in*n_out, num_basis) or (n_out, n_in, num_basis).
        nu_old (jnp.array):
            Second-order moments before extension, shape (n_in*n_out, num_basis) or (n_out, n_in, num_basis).
        new_shape (tuple):
            The new desired shape, either (n_in*n_out, new_num_basis) or (n_out, n_in, new_num_basis).

    Returns:
        mu_new, nu_new (tuple):
            First- and second-order moments after extension, shape new_shape.

    Example:
        >>> mu_old = jnp.array([[1, 2, 3], [4, 5, 6]])
        >>> nu_old = jnp.array([[7, 8, 9], [10, 11, 12]])
        >>> new_shape = (2, 5)
        >>>
        >>> mu_new, nu_new = interpolate_moments(mu_old, nu_old, new_shape)
    """
    
    old_shape = mu_old.shape
    old_j = old_shape[-1] # This is the dimension along which interpolation occurs
    new_j = new_shape[-1]
    
    # At this point, the shape will be either (n_in*n_out, num_basis) if the layer type is 'base'
    # or (n_out, n_in, num_basis) if the layer type is 'spline'
    # So we need a generic approach to handle these two types without control statements
    
    # Flatten all leading dimensions into a single dimension
    batch_size = jnp.prod(jnp.array(old_shape[:-1]))
    # Reshape to (n_in*n_out, num_basis)
    mu_old_2d = mu_old.reshape((batch_size, old_j))
    nu_old_2d = nu_old.reshape((batch_size, old_j))
    
    # Interpolate along the last dimension
    old_indices = jnp.linspace(0, old_j - 1, old_j)
    new_indices = jnp.linspace(0, old_j - 1, new_j)

    # Vectorize the interpolation function for use with vmap
    interpolate_fn = lambda old_row: jnp.interp(new_indices, old_indices, old_row)

    # Apply the interpolation function to each row using vmap
    mu_new_2d = jax.vmap(interpolate_fn)(mu_old_2d)
    nu_new_2d = jax.vmap(interpolate_fn)(nu_old_2d)
    
    # Reshape back to match the original leading dimensions
    # but with updated last dimension = new_j
    mu_new = mu_new_2d.reshape((*old_shape[:-1], new_j))
    nu_new = nu_new_2d.reshape((*old_shape[:-1], new_j))
    
    return mu_new, nu_new


def adam_transition(old_state, model_state):
    """
    Performs the state transition for the Adam optimizer with scheduler after grid extension. Note that the transition happens in-place, i.e. nothing is returned, the optimizer is simply transitioned from the old state to the new.
    
    Args:
        old_state (tuple):
            Collection of Adam state and scheduler state before extension.
        model_state (dict):
            Dict of KAN model state after split.
            
    Example:
        >>> old_state = optimizer.opt_state
        >>> _, model_state = nnx.split(model)
        >>> adam_transition(old_state, model_state)
    """
    
    # Get old state
    adam_mu, adam_nu = old_state[0].mu, old_state[0].nu

    for key in range(len(adam_mu['layers'])):
        # Find the c_basis shape for this layer
        c_shape = model_state['layers'][key]['c_basis'].value.shape
        # Get new mu and nu
        mu_old = adam_mu['layers'][key]['c_basis'].value
        nu_old = adam_nu['layers'][key]['c_basis'].value
        mu_new, nu_new = interpolate_moments(mu_old, nu_old, c_shape)
        # Set them
        adam_mu['layers'][key]['c_basis'].value = mu_new
        adam_nu['layers'][key]['c_basis'].value = nu_new