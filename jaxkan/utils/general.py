import jax
import jax.numpy as jnp


@jax.jit
def solve_single_lstsq(A_single, B_single):
    """
    Simulates linalg.lstsq by reformulating the problem AX = B via the normal equations: (A^T A) X = A^T B
    This is used instead of linalg.lstsq because it's much faster.

    Args:
    -----
        A_single (jnp.array): matrix A of AX = B
            shape (M, N)
        B_single (jnp.array): matrix B of AX = B
            shape (M, K)
        
    Returns:
    --------
        single_solution (jnp.array): matrix X of AX = B
            shape (N, K)
    """
    AtA = jnp.dot(A_single.T, A_single)
    AtB = jnp.dot(A_single.T, B_single)
    single_solution = jax.scipy.linalg.solve(AtA, AtB, assume_a='pos')
    
    return single_solution


@jax.jit
def solve_full_lstsq(A_full, B_full):
    """
    Parallelizes the single case, so that the problem can be solved for matrices with dimensions higher than 2.
    Essentially, solve_single_lstsq and solve_full_lstsq combined are a workaround, because (unlike PyTorch for example), JAX's libraries do not support lstsq for dims > 2.

    Args:
    -----
        A_full (jnp.array): matrix A of AX = B
            shape (batch, M, N)
        B_full (jnp.array): matrix B of AX = B
            shape (batch, M, K)
        
    Returns:
    --------
        full_solution (jnp.array): matrix X of AX = B
            shape (batch, N, K)
    """
    # Define the solver for (*, ., .) dimensions
    solve_full = jax.vmap(solve_single_lstsq, in_axes=(0,0))
    # Apply it to get back the coefficients
    full_solution = solve_full(A_full, B_full)

    return full_solution
    
    
def interpolate_moments(mu_old, nu_old, new_shape):
    '''
        Performs a linear interpolation to assign values to the first and second-order moments of gradients
        of the c_i basis functions coefficients after grid extension
        
        Args:
        -----
            mu_old (jnp.array): first-order moments before extension
                shape (n_in*n_out, num_basis)
            nu_old (jnp.array): second-order moments before extension
                shape (n_in*n_out, num_basis)
            new shape (tuple): (n_in*n_out, new_num_basis)
        
        Returns:
        --------
            mu_new (jnp.array): first-order moments after extension
                shape (n_in*n_out, new_num_basis)
            nu_new (jnp.array): second-order moments after extension
                shape (n_in*n_out, new_num_basis)
    '''
    old_shape = mu_old.shape
    size = old_shape[0]
    old_j = old_shape[1]
    new_j = new_shape[1]
    
    # Create new indices for the second dimension
    old_indices = jnp.linspace(0, old_j - 1, old_j)
    new_indices = jnp.linspace(0, old_j - 1, new_j)

    # Vectorize the interpolation function for use with vmap
    interpolate_fn = lambda old_row: jnp.interp(new_indices, old_indices, old_row)

    # Apply the interpolation function to each row using vmap
    mu_new = jax.vmap(interpolate_fn)(mu_old)
    nu_new = jax.vmap(interpolate_fn)(nu_old)
    
    return mu_new, nu_new


def adam_transition(old_state, model_state):
    '''
        Performs the state transition for the Adam optimizer with scheduler after grid extension
        Note that the transition happens in-place, i.e. nothing is returned, the optimizer is simply
        transitioned from the old state to the new
        
        Args:
        -----
            old_state (tuple): collection of adam state and scheduler state before extension
            model_state (dict): model_state dict of KAN model
    '''
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