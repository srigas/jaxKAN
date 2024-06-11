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