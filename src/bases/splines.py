import jax.numpy as jnp

from jax import jit
from functools import partial

# Partial applies the jit decorator with static arguments, i.e. which should be kept constant for compilation,
# but would require a re-compilation if its value changes. We don't expect k to change throughout a single run.
@partial(jit, static_argnums=(2,))
def get_spline_basis(x_ext, grid, k=3):
    '''
        Calculates the B-spline basis functions for given grid (augmented knot vector)
        and applies them to the extended input x, batched.
        
        Args:
        -----
            x_ext (jnp.array): extended inputs
                shape (n_in*n_out, batch)
            grid (jnp.array): augmented grid
                shape (n_in*n_out, G+2k+1)
            k (int): order of the B-spline basis functions. Default: 3
        
        Returns:
        --------
            basis_splines (jnp.array): spline basis functions applied on inputs
                shape (n_in*n_out, G+k, batch)
    '''

    # Broadcasting for vectorized operations
    grid = jnp.expand_dims(grid, axis=2)
    x = jnp.expand_dims(x_ext, axis=1)

    # k = 0 case
    basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
    
    # Recursion done through iteration
    for K in range(1, k+1):
        left_term = (x - grid[:, :-(K + 1)]) / (grid[:, K:-1] - grid[:, :-(K + 1)])
        right_term = (grid[:, K + 1:] - x) / (grid[:, K + 1:] - grid[:, 1:(-K)])
        
        basis_splines = left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]

    return basis_splines
