import jax.numpy as jnp

from functools import partial
from jax import jit

# Partial applies the jit decorator with static arguments, i.e. which should be kept constant for compilation,
# but would require a re-compilation if its value changes. We don't expect k to change throughout a single run.
@partial(jit, static_argnums=(2,))
def get_spline_basis(x, grid, k=3):
    '''
        Calculates the B-spline basis functions for given grid (augmented knot vector)
        and applies them to the input x, batched.
        
        Args:
            x : inputs
                shape (batch_size, layer_size)
            grid : augmented grid
                shape (layer_size, G + 2k + 1)
            k : order of the B-spline basis functions. Default: 3
        
        Returns:
            splines : spline basis functions applied on input
                shape (batch_size, layer_size, G + k)
    '''

    # Broadcasting for vectorized operations
    x = jnp.expand_dims(x, axis=-1)

    # k = 0 case
    splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
    
    # Recursion done through iteration
    for K in range(1, k+1):
        left_term = (x - grid[:, :-(K + 1)]) / (grid[:, K:-1] - grid[:, :-(K + 1)])
        right_term = (grid[:, K + 1:] - x) / (grid[:, K + 1:] - grid[:, 1:(-K)])
        
        splines = left_term * splines[:, :, :-1] + right_term * splines[:, :, 1:]

    return splines

@jit
def get_coeffs(new_basis, old_spline):

    return