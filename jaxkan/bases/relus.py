import jax
import jax.numpy as jnp

from functools import partial

import flax.linen as nn

@jax.jit
def get_R_basis(x_ext, S, E, r):
    '''
        Calculates the B-spline basis functions for given grid (augmented knot vector)
        and applies them to the extended input x, batched.
        
        Args:
        -----
            x_ext (jnp.array): extended inputs
                shape (n_in*n_out, batch)
            S (jnp.array): S matrix (where the R functions start)
                shape (n_in*n_out, G+1)
            E (jnp.array): E matrix (where the R functions end)
                shape (n_in*n_out, G+1)
            r (jnp.array): r matrix (for normalization)
                shape (n_in*n_out, G+1)
        
        Returns:
        --------
            R_basis (jnp.array): R basis functions applied on inputs
                shape (n_in*n_out, G+1, batch)
    '''
    batch = x_ext.shape[-1]

    # Expand x to shape (n_in*n_out, G+1, batch)
    x_ext = jnp.expand_dims(x_ext, axis=1)
    x_ext = jnp.repeat(x_ext, E.shape[1], axis=1)
    
    # Expand S, E and r
    S_ext = jnp.expand_dims(S, axis=2)
    S_ext = jnp.repeat(S_ext, batch, axis=2)
    E_ext = jnp.expand_dims(E, axis=2)
    E_ext = jnp.repeat(E_ext, batch, axis=2)
    r_ext = jnp.expand_dims(r, axis=2)
    r_ext = jnp.repeat(r_ext, batch, axis=2)

    # Now all items are (n_in*n_out, G+1, batch)-shaped so we can perform operations
    A = nn.relu(E_ext - x_ext)
    B = nn.relu(x_ext - S_ext)
    D = r_ext*A*B
    R_basis = D*D

    return R_basis

@partial(jax.jit, static_argnums=(2,))
def augment_grid(grid, k, p):
    """
        Expands the grid to its left and right by adding p points at each side, depending on the values
        of their nearest k neighbours.
        
        Args:
        -----
            grid (jnp.array): grid
                shape (n_in*n_out, G+1)
            k (int): number of neighbours to take into account
            p (int): number of points to add at each side
        
        Returns:
        --------
            ext_grid (jnp.array): Grid extended by p points at each side
                shape (n_in*n_out, G+2p+1)
    """
    ext_grid = grid
    
    for _ in range(p):
        # Calculate start point using its k nearest neighbors
        start_p = ext_grid[:, 0] - (1/k) * (ext_grid[:, k] - ext_grid[:, 0])
        # Cast to shape (n_in*n_out, 1)
        start_point = jnp.expand_dims(start_p, axis=-1)
        
        # Calculate end point using its k nearest neighbors
        end_p = ext_grid[:, -1] + (1/k) * (ext_grid[:, -1] - ext_grid[:, -k-1])
        # Cast to shape (n_in*n_out, 1)
        end_point = jnp.expand_dims(end_p, axis=-1)
        
        # Add zero point and end point to the grid
        ext_grid = jnp.concatenate((start_point, ext_grid, end_point), axis=1)

    return ext_grid
