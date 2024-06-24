from jax import numpy as jnp

from flax import linen as nn
from flax.linen import initializers

from ..bases.relus import get_R_basis, augment_grid
from ..utils.general import solve_full_lstsq


class ReLUKANLayer(nn.Module):
    """
    ReLUKANLayer class.

    Args:
    -----------
        n_in (int): number of layer's incoming nodes. Default: 2
        n_out (int): number of layer's outgoing nodes. Default: 5
        p (int): number of neighbours to take into account during basis functions initialization. Default: 2
        k (int): number of neighbours to take into account during initial grid expansion. Default: 2
        const_R (float/bool): coefficient of the R function in the overall activation. If set to False, then it is trainable per activation. Default: False
        const_res (float/bool): coefficient of the residual function in the overall activation. If set to False, then it is trainable per activation. Default: False
        residual (nn.Module): function that is applied on samples to calculate residual activation. Default: nn.swish
        noise_std (float): noise for the initialization of basis functions coefficients. Default: 0.1
        grid_e (float): parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases. Default: 0.05
    """
    
    n_in: int = 2
    n_out: int = 5
    p: int = 2
    k: int = 2
    
    const_R: float or bool = False
    const_res: float or bool = False
    residual: nn.Module = nn.swish
    
    noise_std: float = 0.1
    grid_e: float = 0.05

    
    def setup(self):
        """
        Registers and initializes all parameters of the ReLUKANLayer. Specifically:
            * E, S, r: non-trainable variables
            * c_basis, c_R, c_res: trainable parameters
        """

        # Initialize a (uniform) grid based on a default value of G and expand for broadcasting
        # the shape becomes (n_in*n_out, G + 1),
        init_G = 3
        h = 2 / init_G
        grid = jnp.arange(0, init_G + 1, dtype=jnp.float32) * h - 1
        grid = jnp.expand_dims(grid, axis=0)
        grid = jnp.tile(grid, (self.n_in*self.n_out, 1))

        # Using the initial grid, obtain the values for the E and S matrices
        S, E, r = self.get_SEr(grid)

        # Store the non trainable variables
        self.S = self.variable('state', 'S', lambda: S)
        self.E = self.variable('state', 'E', lambda: E)
        self.r = self.variable('state', 'r', lambda: r)
        
        # Register & initialize the basis functions' coefficients as trainable parameters
        # They are drawn from a normal distribution with zero mean and an std of noise_std
        self.c_basis = self.param('c_basis', initializers.normal(stddev=self.noise_std), (self.E.value.shape))

        # If const_R is set as a float value, treat it as non trainable and pass it to the c_R array with shape (n_in*n_out)
        # Otherwise register it as a trainable parameter of the same size and initialize it
        if isinstance(self.const_R, float):
            self.c_R = jnp.ones(self.n_in*self.n_out) * self.const_R
        elif self.const_R is False:
            self.c_R = self.param('c_R', initializers.constant(1.0), (self.n_in * self.n_out,))

        # If const_res is set as a float value, treat it as non trainable and pass it to the c_res array with shape (n_in*n_out)
        # Otherwise register it as a trainable parameter of the same size and initialize it
        if isinstance(self.const_res, float):
            self.c_res = jnp.ones(self.n_in * self.n_out) * self.const_res
        elif self.const_res is False:
            self.c_res = self.param('c_res', initializers.constant(1.0), (self.n_in * self.n_out,))
    

    def get_SEr(self, grid):
        """
        Uses the given grid to retrieve the corresponding values of the E and S matrices, as well as the
        r normalization matrix

        Args:
        -----
            grid (jnp.array): current grid
                shape (n_in*n_out, G+1)

        Returns:
        --------
            S (jnp.array): S matrix (where the R functions start)
                shape (n_in*n_out, G+1)
            E (jnp.array): E matrix (where the R functions end)
                shape (n_in*n_out, G+1)
            r (jnp.array): r matrix (for normalization)
                shape (n_in*n_out, G+1)
        """
        # First augment the grid
        ext_grid = augment_grid(grid, self.k, self.p)

        # Find the new grid's center
        grid_center = ext_grid[:, self.p:-self.p]
        # Calculate the values for S and E
        S = grid_center - ((ext_grid[:, 2*self.p:] - ext_grid[:, :-2*self.p]) / 2 )
        E = 2.0 * grid_center - S
        # Also calculate r
        r = 4.0/((E - S)**2)

        return S, E, r
        

    def basis(self, x):
        """
        Uses the current E, S and r to calculate the values of the basis functions on the input.

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in)

        Returns:
        --------
            bases (jnp.array): R basis functions applied on inputs
                shape (n_in*n_out, G+1, batch)
        """
        batch = x.shape[0]
        # Extend to shape (batch, n_in*n_out)
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in * self.n_out))
        # Transpose to shape (n_in*n_out, batch)
        x_ext = jnp.transpose(x_ext, (1, 0))

        # Draw S, E and r
        S = self.S.value
        E = self.E.value
        r = self.r.value

        bases = get_R_basis(x_ext, S, E, r)
        
        return bases

    
    def new_coeffs(self, x, ciRi):
        """
        Utilizes the new grid's basis functions, Rj(x), and the old grid's basis functions, Sum(ciRix(x)), to approximate a good initialization for the updated grid's parameters.

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in)
            ciRi (jnp.array): values of old grid's R(x) calculated on the inputs
                shape (n_in*n_out, batch)

        Returns:
        --------
            cj (jnp.array): new coefficients corresponding to the updated grid
                shape (n_in*n_out, G'+1)
        """
        
        # Get the Rj(x) for the new grid
        R = self.basis(x) # shape (n_in*n_out, G'+1, batch)
        Rj = jnp.transpose(R, (0, 2, 1)) # shape (n_in*n_out, batch, G'+1)
        
        # Expand ciRi from (n_in*n_out, batch) to (n_in*n_out, batch, 1)
        ciRi = jnp.expand_dims(ciRi, axis=-1)

        # Solve for the new coefficients
        cj = solve_full_lstsq(Rj, ciRi)
        # Cast into shape (n_in*n_out, G'+k)
        cj = jnp.squeeze(cj, axis=-1)
        
        return cj


    def update_grid(self, x, G_new):
        """
        Performs a grid update given a new value for G, G_new, and adapts it to the given data, x.

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in)
            G_new (int): Size of the new grid (in terms of intervals)

        Returns:
        --------
            cj (jnp.array): R function coefficients
                shape (n_in*n_out, G_new+1)
        """

        # Apply the inputs to the current grid to acquire y = Sum(ciRi(x)), where ci are
        # the current coefficients and Ri(x) are the current R basis functions
        Ri = self.basis(x) # (n_in*n_out, G+1, batch)
        ci = self.c_basis # (n_in*n_out, G+1)
        ciRi = jnp.einsum('ij,ijk->ik', ci, Ri) # (n_in*n_out, batch)

        # Now start updating the grid itself, based on the inputs and the new value for G
        
        batch = x.shape[0]
        # Extend to shape (batch, n_in*n_out)
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in * self.n_out))
        # Transpose to shape (n_in*n_out, batch)
        x_ext = jnp.transpose(x_ext, (1, 0))
        # Sort inputs
        x_sorted = jnp.sort(x_ext, axis=1)

        # Get an adaptive grid of size G' + 1
        # Essentially we sample points from x, based on their density
        ids = jnp.concatenate((jnp.floor(batch / G_new * jnp.arange(G_new)).astype(int), jnp.array([-1])))
        grid_adaptive = x_sorted[:, ids]
        
        # Get a uniform grid of size G' + 1
        # Essentially we only consider the maximum and minimum values of x
        margin = 0.01
        uniform_step = (x_sorted[:, -1] - x_sorted[:, 0] + 2 * margin) / G_new
        grid_uniform = (
            jnp.arange(G_new + 1, dtype=jnp.float32)
            * uniform_step[:, None]
            + x_sorted[:, 0][:, None]
            - margin
        )

        # Perform a linear mixing of the two grid types
        grid = self.grid_e * grid_uniform + (1.0 - self.grid_e) * grid_adaptive

        # Given the new grid, calculate the new values for S, E, r
        new_S, new_E, new_r = self.get_SEr(grid)

        # Pass the new values to self (requires to be called with mutable=['state']
        self.S.value = new_S
        self.E.value = new_E
        self.r.value = new_r
        
        # Based on these new values, run the new_coeffs() function to re-initialize the coefficients' values
        cj = self.new_coeffs(x, ciRi)

        return cj


    def __call__(self, x):
        """
        Equivalent to the layer's forward pass.

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in)

        Returns:
        --------
            y (jnp.array): output of the forward pass
                shape (batch, n_out)
            R_reg (jnp.array): the array relevant to the R activation, to be used for the calculation of the loss function
                shape (n_out, n_in)
        """
        batch = x.shape[0]
        # Extend to shape (batch, n_in*n_out)
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in * self.n_out))
        # Transpose to shape (n_in*n_out, batch)
        x_ext = jnp.transpose(x_ext, (1, 0))
        
        # Calculate residual activation - shape (batch, n_in*n_out)
        res = jnp.transpose(self.residual(x_ext), (1,0))
        
        # Calculate R basis activations
        Ri = self.basis(x) # (n_in*n_out, G+1, batch)
        ci = self.c_basis # (n_in*n_out, G+1)
        # Calculate activation
        R = jnp.einsum('ij,ijk->ik', ci, Ri) # (n_in*n_out, batch)
        # Transpose to shape (batch, n_in*n_out)
        R = jnp.transpose(R, (1,0))

        # Calculate the entire activation
        cnst_R = jnp.expand_dims(self.c_R, axis=0)
        cnst_res = jnp.expand_dims(self.c_res, axis=0)
        y = (cnst_R * R) + (cnst_res * res) # (batch, n_in*n_out)
        # Reshape and sum to cast to (batch, n_out) shape
        y_reshaped = jnp.reshape(y, (batch, self.n_out, self.n_in))
        y = (1.0/self.n_in)*jnp.sum(y_reshaped, axis=2)

        # Modify R and return it along with y to be used for the regularization term of the loss function
        # Cast S's and E's shape from (n_in*n_out, G+1) to (n_out, n_in, G+1)
        S_reshaped = self.S.value.reshape(self.n_out, self.n_in, -1)
        E_reshaped = self.E.value.reshape(self.n_out, self.n_in, -1)
        # Range normalization (similar to pykan)
        input_norm = E_reshaped[:, :, -1] - S_reshaped[:, :, 0] + 1e-5
        # Reshape R from (batch, n_in*n_out) to (batch, n_out, n_in)
        R_reshaped = jnp.reshape(R, (batch, self.n_out, self.n_in))
        # Extract the batch mean of |R| and point-to-point normalize it with input norm
        R_reg = (jnp.mean(jnp.abs(R_reshaped), axis=0))/input_norm

        # Return the output of the forward pass, as well as the regularization term
        return y, R_reg
