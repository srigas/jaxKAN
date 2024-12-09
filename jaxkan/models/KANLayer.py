from jax import numpy as jnp

from flax import nnx

from ..bases.splines import get_spline_basis
from ..utils.general import solve_full_lstsq


class LayerGrid:
    """
    LayerGrid class.

    Args:
    -----------
        n_in (int): number of layer's incoming nodes.
        n_out (int): number of layer's outgoing nodes.
        k (int): the order of the spline basis functions.
        G (int): TODO
        grid_e (float): parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
        grid_range (tuple): TODO
    """
    def __init__(self, n_in: int = 2, n_out: int = 5, k: int = 3,
                 G: int = 3, grid_range: tuple = (-1,1), grid_e: float = 0.05
                ):
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.G = G
        self.grid_range = grid_range
        self.grid_e = grid_e

        self.item = self._initialize_grid()

    def _initialize_grid(self):
        """
        Create and initialize the grid.
        """
        # Calculate the step size for the knot vector based on its end values
        h = (self.grid_range[1] - self.grid_range[0]) / self.G

        # Create the initial knot vector and perform augmentation
        # Now it is expanded from G+1 points to G+1 + 2k points, because k points are appended at each of its ends
        grid = jnp.arange(-self.k, self.G + self.k + 1, dtype=jnp.float32) * h + self.grid_range[0]

        # Expand for broadcasting - the shape becomes (n_in*n_out, G + 2k + 1), so that the grid
        # can be passed in all n_in*n_out spline basis functions simultaneously
        # TODO: grid = jnp.expand_dims(grid, axis=0).repeat(self.in_features, axis=0)
        grid = jnp.expand_dims(grid, axis=0)
        grid = jnp.tile(grid, (self.n_in*self.n_out, 1))
        
        return grid

    def update(self, x, G_new):
        """
        Update the grid based on new input data and grid size.

        Args:
        -----------
            x (jnp.ndarray): Input data of shape (batch, n_in).
            G_new (int): New grid size in terms of intervals.
        """

        batch = x.shape[0]
        
        # Extend to shape (batch, n_in*n_out)
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in * self.n_out))
        # Transpose to shape (n_in*n_out, batch)
        x_ext = jnp.transpose(x_ext, (1, 0))
        # Sort inputs
        x_sorted = jnp.sort(x_ext, axis=1)

        # Get an adaptive grid of size G_new + 1
        # Essentially we sample points from x, based on their density
        ids = jnp.concatenate((jnp.floor(batch / G_new * jnp.arange(G_new)).astype(int), jnp.array([-1])))
        grid_adaptive = x_sorted[:, ids]
        
        # Get a uniform grid of size G_new + 1
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

        # Perform grid augmentation, so that the grid is extended from G_new + 1 to G_new + 2k + 1 points
        # First get a new step vector
        h = (grid[:, [-1]] - grid[:, [0]]) / G_new
        # Then calculate the left and right additions in terms of h
        left = jnp.squeeze((jnp.arange(self.k, 0, -1)*h[:,None]), axis=1) 
        right = jnp.squeeze((jnp.arange(1, self.k+1)*h[:,None]), axis=1) 
        # Finally, concatenate left and right
        grid = jnp.concatenate(
            [
                grid[:, [0]] - left,
                grid,
                grid[:, [-1]] + right
            ],
            axis=1,
        )
        
        # Update the grid value and size
        self.item = grid
        self.G = G_new
        
        
        
class KANLayer(nnx.Module):
    """
    KANLayer class.

    Args:
    -----------
        n_in (int): number of layer's incoming nodes.
        n_out (int): number of layer's outgoing nodes.
        k (int): the order of the spline basis functions.
        G (int): TODO
        residual (nn.Module): function that is applied on samples to calculate residual activation.
        noise_std (float): noise for the initialization of spline coefficients.
        grid_e (float): parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
        grid_range (tuple): TODO
    """
    def __init__(self,
                 n_in: int = 2, n_out: int = 5, k: int = 3,
                 G: int = 3, grid_range: tuple = (-1,1), grid_e: float = 0.05,
                 residual: nnx.Module = nnx.silu,
                 noise_std: float = 0.15,
                 rngs: nnx.Rngs = nnx.Rngs(42)
                ):

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.residual = residual

        # Now registers and initialize all parameters of the KANLayer. Specifically:
        # grid: non-trainable variable
        # c_basis: trainable parameter
        # c_spl, c_res: trainable or non-trainable depending on value of const_spl, const_res

        # Initialize the grid
        self.grid = LayerGrid(n_in=n_in, n_out=n_out, k=k, G=G, grid_range=grid_range, grid_e=grid_e)

        # Register & initialize the spline basis functions' coefficients as trainable parameters
        # They are drawn from a normal distribution with zero mean and an std of noise_std
        self.c_basis = nnx.Param(
            nnx.initializers.normal(stddev=noise_std)(
                rngs.params(), (n_in * n_out, G+k), jnp.float32)
        )

        # Register the factors of spline and residual activations as parameters
        self.c_spl = nnx.Param(jnp.ones(self.n_in * self.n_out))
        self.c_res = nnx.Param(jnp.ones(self.n_in * self.n_out))
        
        


    def basis(self, x):
        """
        Uses k and the current grid to calculate the values of spline basis functions on the input.

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in)

        Returns:
        --------
            bases (jnp.array): spline basis functions applied on inputs
                shape (n_in*n_out, G+k, batch)
        """
        batch = x.shape[0]
        # Extend to shape (batch, n_in*n_out)
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in * self.n_out))
        # Transpose to shape (n_in*n_out, batch)
        x_ext = jnp.transpose(x_ext, (1, 0))
        
        bases = get_spline_basis(x_ext, self.grid.item, self.k)
        
        return bases


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
            cj (jnp.array): spline function coefficients
                shape (n_in*n_out, G_new+k)
        """

        # Apply the inputs to the current grid to acquire y = Sum(ciBi(x)), where ci are
        # the current coefficients and Bi(x) are the current spline basis functions
        Bi = self.basis(x) # (n_in*n_out, G+k, batch)
        ci = self.c_basis.value # (n_in*n_out, G+k)
        ciBi = jnp.einsum('ij,ijk->ik', ci, Bi) # (n_in*n_out, batch)

        # Update the grid
        self.grid.update(x, G_new)

        # Get the Bj(x) for the new grid
        A = self.basis(x) # shape (n_in*n_out, G'+k, batch)
        Bj = jnp.transpose(A, (0, 2, 1)) # shape (n_in*n_out, batch, G'+k)
        
        # Expand ciBi from (n_in*n_out, batch) to (n_in*n_out, batch, 1)
        ciBi = jnp.expand_dims(ciBi, axis=-1)

        # Solve for the new coefficients
        cj = solve_full_lstsq(Bj, ciBi)
        # Cast into shape (n_in*n_out, G'+k)
        cj = jnp.squeeze(cj, axis=-1)

        self.c_basis.value = cj


    def __call__(self, x):
        """
        Equivalent to the layer's forward pass.

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in)

        Returns:
        --------
            y (jnp.array): output of the forward pass, corresponding to the weighted sum of the B-spline activation and the residual activation
                shape (batch, n_out)
            spl_reg (jnp.array): the array relevant to the B-spline activation, to be used for the calculation of the loss function
                shape (n_out, n_in)
        """
        batch = x.shape[0]
        # Extend to shape (batch, n_in*n_out)
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in * self.n_out))
        # Transpose to shape (n_in*n_out, batch)
        x_ext = jnp.transpose(x_ext, (1, 0))
        
        # Calculate residual activation - shape (batch, n_in*n_out)
        res = jnp.transpose(self.residual(x_ext), (1,0))

        # Calculate spline basis activations
        Bi = self.basis(x) # (n_in*n_out, G+k, batch)
        ci = self.c_basis.value # (n_in*n_out, G+k)
        # Calculate spline activation
        spl = jnp.einsum('ij,ijk->ik', ci, Bi) # (n_in*n_out, batch)
        # Transpose to shape (batch, n_in*n_out)
        spl = jnp.transpose(spl, (1,0))

        # Calculate the entire activation
        cnst_spl = jnp.expand_dims(self.c_spl.value, axis=0)
        cnst_res = jnp.expand_dims(self.c_res.value, axis=0)
        y = (cnst_spl * spl) + (cnst_res * res) # (batch, n_in*n_out)
        # Reshape and sum to cast to (batch, n_out) shape
        y_reshaped = jnp.reshape(y, (batch, self.n_out, self.n_in))
        y = (1.0/self.n_in)*jnp.sum(y_reshaped, axis=2)

        # Modify the spline activation array and return it along with y to be used for the regularization term of the loss function
        # Cast grid shape from (n_in*n_out, G+2k+1) to (n_out, n_in, G+2k+1)
        grid_reshaped = self.grid.item.reshape(self.n_out, self.n_in, -1)
        # Extra normalization inspired by pykan - get the grid range per spline function
        input_norm = grid_reshaped[:, :, -1] - grid_reshaped[:, :, 0] + 1e-5
        # Reshape spl from (batch, n_in*n_out) to (batch, n_out, n_in)
        spl_reshaped = jnp.reshape(spl, (batch, self.n_out, self.n_in))
        # Extract the batch mean of |spl| (Eq. 2.17 in arXiv pre-print) and point-to-point normalize it with input norm
        spl_reg = (jnp.mean(jnp.abs(spl_reshaped), axis=0))/input_norm

        # Return the output of the forward pass, as well as the regularization term
        return y, spl_reg
