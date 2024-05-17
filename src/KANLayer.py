import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.linen import initializers

from bases.splines import get_spline_basis
from utils import solve_full_lstsq

class KANLayer(nn.Module):
    """
    KANLayer class

    Args:
    -----------
        n_in (int): number of layer's incoming nodes. Default: 2
        n_out (int): number of layer's outgoing nodes. Default: 5
        k (int): the order of the spline basis functions. Default: 3
        const_spl (float/bool): coefficient of the spline function in the overall activation. If set to False, then it is trainable per activation. Default: False
        const_res (float/bool): coefficient of the residual function in the overall activation. If set to False, then it is trainable per activation. Default: False
        residual (nn.Module): function that is applied on samples to calculate residual activation. Default: nn.swish
        noise_std (float): noise for the initialization of spline coefficients. Default: 0.1
        grid_e (float): parameter that defines if the grids are uniform (grid_e = 0.0) or sample-dependent (grid_e = 1.0). Intermediate values correspond to a linear mixing of the two cases. Default: 0.05
    """
    
    n_in: int = 2
    n_out: int = 5
    k: int = 3

    """
    init_G: int = 5
    init_knot: tuple = (-1, 1)
    """

    const_spl: float or bool = False
    const_res: float or bool = False
    residual: nn.Module = nn.swish
    
    noise_std: float = 0.1
    grid_e: float = 0.02

    
    def setup(self):
        """
        Registers and initializes all parameters of the KANLayer. Specifically:
            * grid: non-trainable variable
            * c_basis, c_spl, c_res: trainable parameters
        """

        # What follows resembles the efficientkan implementation
        # It is kind of pointless to do all of these initializations, so instead
        # we will initialize dummies and create an update_grid function that
        # will have to be applied for all layers during the first forward pass, as well as
        # every X epochs - or whenever some other criterion is met
        # These will be kept as a reference, until we decide that we're sticking to our method
        """
        # Initialize a value of G, this will be updated during training
        self.G = self.variable('state', 'G', lambda: jnp.array(self.init_G)
        
        # Calculate the step size for the knot vector based on its end values
        h = (self.init_knot[1] - self.init_knot[0]) / (self.G - 1)

        # Create the initial knot vector and perform augmentation
        # Now it is expanded from G+1 points to G+1 + 2k points, because k points are appended at each of its ends
        grid = jnp.arange(-self.k, self.G + self.k + 1, dtype=jnp.float32) * h + self.init_knot[0]
        
        # Expand for broadcasting - the shape becomes (n_in*n_out, G + 2k + 1), so that the grid
        # can be passed in all n_in*n_out spline basis functions simultaneously
        grid = jnp.expand_dims(grid, axis=0)
        grid = jnp.tile(grid, (self.n_in*self.n_out, 1))
        """

        # Random initialization of grid corresponding to a random value of G, here 3
        grid = jnp.ones((self.n_in*self.n_out, 3+2*self.k+1))

        # Store the grid as a non trainable variable
        self.grid = self.variable('state', 'grid', lambda: grid)
        
        # Register & initialize the spline basis functions' coefficients as trainable parameters
        # They are drawn from a normal distribution with zero mean and an std of noise_std
        self.c_basis = self.param('c_basis', initializers.normal(stddev=self.noise_std), (self.n_in * self.n_out, 3 + self.k))
        
        # If const_spl is set as a float value, treat it as non trainable and pass it to the c_spl array with shape (n_in*n_out)
        # Otherwise register it as a trainable parameter of the same size and initialize it
        if isinstance(self.const_spl, float):
            self.c_spl = jnp.ones(self.n_in*self.n_out) * self.const_spl
        elif self.const_spl is False:
            self.c_spl = self.param('c_spl', initializers.constant(1.0), (self.n_in * self.n_out,))

        # If const_res is set as a float value, treat it as non trainable and pass it to the c_res array with shape (n_in*n_out)
        # Otherwise register it as a trainable parameter of the same size and initialize it
        if isinstance(self.const_res, float):
            self.c_res = jnp.ones(self.n_in * self.n_out) * self.const_res
        elif self.const_res is False:
            self.c_res = self.param('c_res', initializers.constant(1.0), (self.n_in * self.n_out,))


    def basis(self, x):
        """
        Uses k and the current grid to calculate the values of spline basis functions on the input

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in*n_out)

        Returns:
        --------
            bases (jnp.array): spline basis functions applied on inputs
                shape (batch, n_in*n_out, G + k)
        """
        grid = self.grid.value
        k = self.k
        bases = get_spline_basis(x, grid, k)
        
        return bases

    
    def new_coeffs(self, x, ciBi):
        """
        Utilizes the new grid's basis functions, Bj(x), and the old grid's splines, Sum(ciBix(x)), to approximate a good initialization for the updated grid's parameters

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in*n_out)
            ciBi (jnp.array): values of old grid's splines calculated on the inputs
                shape (n_in*n_out, batch)

        Returns:
        --------
            cj (jnp.array): new coefficients corresponding to the updated grid
                shape (n_in*n_out, G' + k)
        """
        # Get the Bj(x) for the new grid
        A = self.basis(x)
        # Cast into shape (n_in*n_out, batch, G' + k)
        Bj = jnp.transpose(A, (1, 0, 2))
        
        # Expand ciBi from (n_in*n_out, batch) to (n_in*n_out, batch, 1)
        ciBi = jnp.expand_dims(ciBi, axis=-1)

        # Solve for the new coefficients
        cj = solve_full_lstsq(Bj, ciBi)
        # Cast into shape (n_in*n_out, G' + k)
        cj = jnp.squeeze(ci, axis=-1)
        
        return cj


    def update_grid(self, x, G_new):
        """
        Performs a grid update given a new value for G, G_new, and adapts it to the given data, x.

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in*n_out)
            G_new (int): Size of the new grid (in terms of intervals)

        """

        # Apply the inputs to the current grid to acquire y = Sum(ciBi(x)), where ci are
        # the current coefficients and Bi(x) are the current spline basis functions
        batch = x.size(0)
        margin = 0.01

        Bi = self.basis(x) # (batch, n_in*n_out, G+k)
        Bi = jnp.transpose(Bi, (1, 2, 0)) # (n_in*n_out, G+k, batch)
        ci = self.c_basis # (n_in*n_out, G+k)
        ciBi = jnp.einsum('ij,ijk->ik', ci, Bi) # (n_in*n_out, batch)

        # Update the grid itself, based on the inputs and the new value for G
        # Sort inputs
        x_sorted = jnp.sort(x, axis=0)

        # Get an adaptive grid of size G' + 1
        # Essentially we sample points from x, based on their density
        grid_adaptive = x_sorted[jnp.linspace(0, batch - 1, G_new + 1, dtype=jnp.int32)]

        # Get a uniform grid of size G' + 1
        # Essentially we only consider the maximum and minimum values of x
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / G_new
        grid_uniform = (
            jnp.arange(G_new + 1).reshape(-1, 1) * uniform_step
            + x_sorted[0]
            - margin
        )

        # Perform a linear mixing of the two grid types
        grid = self.grid_e * grid_uniform + (1 - self.grid_e) * grid_adaptive

        # Perform grid augmentation, so that the grid is extended from G' + 1 to G' + 2k + 1 points
        grid = jnp.concatenate(
            [
                grid[:1] - uniform_step * jnp.arange(spline_order, 0, -1).reshape(-1, 1),
                grid,
                grid[-1:] + uniform_step * jnp.arange(1, spline_order + 1).reshape(-1, 1),
            ],
            axis=0,
        )

        # Based on the new grid, run the new_coeffs() function to re-initialize the coefficients' values
        cj = self.new_coeffs(x, ciBi)

        # Update the grid and the coefficients

        # TORCH VERSION
        #self.grid.copy_(grid.T)
        #self.c_basis.data.copy_(self.curve2coeff(x, unreduced_spline_output))


    def __call__(self, x):
        """
        Equivalent to the layer's forward pass.

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in*n_out)

        Returns:
        --------
            TODO : TODO
                shape (TODO)
        """
        # Dummy
        return x
