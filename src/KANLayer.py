import jax
import jax.numpy as jnp
import math
from flax import linen as nn
from flax.training import train_state
from flax.linen import initializers

from bases.splines import get_spline_basis
from bases.splines import get_coeffs

class KANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        n_in (int): number of layer's incoming nodes. Default: 2
        n_out (int): number of layer's outgoing nodes. Default: 5
        G (int): the size of the initial knot vector's intervals. Default: 5
        grid_range (tuple): the ends of the initial knot vector. Default (-1,1)
        k (int): the order of the spline basis functions. Default: 3
        const_spl (float/bool): coefficient of the spline function in the overall activation. If set to False, then it is trainable per activation. Default: False
        const_res (float/bool): coefficient of the residual function in the overall activation. If set to False, then it is trainable per activation. Default: False
        residual (nn.Module): function that is applied on samples to calculate residual activation. Default: nn.swish
        noise_std (float): noise for the initialization of spline coefficients. Default: 0.1
        grid_e (float): parameter that defines if the grids are uniform (grid_e = 0.0) or sample-dependent (grid_e = 1.0). Intermediate values correspond to a linear mixing of the two cases. Default: 0.05
    
    Methods:
    --------
        setup(): Initializes the grid and the trainable parameters of the KAN Layer
        basis(x): Passes the input x, the grid and k into the get_spline_basis() function of the bases module
        call(): Corresponds to the layer's forward pass
    """
    
    n_in: int = 2
    n_out: int = 5
    G: int = 5
    grid_range: tuple = (-1, 1)
    k: int = 3

    const_spl: float or bool = False
    const_res: float or bool = False
    residual: nn.Module = nn.swish
    
    noise_std: float = 0.1
    grid_e: float = 0.02

    
    def setup(self):
        # Calculate the step size for the knot vector based on its end values
        h = (self.grid_range[1] - self.grid_range[0]) / (self.G - 1)

        # Create the initial knot vector and perform augmentation
        # Now it is expanded from G+1 points to G+1 + 2k points, because k points are appended at each of its ends
        grid = jnp.arange(-self.k, self.G + self.k + 1, dtype=jnp.float32) * h + self.grid_range[0]
        
        # Expand for broadcasting - the shape becomes (n_in*n_out, G + 2k + 1), so that the grid
        # can be passed in all n_in*n_out spline basis functions simultaneously
        grid = jnp.expand_dims(grid, axis=0)
        grid = jnp.tile(grid, (self.n_in*self.n_out, 1))

        # Store the grid as a non trainable variable
        self.grid = self.variable('state', 'grid', lambda: grid)
        
        # Register & initialize the spline basis functions' coefficients as trainable parameters
        # They are drawn from a normal distribution with zero mean and an std of noise_std
        self.c_basis = self.param('c_basis', initializers.normal(stddev=self.noise_std), (self.n_in * self.n_out, self.G + self.k))
        
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
        grid = self.grid.value
        return get_spline_basis(x, grid, self.k)


    def __call__(self, x):
        # Dummy
        return x
