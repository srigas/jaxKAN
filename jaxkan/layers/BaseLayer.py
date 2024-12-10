from jax import numpy as jnp

from flax import nnx

from ..utils.general import solve_full_lstsq


class BaseGrid:
    """
        BaseGrid class, corresponding to the grid of the BaseLayer class. It comprises an initialization as well as an update procedure.

        Args:
        -----
            n_in (int): number of layer's incoming nodes.
            n_out (int): number of layer's outgoing nodes.
            k (int): order of the spline basis functions.
            G (int): number of grid intervals.
            grid_range (tuple): an initial range for the grid's ends, although adaptivity can completely change it.
            grid_e (float): parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
            
        Example Usage:
        --------------
            grid_type = BaseGrid(n_in = 2, n_out = 5, k = 3, G = 3, grid_range = (-1,1), grid_e = 0.05)
            grid = grid_type.item
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

        # Initialize the grid, which is henceforth callable as .item
        self.item = self._initialize()

    def _initialize(self):
        """
            Create and initialize the grid. Can also be used to reset a grid to the default value.
            
            Example Usage:
            --------------
                grid.item = grid._initialize()
        """
        
        # Calculate the step size for the knot vector based on its end values
        h = (self.grid_range[1] - self.grid_range[0]) / self.G

        # Create the initial knot vector and perform augmentation
        # Now it is expanded from G+1 points to G+1 + 2k points, because k points are appended at each of its ends
        grid = jnp.arange(-self.k, self.G + self.k + 1, dtype=jnp.float32) * h + self.grid_range[0]

        # Expand for broadcasting - the shape becomes (n_in*n_out, G + 2k + 1), so that the grid
        # can be passed in all n_in*n_out spline basis functions simultaneously
        grid = jnp.expand_dims(grid, axis=0)
        grid = jnp.tile(grid, (self.n_in*self.n_out, 1))
        
        return grid

    def update(self, x, G_new):
        """
            Update the grid based on input data and new grid size.

            Args:
            -----
                x (jnp.ndarray): Input data of shape (batch, n_in).
                G_new (int): New grid size in terms of intervals.
                
            Example Usage:
            --------------
                key = jax.random.PRNGKey(42)
                x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
                
                grid = BaseGrid(n_in = 2, n_out = 5, k = 3, G = 3, grid_range = (-1,1), grid_e = 0.05)
                grid.update(x=x_batch, G_new=5)
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
        
        
class BaseLayer(nnx.Module):
    """
        BaseLayer class. Corresponds to the original spline-based KAN Layer introduced in the original version of KAN.
        Ref: https://arxiv.org/abs/2404.19756

        Args:
        -----
            n_in (int): number of layer's incoming nodes.
            n_out (int): number of layer's outgoing nodes.
            k (int): order of the spline basis functions.
            G (int): number of grid intervals.
            grid_range (tuple): an initial range for the grid's ends, although adaptivity can completely change it.
            grid_e (float): parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
            residual (nn.Module): function that is applied on samples to calculate residual activation.
            noise_std (float): noise for the initialization of spline coefficients.
            rngs (nnx.Rngs): random key selection for initializations wherever necessary.
            
        Example Usage:
        --------------
            layer = BaseLayer(n_in = 2, n_out = 5, k = 3, G = 3, grid_range = (-1,1),
                              grid_e = 0.05, residual = nnx.silu, noise_std = 0.1, rngs = nnx.Rngs(42))
    """
    
    def __init__(self,
                 n_in: int = 2, n_out: int = 5, k: int = 3,
                 G: int = 3, grid_range: tuple = (-1,1), grid_e: float = 0.05,
                 residual: nnx.Module = nnx.silu,
                 noise_std: float = 0.1,
                 rngs: nnx.Rngs = nnx.Rngs(42)
                ):

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.residual = residual

        # Now register and initialize all parameters of the layer. Specifically:
        # grid: non-trainable variable
        # c_basis, c_spl, c_res: trainable parameters

        # Initialize the grid
        self.grid = BaseGrid(n_in=n_in, n_out=n_out, k=k, G=G, grid_range=grid_range, grid_e=grid_e)

        # Register & initialize the spline basis functions' coefficients as trainable parameters
        # They are drawn from a normal distribution with zero mean and an std of noise_std
        self.c_basis = nnx.Param(
            nnx.initializers.normal(stddev=noise_std)(
                rngs.params(), (self.n_in * self.n_out, self.grid.G+self.k), jnp.float32)
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
                basis_splines (jnp.array): spline basis functions applied on inputs
                    shape (n_in*n_out, G+k, batch)
                
            Example Usage:
            --------------
                layer = BaseLayer(n_in = 2, n_out = 5, k = 3, G = 3, grid_range = (-1,1),
                                  grid_e = 0.05, residual = nnx.silu, noise_std = 0.1, rngs = nnx.Rngs(42))
                              
                key = jax.random.PRNGKey(42)
                x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
                
                output = layer.basis(x_batch)
        """
        
        batch = x.shape[0]
        # Extend to shape (batch, n_in*n_out)
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in * self.n_out))
        # Transpose to shape (n_in*n_out, batch)
        x_ext = jnp.transpose(x_ext, (1, 0))
        
        # Broadcasting for vectorized operations
        grid = jnp.expand_dims(self.grid.item, axis=2)
        x = jnp.expand_dims(x_ext, axis=1)

        # k = 0 case
        basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
        
        # Recursion done through iteration
        for K in range(1, self.k+1):
            left_term = (x - grid[:, :-(K + 1)]) / (grid[:, K:-1] - grid[:, :-(K + 1)])
            right_term = (grid[:, K + 1:] - x) / (grid[:, K + 1:] - grid[:, 1:(-K)])
            
            basis_splines = left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]

        return basis_splines


    def update_grid(self, x, G_new):
        """
            Performs a grid update given a new value for G (i.e., G_new) and adapts it to the given data, x.
            Additionally, re-initializes the c_basis parameters to a better estimate, based on the new grid.

            Args:
            -----
                x (jnp.array): inputs
                    shape (batch, n_in)
                G_new (int): Size of the new grid (in terms of intervals)
                
            Example Usage:
            --------------
                layer = BaseLayer(n_in = 2, n_out = 5, k = 3, G = 3, grid_range = (-1,1),
                                  grid_e = 0.05, residual = nnx.silu, noise_std = 0.1, rngs = nnx.Rngs(42))
                              
                key = jax.random.PRNGKey(42)
                x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
                
                layer.update_grid(x=x_batch, G_new=5)
        """

        # Apply the inputs to the current grid to acquire y = Sum(ciBi(x)), where ci are
        # the current coefficients and Bi(x) are the current spline basis functions
        Bi = self.basis(x) # (n_in*n_out, G+k, batch)
        ci = self.c_basis.value # (n_in*n_out, G+k)
        ciBi = jnp.einsum('ij,ijk->ik', ci, Bi) # (n_in*n_out, batch)

        # Update the grid
        self.grid.update(x, G_new)

        # Get the Bj(x) for the new grid
        A = self.basis(x) # shape (n_in*n_out, G_new+k, batch)
        Bj = jnp.transpose(A, (0, 2, 1)) # shape (n_in*n_out, batch, G_new+k)
        
        # Expand ciBi from (n_in*n_out, batch) to (n_in*n_out, batch, 1)
        ciBi = jnp.expand_dims(ciBi, axis=-1)

        # Solve for the new coefficients
        cj = solve_full_lstsq(Bj, ciBi)
        # Cast into shape (n_in*n_out, G_new+k)
        cj = jnp.squeeze(cj, axis=-1)

        self.c_basis = nnx.Param(cj)


    def __call__(self, x):
        """
            The layer's forward pass.

            Args:
            -----
                x (jnp.array): inputs
                    shape (batch, n_in)

            Returns:
            --------
                y (jnp.array): output of the forward pass, corresponding to the weighted sum of the B-spline activation and the residual activation
                    shape (batch, n_out)
                
            Example Usage:
            --------------
                layer = BaseLayer(n_in = 2, n_out = 5, k = 3, G = 3, grid_range = (-1,1),
                                  grid_e = 0.05, residual = nnx.silu, noise_std = 0.1, rngs = nnx.Rngs(42))
                              
                key = jax.random.PRNGKey(42)
                x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
                
                output = layer(x_batch)
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
        
        return y