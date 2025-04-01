from jax import numpy as jnp


class BaseGrid:
    """
    BaseGrid class, corresponding to the grid of the BaseLayer class. It comprises an initialization as well as an update procedure.

    Attributes:
        n_in (int):
            Number of layer's incoming nodes.
        n_out (int):
            Number of layer's outgoing nodes.
        k (int):
            Order of the spline basis functions.
        G (int):
            Number of grid intervals.
        grid_range (tuple):
            An initial range for the grid's ends, although adaptivity can completely change it.
        grid_e (float):
            Parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
    """
    
    def __init__(self, n_in: int = 2, n_out: int = 5, k: int = 3,
                 G: int = 3, grid_range: tuple = (-1,1), grid_e: float = 0.05
                ):
        """
        Initializes a BaseGrid instance.
        
        Args:
            n_in (int):
                Number of layer's incoming nodes.
            n_out (int):
                Number of layer's outgoing nodes.
            k (int):
                Order of the spline basis functions.
            G (int):
                Number of grid intervals.
            grid_range (tuple):
                An initial range for the grid's ends, although adaptivity can completely change it.
            grid_e (float):
                Parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
            
        Example:
            >>> grid_type = BaseGrid(n_in = 2, n_out = 5, k = 3, G = 3, grid_range = (-1,1), grid_e = 0.05)
            >>> grid = grid_type.item
        """
                
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

        Returns:
            grid (jnp.array):
                Grid for the BaseLayer, shape (n_in*n_out, G + 2k + 1).
        
        Example:
            >>> grid = BaseGrid(n_in = 2, n_out = 5, k = 3, G = 3, grid_range = (-1,1), grid_e = 0.05)
            >>> grid.item = grid._initialize()
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
            x (jnp.ndarray):
                Input data, shape (batch, n_in).
            G_new (int):
                New grid size in terms of intervals.
            
        Example:
            >>> key = jax.random.PRNGKey(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
            >>>
            >>> grid = BaseGrid(n_in = 2, n_out = 5, k = 3, G = 3, grid_range = (-1,1), grid_e = 0.05)
            >>> grid.update(x=x_batch, G_new=5)
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