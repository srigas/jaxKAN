import jax.numpy as jnp


class RBFGrid:
    """
    RBFGrid class, corresponding to the grid of the RBFLayer class. It comprises an initialization as well as an update procedure.

    Attributes:
        n_nodes (int):
            Number of layer nodes.
        D (int):
            Number of radial basis functions.
        grid_range (tuple):
            The range of the grid's ends, on which the basis functions are defined.
        grid_e (float):
            Parameter that defines if the grid is uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
        item (jnp.array):
            The actual grid array, shape (n_nodes, D).
    """
    
    def __init__(self, n_nodes: int = 2, D: int = 3, grid_range: tuple = (-2.0, 2.0), grid_e: float = 1.0):
        """
        Initializes a RBFGrid instance.
        
        Args:
            n_nodes (int):
                Number of layer nodes.
            D (int):
                Number of radial basis functions.
            grid_range (tuple):
                The range of the grid's ends, on which the basis functions are defined.
            grid_e (float):
                Parameter that defines if the grid is uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
            
        Example:
            >>> grid_type = RBFGrid(n_nodes = 2, D = 4, grid_range = (-2.0, 2.0), grid_e = 1.0)
            >>> grid = grid_type.item
        """
                
        self.n_nodes = n_nodes
        self.D = D
        self.grid_range = grid_range
        self.grid_e = grid_e

        # Initialize the grid, which is henceforth callable as .item
        self.item = self._initialize()

    
    def _initialize(self):
        """
        Create and initialize the grid. Can also be used to reset a grid to the default value.

        Returns:
            grid (jnp.array):
                Grid for the RBFLayer, shape (n_nodes, D).
        
        Example:
            >>> grid = RBFGrid(n_nodes = 2, D = 4, grid_range = (-2.0, 2.0), grid_e = 1.0)
            >>> grid.item = grid._initialize()
        """
        
        grid = jnp.linspace(self.grid_range[0], self.grid_range[-1], self.D)
        grid = jnp.tile(grid, (self.n_nodes, 1))
        
        return grid

    
    def update(self, x, D_new):
        """
        Update the grid based on input data and new grid size.

        Args:
            x (jnp.ndarray):
                Input data, shape (batch, n_nodes).
            D_new (int):
                New number of basis functions.
            
        Example:
            >>> key = jax.random.key(42)
            >>> grid_range = (-2.0, 2.0)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=grid_range[0], maxval=grid_range[-1])
            >>>
            >>> grid = RBFGrid(n_nodes = 2, D = 4, grid_range = grid_range, grid_e = 1.0)
            >>> grid.update(x=x_batch, D_new=8)
        """

        batch = x.shape[0]

        # Sort inputs
        x_sorted = jnp.sort(x, axis=0)
        
        # Get an adaptive grid of size D_new by sampling points from x, based on their density
        ids = jnp.concatenate((jnp.floor(batch / (D_new-1) * jnp.arange(D_new-1)).astype(int), jnp.array([-1])))
        grid_adaptive = x_sorted[ids]
        
        # Get a uniform grid of size D_new by considering only the maximum and minimum values of x
        uniform_step = (x_sorted[-1] - x_sorted[0]) / (D_new-1)
        
        grid_uniform = (
            jnp.arange(D_new, dtype=jnp.float32)[:, None]
            * uniform_step
            + x_sorted[0]
        )

        # Perform a linear mixing of the two grid types
        grid = self.grid_e * grid_uniform + (1.0 - self.grid_e) * grid_adaptive
        
        # Update the grid value and size
        self.item = grid.T
        self.D = D_new
        