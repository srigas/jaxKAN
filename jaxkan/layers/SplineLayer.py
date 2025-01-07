from jax import numpy as jnp

from flax import nnx

from ..utils.general import solve_full_lstsq


class SplineGrid:
    """
    SplineGrid class, corresponding to the grid of the SplineLayer class. It comprises an initialization as well as an update procedure.

    Attributes:
        n_nodes (int):
                Number of layer nodes.
        k (int):
            Order of the spline basis functions.
        G (int):
            Number of grid intervals.
        grid_range (tuple):
            An initial range for the grid's ends, although adaptivity can completely change it.
        grid_e (float):
            Parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
    """
    
    def __init__(self, n_nodes: int = 2, k: int = 3, G: int = 3, grid_range: tuple = (-1,1), grid_e: float = 0.05):
        """
        Initializes a SplineGrid instance.
        
        Args:
            n_nodes (int):
                Number of layer nodes.
            k (int):
                Order of the spline basis functions.
            G (int):
                Number of grid intervals.
            grid_range (tuple):
                An initial range for the grid's ends, although adaptivity can completely change it.
            grid_e (float):
                Parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
            
        Example:
            >>> grid_type = SplineGrid(n_nodes = 2, k = 3, G = 3, grid_range = (-1,1), grid_e = 0.05)
            >>> grid = grid_type.item
        """
                
        self.n_nodes = n_nodes
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
                Grid for the SplineLayer, shape (n_nodes, G + 2k + 1).
        
        Example:
            >>> grid = BaseGrid(n_nodes = 2, k = 3, G = 3, grid_range = (-1,1), grid_e = 0.05)
            >>> grid.item = grid._initialize()
        """
        
        # Calculate the step size for the knot vector based on its end values
        h = (self.grid_range[1] - self.grid_range[0]) / self.G

        # Create the initial knot vector and perform augmentation
        # Now it is expanded from G+1 points to G+1 + 2k points, because k points are appended at each of its ends
        grid = jnp.arange(-self.k, self.G + self.k + 1, dtype=jnp.float32) * h + self.grid_range[0]

        # Expand across nodes - the shape becomes (n_nodes, G + 2k + 1), with a grid corresponding to each node
        grid = jnp.expand_dims(grid, axis=0).repeat(self.n_nodes, axis=0)
        
        return grid

    def update(self, x, G_new):
        """
        Update the grid based on input data and new grid size.

        Args:
            x (jnp.ndarray):
                Input data, shape (batch, n_nodes).
            G_new (int):
                New grid size in terms of intervals.
            
        Example:
            >>> key = jax.random.PRNGKey(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
            >>>
            >>> grid = SplineGrid(n_nodes = 2, k = 3, G = 3, grid_range = (-1,1), grid_e = 0.05)
            >>> grid.update(x=x_batch, G_new=5)
        """

        batch = x.shape[0]
        
        # Sort inputs
        x_sorted = jnp.sort(x, axis=0)

        # Get an adaptive grid of size G_new + 1
        # Essentially we sample points from x, based on their density
        ids = jnp.concatenate((jnp.floor(batch / G_new * jnp.arange(G_new)).astype(int), jnp.array([-1])))
        grid_adaptive = x_sorted[ids]
        
        # Get a uniform grid of size G_new + 1
        # Essentially we only consider the maximum and minimum values of x
        margin = 0.01
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / G_new
        
        grid_uniform = (
            jnp.arange(G_new + 1, dtype=jnp.float32)[:, None]
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # Perform a linear mixing of the two grid types
        grid = self.grid_e * grid_uniform + (1.0 - self.grid_e) * grid_adaptive
        
        # Perform grid augmentation, so that the grid is extended from G_new + 1 to G_new + 2k + 1 points
        # First get a new step vector
        h = (grid[-1] - grid[0]) / G_new
        # Then calculate the left and right additions in terms of h
        left = h * jnp.arange(self.k, 0, -1)[:, None]
        right = h * jnp.arange(1, self.k + 1)[:, None]
        # Finally, concatenate left and right
        grid = jnp.concatenate(
            [
                grid[:1] - left,
                grid,
                grid[-1:] + right
            ],
            axis = 0,
        )
        
        # Update the grid value and size
        self.item = grid.T
        self.G = G_new
        
        
class SplineLayer(nnx.Module):
    """
    SplineLayer class. Corresponds to the "efficient" version of the spline-based KAN Layer. Ref: https://github.com/Blealtan/efficient-kan

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
        residual (nn.Module):
            Function that is applied on samples to calculate residual activation.
        base_basis (float):
            std coefficient for initialization of basis weights.
        base_spline (float):
            std coefficient for initialization of spline weights.
        base_res (float):
            std coefficient for initialization of residual weights.
        pow_basis (float):
            Power to raise the 1.0/n_in term for basis weights initialization. 
        pow_spline (float):
            Power to raise the 1.0/n_in term for spline weights initialization.
        pow_res (float):
            Power to raise the 1.0/n_in term for residual weights initialization.
        rngs (nnx.Rngs):
            Random key selection for initializations wherever necessary.
    """
    
    def __init__(self,
                 n_in: int = 2, n_out: int = 5, k: int = 3,
                 G: int = 3, grid_range: tuple = (-1,1), grid_e: float = 0.05,
                 residual: nnx.Module = nnx.silu,
                 base_basis: float = 1.0,
                 base_spline: float = 1.0,
                 base_res: float = 1.0,
                 pow_basis: float = 0.5,
                 pow_spline: float = 0.5,
                 pow_res: float = 0.5,
                 rngs: nnx.Rngs = nnx.Rngs(42)
                ):
        """
        Initializes a BaseLayer instance.

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
            residual (nn.Module):
                Function that is applied on samples to calculate residual activation.
            base_basis (float):
                std coefficient for initialization of basis weights.
            base_spline (float):
                std coefficient for initialization of spline weights.
            base_res (float):
                std coefficient for initialization of residual weights.
            pow_basis (float):
                Power to raise the 1.0/n_in term for basis weights initialization. 
            pow_spline (float):
                Power to raise the 1.0/n_in term for spline weights initialization.
            pow_res (float):
                Power to raise the 1.0/n_in term for residual weights initialization.
            rngs (nnx.Rngs):
                Random key selection for initializations wherever necessary.
            
        Example:
            >>> layer = SplineLayer(n_in = 2, n_out = 5, k = 3,
            >>>                     G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                     base_basis = 1.0, base_spline = 1.0, base_res = 1.0,
            >>>                     pow_basis = 0.5, pow_spline = 0.5, pow_res = 0.5, 
            >>>                     rngs = nnx.Rngs(42))
        """

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.residual = residual

        # Now register and initialize all parameters of the layer. Specifically:
        # grid: non-trainable variable
        # c_basis, c_spl, c_res: trainable parameters

        # Initialize the grid - shape (n_in, G+2k+1)
        self.grid = SplineGrid(n_nodes=n_in, k=k, G=G, grid_range=grid_range, grid_e=grid_e)

        # Register & initialize the spline basis functions' coefficients as trainable parameters
        # shape (n_out, n_in, G+k)
        basis_std = base_basis/(self.n_in**pow_basis)
        self.c_basis = nnx.Param(
            nnx.initializers.normal(stddev=basis_std)(
                rngs.params(), (self.n_out, self.n_in, self.grid.G+self.k), jnp.float32)
        )

        # Register the factors of spline and residual activations as parameters
        # shape (n_out, n_in)
        spline_std = base_spline/(self.n_in**pow_spline)
        self.c_spl = nnx.Param(
        nnx.initializers.normal(stddev=spline_std)(
                rngs.params(), (self.n_out, self.n_in), jnp.float32)
        )
        
        res_std = base_res/(self.n_in**pow_res)
        self.c_res = nnx.Param(
        nnx.initializers.normal(stddev=res_std)(
                rngs.params(), (self.n_out, self.n_in), jnp.float32)
        )

    def basis(self, x):
        """
        Uses k and the current grid to calculate the values of spline basis functions on the input.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).

        Returns:
            basis_splines (jnp.array):
                Spline basis functions applied on inputs, shape (n_in*n_out, G+k, batch).
            
        Example:
            >>> layer = SplineLayer(n_in = 2, n_out = 5, k = 3,
            >>>                     G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                     base_basis = 1.0, base_spline = 1.0, base_res = 1.0,
            >>>                     pow_basis = 0.5, pow_spline = 0.5, pow_res = 0.5, 
            >>>                     rngs = nnx.Rngs(42))
            >>>
            >>> key = jax.random.PRNGKey(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
            >>>
            >>> output = layer.basis(x_batch)
        """
        
        grid = self.grid.item # shape (n_in, G+2k+1)
        x = jnp.expand_dims(x, axis=-1) # shape (batch, n_in, 1)

        # k = 0 case
        basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
        
        # Recursion done through iteration
        for K in range(1, self.k+1):
            left_term = (x - grid[:, :-(K + 1)]) / (grid[:, K:-1] - grid[:, :-(K + 1)])
            right_term = (grid[:, K + 1:] - x) / (grid[:, K + 1:] - grid[:, 1:(-K)])
            
            basis_splines = left_term * basis_splines[:, :, :-1] + right_term * basis_splines[:, :, 1:]

        return basis_splines


    def update_grid(self, x, G_new):
        """
        Performs a grid update given a new value for G (i.e., G_new) and adapts it to the given data, x. Additionally, re-initializes the c_basis parameters to a better estimate, based on the new grid.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).
            G_new (int):
                Size of the new grid (in terms of intervals).
            
        Example:
            >>> layer = SplineLayer(n_in = 2, n_out = 5, k = 3,
            >>>                     G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                     base_basis = 1.0, base_spline = 1.0, base_res = 1.0,
            >>>                     pow_basis = 0.5, pow_spline = 0.5, pow_res = 0.5, 
            >>>                     rngs = nnx.Rngs(42))
            >>>
            >>> key = jax.random.PRNGKey(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
            >>>
            >>> layer.update_grid(x=x_batch, G_new=5)
        """

        # Apply the inputs to the current grid to acquire y = Sum(ciBi(x)), where ci are
        # the current coefficients and Bi(x) are the current spline basis functions
        Bi = self.basis(x).transpose(1, 0, 2) # (n_in, batch, G+k)
        ci = self.c_basis.value.transpose(1, 2, 0) # (n_in, G+k, n_out)
        ciBi = jnp.einsum('ijk,ikm->ijm', Bi, ci) # (n_in, batch, n_out)

        # Update the grid
        self.grid.update(x, G_new)

        # Get the Bj(x) for the new grid
        Bj = self.basis(x).transpose(1, 0, 2) # (n_in, batch, G_new+k)

        # Solve for the new coefficients
        cj = solve_full_lstsq(Bj, ciBi) # (n_in, G_new+k, n_out)
        # Cast into shape (n_out, n_in, G_new+k)
        cj = cj.transpose(2, 0, 1)

        self.c_basis = nnx.Param(cj)


    def __call__(self, x):
        """
        The layer's forward pass.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).

        Returns:
            y (jnp.array):
                Output of the forward pass, corresponding to the weighted sum of the B-spline activation and the residual activation, shape (batch, n_out).
            
        Example:
            >>> layer = SplineLayer(n_in = 2, n_out = 5, k = 3,
            >>>                     G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                     base_basis = 1.0, base_spline = 1.0, base_res = 1.0,
            >>>                     pow_basis = 0.5, pow_spline = 0.5, pow_res = 0.5, 
            >>>                     rngs = nnx.Rngs(42))
            >>>
            >>> key = jax.random.PRNGKey(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
            >>>
            >>> output = layer(x_batch)
        """
        
        batch = x.shape[0]
        
        # Calculate residual activation
        res = self.residual(x) # (batch, n_in)
        
        # Multiply by trainable weights
        res_w = self.c_res.value # (n_out, n_in)
        full_res = jnp.matmul(res, res_w.T) # (batch, n_out)
        
        # Calculate spline basis
        Bi = self.basis(x) # (batch, n_in, G+k)
        spl = Bi.reshape(batch, -1) # (batch, n_in * (G+k))
        
        # Calculate spline coefficients
        spl_w = self.c_basis.value * self.c_spl[..., None] # (n_out, n_in, G+k)
        spl_w = spl_w.reshape(self.n_out, -1) # (n_out, n_in * (G+k))

        full_spl = jnp.matmul(spl, spl_w.T) # (batch, n_out)

        # Calculate the entire activation
        y = full_res + full_spl # (batch, n_out)
        
        return y