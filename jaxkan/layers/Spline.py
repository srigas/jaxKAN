import jax
import jax.numpy as jnp

from flax import nnx

from typing import Union

from ..grids.BaseGrid import BaseGrid
from ..grids.SplineGrid import SplineGrid

from ..utils.general import solve_full_lstsq


class BaseLayer(nnx.Module):
    """
    BaseLayer class. Corresponds to the original spline-based KAN Layer introduced in the original version of KAN. Ref: https://arxiv.org/abs/2404.19756

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
        residual (Union[nnx.Module, None]):
            Function that is applied on samples to calculate residual activation.
        external_weights (bool):
            Boolean that controls if the trainable weights of shape (n_out, n_in) applied to the splines should be used.
        init_scheme (Union[dict, None]):
            Dictionary that defines how the trainable parameters of the layer are initialized.
        add_bias (bool):
            Boolean that controls wether bias terms are also included during the forward pass or not.
        seed (int):
            Random key selection for initializations wherever necessary.
    """
    
    def __init__(self,
                 n_in: int = 2, n_out: int = 5, k: int = 3,
                 G: int = 3, grid_range: tuple = (-1,1), grid_e: float = 0.05,
                 residual: Union[nnx.Module, None] = nnx.silu,
                 external_weights: bool = True, init_scheme: Union[dict, None] = None,
                 add_bias: bool = True, seed: int = 42
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
            residual (Union[nnx.Module, None]):
                Function that is applied on samples to calculate residual activation.
            external_weights (bool):
                Boolean that controls if the trainable weights of shape (n_out, n_in) applied to the splines should be used.
            init_scheme (Union[dict, None]):
                Dictionary that defines how the trainable parameters of the layer are initialized.
            add_bias (bool):
                Boolean that controls wether bias terms are also included during the forward pass or not.
            seed (int):
                Random key selection for initializations wherever necessary.
            
        Example:
            >>> layer = BaseLayer(n_in = 2, n_out = 5, k = 3,
            >>>                   G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                   external_weights = True, init_scheme = None, add_bias = True,
            >>>                   seed = 42)
        """

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.grid_range = grid_range
        self.residual = residual

        # Setup nnx rngs
        self.rngs = nnx.Rngs(seed)

        # Initialize the grid
        self.grid = BaseGrid(n_in=n_in, n_out=n_out, k=k, G=G, grid_range=grid_range, grid_e=grid_e)

        # If external_weights == True, we initialize weights for the spline functions equal to unity
        if external_weights == True:
            self.c_spl = nnx.Param(
                nnx.initializers.ones(
                    self.rngs.params(), (self.n_out, self.n_in), jnp.float32)
            )
        else:
            self.c_spl = None

        # Initialize the remaining trainable parameters, based on the selected initialization scheme
        c_res, c_basis = self._initialize_params(init_scheme, seed)

        self.c_basis = nnx.Param(c_basis)
        
        if residual is not None:
            self.c_res = nnx.Param(c_res)

        # Add bias
        if add_bias == True:
            self.bias = nnx.Param(jnp.zeros((n_out,)))
        else:
            self.bias = None

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
            >>> layer = BaseLayer(n_in = 2, n_out = 5, k = 3,
            >>>                   G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                   external_weights = True, init_scheme = None, add_bias = True,
            >>>                   seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = layer.basis(x_batch)
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


    def _initialize_params(self, init_scheme, seed):
        """
        Initializes the c_res (if residual is activated) and c_basis trainable parameters (only used in __init__)

        Args:
            init_scheme (Union[dict, None]):
                Dictionary that defines how the trainable parameters of the layer are initialized. Options: "default", "lecun", "custom"
        """

        if init_scheme is None:
            init_scheme = {"type" : "default"}

        init_type = init_scheme.get("type", "default")

        # Case where no residual is used
        if self.residual is None:
            c_res = None

        # Default initialization presented in original paper
        if init_type == "default":

            if self.residual is not None:
                c_res = nnx.initializers.glorot_uniform(in_axis=-1, out_axis=-2)(
                    self.rngs.params(), (self.n_out, self.n_in), jnp.float32
                )
            
            std = init_scheme.get("std", 0.1)
            c_basis = nnx.initializers.normal(stddev=std)(
                self.rngs.params(), (self.n_in * self.n_out, self.grid.G + self.k), jnp.float32
            )

        # LeCun-like initialization, where Var[in] = Var[out]
        # Thanks are owed to Verma Dhruv for collaborating on this initialization type
        elif init_type == "lecun":

            key = jax.random.key(seed)

            # Also get distribution type
            distrib = init_scheme.get("distribution", "uniform")

            if distrib is None:
                distrib = "uniform"

            # Generate a sample of 10^5 points
            if distrib == "uniform":
                sample = jax.random.uniform(key, shape=(100000,), minval=-1.0, maxval=1.0)
            elif distrib == "normal":
                sample = jax.random.normal(key, shape=(100000,))

            # Finally get gain
            gain = init_scheme.get("gain", None)
            if gain is None:
                gain = sample.std().item()
            
            # Extend the sample to be able to pass through basis
            sample_ext = jnp.tile(sample[:, None], (1, self.n_in))
            # Calculate B_m^2(x)
            y_b = self.basis(sample_ext)
            # Calculate the average of B_m^2(x)
            y_b_sq = y_b**2
            y_b_sq_mean = y_b_sq.mean().item()

            if self.residual is not None:
                # Variance equipartitioned across all terms
                scale = self.n_in * (self.grid.G + self.k + 1)
                # Apply the residual function
                y_res = self.residual(sample)
                # Calculate the average of residual^2(x)
                y_res_sq = y_res**2
                y_res_sq_mean = y_res_sq.mean().item()

                std_res = gain/jnp.sqrt(scale*y_res_sq_mean)
                c_res = nnx.initializers.normal(stddev=std_res)(self.rngs.params(), (self.n_out, self.n_in), jnp.float32)
            
            else:
                # Variance equipartitioned across G+k terms
                scale = self.n_in * (self.grid.G + self.k)

            std_b = gain/jnp.sqrt(scale*y_b_sq_mean)
            c_basis = nnx.initializers.normal(stddev=std_b)(
                self.rngs.params(), (self.n_in * self.n_out, self.grid.G + self.k), jnp.float32
            )

        # Glorot-like initialization, where we attempt to balance Var[in] = Var[out] and Var[δin] = Var[δout]
        elif init_type == "glorot":

            key = jax.random.key(seed)

            # Also get distribution type
            distrib = init_scheme.get("distribution", "uniform")

            if distrib is None:
                distrib = "uniform"

            # Generate a sample of 10^5 points
            if distrib == "uniform":
                sample = jax.random.uniform(key, shape=(100000,), minval=-1.0, maxval=1.0)
            elif distrib == "normal":
                sample = jax.random.normal(key, shape=(100000,))

            # Finally get gain
            gain = init_scheme.get("gain", None)
            if gain is None:
                gain = sample.std().item()

            # Extend the sample to be able to pass through basis
            sample_ext = jnp.tile(sample[:, None], (1, self.n_in))

            # ------------- Basis function gradient ----------------------
            # Define a scalar version of the basis function
            def u(x):
                return self.basis(x)

            def basis_scalar(x):
                # Convert scalar x into an array with shape (1, 1) so that it matches the expected (batch, n_in) shape.
                x_arr = jnp.array([[x]])
                # Call the existing basis method.
                # This returns an array of shape (1, n_in, D) (or (1, n_in, D+1)) depending on the bias.
                out = u(x_arr)
                # Since the n_in dimension is redundant (due to tiling), extract the first element
                # from both the batch and n_in dimensions.
                return out[0, 0, :]

            # Create a Jacobian function for the scalar wrapper
            jac_basis = jax.jacobian(basis_scalar)

            # Use jax.vmap twice to vectorize over batch and n_in.
            grad_basis = jax.vmap(jax.vmap(jac_basis))(sample_ext)
            # ------------------------------------------------------------
            
            # Calculate E[B_m^2(x)]
            y_b = u(sample_ext)
            y_b_sq = y_b**2
            y_b_sq_mean = y_b_sq.mean().item()

            # Calculate E[B'_m^2(x)]
            grad_b_sq = grad_basis**2
            grad_b_sq_mean = grad_b_sq.mean().item()
            
            # Deal with residual if available
            if self.residual is not None:
                # Variance equipartitioned across all terms
                scale_in = self.n_in * (self.grid.G + self.k + 1)
                scale_out = self.n_out * (self.grid.G + self.k + 1)

                # ------------- Residual function gradient ----------------------
                # Similar idea to the basis function
                def r(x):
                    return self.residual(x)

                jac_res = jax.jacobian(r)
                
                grad_res = jax.vmap(jac_res)(sample)
                # ------------------------------------------------------------
                
                # Calculate E[R^2(x)]
                y_res = self.residual(sample)
                y_res_sq = y_res**2
                y_res_sq_mean = y_res_sq.mean().item()

                # Calculate E[R'^2(x)]
                grad_res_sq = grad_res**2
                grad_res_sq_mean = grad_res_sq.mean().item()

                std_res = gain*jnp.sqrt(2.0 / (scale_in*y_res_sq_mean + scale_out*grad_res_sq_mean))
                c_res = nnx.initializers.normal(stddev=std_res)(self.rngs.params(), (self.n_out, self.n_in), jnp.float32)
            
            else:
                # Variance equipartitioned across G+k terms
                scale_in = self.n_in * (self.grid.G + self.k)
                scale_out = self.n_out * (self.grid.G + self.k)

            std_b = gain*jnp.sqrt(2.0 / (scale_in*y_b_sq_mean + scale_out*grad_b_sq_mean))
            c_basis = nnx.initializers.normal(stddev=std_b)(
                self.rngs.params(), (self.n_in * self.n_out, self.grid.G + self.k), jnp.float32
            )

        # Custom initialization, where the user inputs pre-determined arrays
        elif init_type == "custom":
            
            if self.residual is not None:
                c_res = init_scheme.get("c_res", None)

            c_basis = init_scheme.get("c_basis", None)
            
        else:
            raise ValueError(f"Unknown initialization method: {init_type}")

        return c_res, c_basis


    def update_grid(self, x, G_new):
        """
        Performs a grid update given a new value for G (i.e., G_new) and adapts it to the given data, x. Additionally, re-initializes the c_basis parameters to a better estimate, based on the new grid.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).
            G_new (int):
                Size of the new grid (in terms of intervals).
            
        Example:
            >>> layer = BaseLayer(n_in = 2, n_out = 5, k = 3,
            >>>                   G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                   external_weights = True, init_scheme = None, add_bias = True,
            >>>                   seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> layer.update_grid(x=x_batch, G_new=5)
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
            x (jnp.array):
                Inputs, shape (batch, n_in).

        Returns:
            y (jnp.array):
                Output of the forward pass, corresponding to the weighted sum of the B-spline activation and the residual activation, shape (batch, n_out).
            
        Example:
            >>> layer = BaseLayer(n_in = 2, n_out = 5, k = 3,
            >>>                   G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                   external_weights = True, init_scheme = None, add_bias = True,
            >>>                   seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = layer(x_batch)
        """
        
        batch = x.shape[0]
            
        # Calculate spline basis activations
        Bi = self.basis(x) # (n_in*n_out, G+k, batch)
        ci = self.c_basis.value # (n_in*n_out, G+k)
        # Calculate spline activation
        spl = jnp.einsum('ij,ijk->ik', ci, Bi) # (n_in*n_out, batch)
        # Transpose to shape (batch, n_in*n_out)
        spl = jnp.transpose(spl, (1,0))

        # Check if external_weights == True
        if self.c_spl is not None:
            # Reshape constants to (1, n_in*n_out)
            cnst_spl = jnp.expand_dims(self.c_spl.value, axis=0).reshape((1, self.n_in * self.n_out))
            # Calculate spline term
            y = cnst_spl * spl # (batch, n_in*n_out)
        else:
            y = spl # (batch, n_in*n_out)

        # Check if there is a residual function
        if self.residual is not None:
            # Extend x to shape (batch, n_in*n_out)
            x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in * self.n_out))
            # Transpose to shape (n_in*n_out, batch)
            x_ext = jnp.transpose(x_ext, (1, 0))
            # Calculate residual activation - shape (batch, n_in*n_out)
            res = jnp.transpose(self.residual(x_ext), (1,0))
            # Reshape constant to (1, n_in*n_out)
            cnst_res = jnp.expand_dims(self.c_res.value, axis=0).reshape((1, self.n_in * self.n_out))
            # Calculate the entire activation
            y += cnst_res * res # (batch, n_in*n_out)
        
        # Reshape and sum
        y_reshaped = jnp.reshape(y, (batch, self.n_out, self.n_in))
        y = jnp.sum(y_reshaped, axis=2)  # (batch, n_out)

        if self.bias is not None:
            y += self.bias.value  # (batch, n_out)
        
        return y


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
        residual (Union[nnx.Module, None]):
            Function that is applied on samples to calculate residual activation.
        external_weights (bool):
            Boolean that controls if the trainable weights of shape (n_out, n_in) applied to the splines should be used.
        init_scheme (Union[dict, None]):
            Dictionary that defines how the trainable parameters of the layer are initialized.
        add_bias (bool):
            Boolean that controls wether bias terms are also included during the forward pass or not.
        seed (int):
            Random key selection for initializations wherever necessary.
    """
    
    def __init__(self,
                 n_in: int = 2, n_out: int = 5, k: int = 3,
                 G: int = 3, grid_range: tuple = (-1,1), grid_e: float = 0.05,
                 residual: Union[nnx.Module, None] = nnx.silu,
                 external_weights: bool = True, init_scheme: Union[dict, None] = None,
                 add_bias: bool = True, seed: int = 42
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
            residual (Union[nnx.Module, None]):
                Function that is applied on samples to calculate residual activation.
            external_weights (bool):
                Boolean that controls if the trainable weights of shape (n_out, n_in) applied to the splines should be used.
            init_scheme (Union[dict, None]):
                Dictionary that defines how the trainable parameters of the layer are initialized.
            add_bias (bool):
                Boolean that controls wether bias terms are also included during the forward pass or not.
            seed (int):
                Random key selection for initializations wherever necessary.
            
        Example:
            >>> layer = SplineLayer(n_in = 2, n_out = 5, k = 3,
            >>>                     G = 3, grid_range = (-1,1), grid_e = 0.05, residual = nnx.silu,
            >>>                     external_weights = True, init_scheme = None, add_bias = True,
            >>>                     seed = 42)
        """

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.grid_range = grid_range
        self.residual = residual

        # Setup nnx rngs
        self.rngs = nnx.Rngs(seed)

        # Initialize the grid - shape (n_in, G+2k+1)
        self.grid = SplineGrid(n_nodes=n_in, k=k, G=G, grid_range=grid_range, grid_e=grid_e)

        # If external_weights == True, we initialize weights for the spline functions equal to unity
        if external_weights == True:
            self.c_spl = nnx.Param(
                nnx.initializers.ones(
                    self.rngs.params(), (self.n_out, self.n_in), jnp.float32)
            )
        else:
            self.c_spl = None

        # Initialize the remaining trainable parameters, based on the selected initialization scheme
        c_res, c_basis = self._initialize_params(init_scheme, seed)

        self.c_basis = nnx.Param(c_basis)
        
        if residual is not None:
            self.c_res = nnx.Param(c_res)

        # Add bias
        if add_bias == True:
            self.bias = nnx.Param(jnp.zeros((n_out,)))
        else:
            self.bias = None

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
            >>>                     external_weights = True, init_scheme = None, add_bias = True,
            >>>                     seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
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


    def _initialize_params(self, init_scheme, seed):
        """
        Initializes the c_res (if residual is activated) and c_basis trainable parameters (only used in __init__)

        Args:
            init_scheme (Union[dict, None]):
                Dictionary that defines how the trainable parameters of the layer are initialized. Options: "default", "lecun", "custom"
        """

        if init_scheme is None:
            init_scheme = {"type" : "default"}

        init_type = init_scheme.get("type", "default")

        # Case where no residual is used
        if self.residual is None:
            c_res = None

        # Default initialization presented in original paper
        if init_type == "default":

            if self.residual is not None:
                c_res = nnx.initializers.glorot_uniform(in_axis=-1, out_axis=-2)(
                    self.rngs.params(), (self.n_out, self.n_in), jnp.float32
                )
            
            std = init_scheme.get("std", 0.1)
            c_basis = nnx.initializers.normal(stddev=std)(
                self.rngs.params(), (self.n_out, self.n_in, self.grid.G + self.k), jnp.float32
            )

        # LeCun-like initialization, where Var[in] = Var[out]
        elif init_type == "lecun":

            key = jax.random.key(seed)

            # Also get distribution type
            distrib = init_scheme.get("distribution", "uniform")

            # Generate a sample of 10^5 points
            if distrib == "uniform":
                sample = jax.random.uniform(key, shape=(100000,), minval=-1.0, maxval=1.0)
            elif distrib == "normal":
                sample = jax.random.normal(key, shape=(100000,))

            # Finally get gain
            gain = init_scheme.get("gain", None)
            if gain is None:
                gain = sample.std().item()
            
            # Extend the sample to be able to pass through basis
            sample_ext = jnp.tile(sample[:, None], (1, self.n_in))
            # Calculate B_m^2(x)
            y_b = self.basis(sample_ext)
            # Calculate the average of B_m^2(x)
            y_b_sq = y_b**2
            y_b_sq_mean = y_b_sq.mean().item()

            if self.residual is not None:
                # Variance equipartitioned across all terms
                scale = self.n_in * (self.grid.G + self.k + 1)
                # Apply the residual function
                y_res = self.residual(sample)
                # Calculate the average of residual^2(x)
                y_res_sq = y_res**2
                y_res_sq_mean = y_res_sq.mean().item()

                std_res = gain/jnp.sqrt(scale*y_res_sq_mean)
                c_res = nnx.initializers.normal(stddev=std_res)(self.rngs.params(), (self.n_out, self.n_in), jnp.float32)
            
            else:
                # Variance equipartitioned across G+k terms
                scale = self.n_in * (self.grid.G + self.k)

            std_b = gain/jnp.sqrt(scale*y_b_sq_mean)
            c_basis = nnx.initializers.normal(stddev=std_b)(
                self.rngs.params(), (self.n_out, self.n_in, self.grid.G + self.k), jnp.float32
            )

        # Glorot-like initialization, where we attempt to balance Var[in] = Var[out] and Var[δin] = Var[δout]
        elif init_type == "glorot":

            key = jax.random.key(seed)

            # Also get distribution type
            distrib = init_scheme.get("distribution", "uniform")

            # Generate a sample of 10^5 points
            if distrib == "uniform":
                sample = jax.random.uniform(key, shape=(100000,), minval=-1.0, maxval=1.0)
            elif distrib == "normal":
                sample = jax.random.normal(key, shape=(100000,))

            # Finally get gain
            gain = init_scheme.get("gain", None)
            if gain is None:
                gain = sample.std().item()

            # Extend the sample to be able to pass through basis
            sample_ext = jnp.tile(sample[:, None], (1, self.n_in))

            # ------------- Basis function gradient ----------------------
            # Define a scalar version of the basis function
            def u(x):
                return self.basis(x)

            def basis_scalar(x):
                # Convert scalar x into an array with shape (1, 1) so that it matches the expected (batch, n_in) shape.
                x_arr = jnp.array([[x]])
                # Call the existing basis method.
                # This returns an array of shape (1, n_in, D) (or (1, n_in, D+1)) depending on the bias.
                out = u(x_arr)
                # Since the n_in dimension is redundant (due to tiling), extract the first element
                # from both the batch and n_in dimensions.
                return out[0, 0, :]

            # Create a Jacobian function for the scalar wrapper
            jac_basis = jax.jacobian(basis_scalar)

            # Use jax.vmap twice to vectorize over batch and n_in.
            grad_basis = jax.vmap(jax.vmap(jac_basis))(sample_ext)
            # ------------------------------------------------------------
            
            # Calculate E[B_m^2(x)]
            y_b = u(sample_ext)
            y_b_sq = y_b**2
            y_b_sq_mean = y_b_sq.mean().item()

            # Calculate E[B'_m^2(x)]
            grad_b_sq = grad_basis**2
            grad_b_sq_mean = grad_b_sq.mean().item()
            
            # Deal with residual if available
            if self.residual is not None:
                # Variance equipartitioned across all terms
                scale_in = self.n_in * (self.grid.G + self.k + 1)
                scale_out = self.n_out * (self.grid.G + self.k + 1)

                # ------------- Residual function gradient ----------------------
                # Similar idea to the basis function
                def r(x):
                    return self.residual(x)

                jac_res = jax.jacobian(r)
                
                grad_res = jax.vmap(jac_res)(sample)
                # ------------------------------------------------------------
                
                # Calculate E[R^2(x)]
                y_res = self.residual(sample)
                y_res_sq = y_res**2
                y_res_sq_mean = y_res_sq.mean().item()

                # Calculate E[R'^2(x)]
                grad_res_sq = grad_res**2
                grad_res_sq_mean = grad_res_sq.mean().item()

                std_res = gain*jnp.sqrt(2.0 / (scale_in*y_res_sq_mean + scale_out*grad_res_sq_mean))
                c_res = nnx.initializers.normal(stddev=std_res)(self.rngs.params(), (self.n_out, self.n_in), jnp.float32)
            
            else:
                # Variance equipartitioned across G+k terms
                scale_in = self.n_in * (self.grid.G + self.k)
                scale_out = self.n_out * (self.grid.G + self.k)

            std_b = gain*jnp.sqrt(2.0 / (scale_in*y_b_sq_mean + scale_out*grad_b_sq_mean))
            c_basis = nnx.initializers.normal(stddev=std_b)(
                self.rngs.params(), (self.n_out, self.n_in, self.grid.G + self.k), jnp.float32
            )

        # Custom initialization, where the user inputs pre-determined arrays
        elif init_type == "custom":
            
            if self.residual is not None:
                c_res = init_scheme.get("c_res", None)

            c_basis = init_scheme.get("c_basis", None)
            
        else:
            raise ValueError(f"Unknown initialization method: {init_type}")

        return c_res, c_basis


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
            >>>                     external_weights = True, init_scheme = None, add_bias = True,
            >>>                     seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
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
            >>>                     external_weights = True, init_scheme = None, add_bias = True,
            >>>                     seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = layer(x_batch)
        """
        
        batch = x.shape[0]
        
        # Calculate spline basis
        Bi = self.basis(x) # (batch, n_in, G+k)
        spl = Bi.reshape(batch, -1) # (batch, n_in * (G+k))

        # Check if external_weights == True
        if self.c_spl is not None:
            spl_w = self.c_basis.value * self.c_spl[..., None] # (n_out, n_in, G+k)
        else:
            spl_w = self.c_basis.value

        # Reshape spline coefficients
        spl_w = spl_w.reshape(self.n_out, -1) # (n_out, n_in * (G+k))

        y = jnp.matmul(spl, spl_w.T) # (batch, n_out)
        
        # Check if there is a residual function
        if self.residual is not None:
            # Calculate residual activation
            res = self.residual(x) # (batch, n_in)
        
            # Multiply by trainable weights
            res_w = self.c_res.value # (n_out, n_in)
            full_res = jnp.matmul(res, res_w.T) # (batch, n_out)

            y += full_res # (batch, n_out)

        if self.bias is not None:
            y += self.bias.value  # (batch, n_out)
        
        return y
        