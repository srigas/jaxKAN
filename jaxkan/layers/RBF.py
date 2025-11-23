import jax
import jax.numpy as jnp

from flax import nnx

from typing import Union

from ..grids.RBFGrid import RBFGrid
from ..utils.general import solve_full_lstsq
        
        
class RBFLayer(nnx.Module):
    """
    RBFLayer class. Corresponds to the RBF version of KANs using different kernels.

    Attributes:
        n_in (int):
            Number of layer's incoming nodes.
        n_out (int):
            Number of layer's outgoing nodes.
        D (int):
            Number of basis functions.
        kernel (Union[dict, None]):
            Kernel to be used for the RBFs.
        grid_range (tuple):
            An initial range for the grid's ends, although adaptivity can completely change it.
        grid_e (float):
            Parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
        residual (Union[nnx.Module, None]):
            Function that is applied on samples to calculate residual activation.
        external_weights (bool):
            Boolean that controls if the trainable weights (n_out, n_in) should be applied to the activations.
        init_scheme (Union[dict, None]):
            Dictionary that defines how the trainable parameters of the layer are initialized.
        add_bias (bool):
            Boolean that controls wether bias terms are also included during the forward pass or not.
        seed (int):
            Random key selection for initializations wherever necessary.
    """
    
    def __init__(self, n_in: int = 2, n_out: int = 5, D: int = 5, kernel: Union[dict, None] = None,
                 grid_range: tuple = (-2.0, 2.0), grid_e: float = 1.0,
                 residual: Union[nnx.Module, None] = None, external_weights: bool = False,
                 init_scheme: Union[dict, None] = None, add_bias: bool = True, seed: int = 42):
        """
        Initializes a RBFLayer instance.
        
        Args:
            n_in (int):
                Number of layer's incoming nodes.
            n_out (int):
                Number of layer's outgoing nodes.
            D (int):
                Number of basis functions.
            kernel (Union[dict, None]):
                Kernel to be used for the RBFs.
            grid_range (tuple):
                An initial range for the grid's ends, although adaptivity can completely change it.
            grid_e (float):
                Parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases.
            residual (Union[nnx.Module, None]):
                Function that is applied on samples to calculate residual activation.
            external_weights (bool):
                Boolean that controls if the trainable weights (n_out, n_in) should be applied to the activations.
            init_scheme (Union[dict, None]):
                Dictionary that defines how the trainable parameters of the layer are initialized.
            add_bias (bool):
                Boolean that controls wether bias terms are also included during the forward pass or not.
            seed (int):
                Random key selection for initializations wherever necessary.
            
        Example:
            >>> layer = RBFLayer(n_in = 2, n_out = 5, D = 4, kernel = "gaussian",
            >>>                  grid_range = (-2.0, 2.0), grid_e = 1.0, residual = None,
            >>>                  external_weights = False, init_scheme = None, add_bias = True,
            >>>                  seed = 42)
        """
        if kernel is None:
            kernel = {"type" : "gaussian"}

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.D = D
        self.kernel = kernel
        self.grid_range = grid_range
        self.residual = residual

        # Setup nnx rngs
        self.rngs = nnx.Rngs(seed)

        # Initialize the grid
        self.grid = RBFGrid(n_nodes=n_in, D=D, grid_range=grid_range, grid_e=grid_e)

        # Add bias
        if add_bias == True:
            self.bias = nnx.Param(jnp.zeros((n_out,)))
        else:
            self.bias = None

        # If external_weights == True, we initialize weights for the activation functions equal to unity
        if external_weights == True:
            self.c_ext = nnx.Param(
                nnx.initializers.ones(
                    self.rngs.params(), (self.n_out, self.n_in), jnp.float32)
            )
        else:
            self.c_ext = None

        # Initialize the remaining trainable parameters, based on the selected initialization scheme
        c_res, c_basis = self._initialize_params(init_scheme, seed)

        self.c_basis = nnx.Param(c_basis)

        if residual is not None:
            self.c_res = nnx.Param(c_res)
            

    def basis(self, x):
        """
        Based on the degree and kernel, the values of the radial basis functions are calculated on the input.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).

        Returns:
            rbf (jnp.array):
                Radial basis functions applied on inputs, shape (batch, n_in, D).
            
        Example:
            >>> layer = RBFLayer(n_in = 2, n_out = 5, D = 4, kernel = "gaussian",
            >>>                  grid_range = (-2.0, 2.0), grid_e = 1.0, residual = None,
            >>>                  external_weights = False, init_scheme = None, add_bias = True,
            >>>                  seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-2.0, maxval=2.0)
            >>>
            >>> output = layer.basis(x_batch)
        """
        kernel_type = self.kernel.get("type", "gaussian")
        grid_item = self.grid.item

        # Gaussian kernel
        if kernel_type == "gaussian":
            std = self.kernel.get("std", 1.0)
            
            rbf = jnp.exp(-0.5 * ((x[..., :, None] - grid_item[None, ...]) / std) ** 2)

        # Other kernel
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        return rbf


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

        # Default initialization
        if init_type == "default":

            if self.residual is not None:
                c_res = nnx.initializers.glorot_uniform(in_axis=-1, out_axis=-2)(
                    self.rngs.params(), (self.n_out, self.n_in), jnp.float32
                )
            
            std = 1.0/jnp.sqrt(self.n_in * self.D)
            c_basis = nnx.initializers.truncated_normal(stddev=std)(
                self.rngs.params(), (self.n_out, self.n_in, self.D), jnp.float32
            )

        # Custom power law initialization
        # c_basis ~ N(0, s_b), s_b = const_b / [ (D+1)^pow_b1 * n_in^pow_b2 ]
        # c_res ~ N(0, s_r), s_r = const_r / [ (D+1)^pow_r1 * n_in^pow_r2 ]
        elif init_type == "power":

            const_b = init_scheme.get("const_b", 1.0)
            pow_b1 = init_scheme.get("pow_b1", 0.5)
            pow_b2 = init_scheme.get("pow_b2", 0.5)

            if self.residual is not None:
                basis_term = self.D + 1
                
                const_r = init_scheme.get("const_r", 1.0)
                pow_r1 = init_scheme.get("pow_r1", 0.5)
                pow_r2 = init_scheme.get("pow_r2", 0.5)
                
                std_res = const_r / ( (basis_term**pow_r1) * (self.n_in**pow_r2) )
                c_res = nnx.initializers.normal(stddev=std_res)(
                    self.rngs.params(), (self.n_out, self.n_in), jnp.float32
                )                
            else:
                basis_term = self.D

            std_b = const_b / ( (basis_term**pow_b1) * (self.n_in**pow_b2) )
            c_basis = nnx.initializers.normal(stddev=std_b)(
                self.rngs.params(), (self.n_out, self.n_in, self.D), jnp.float32
            )

        # LeCun-like initialization, where Var[in] = Var[out]
        elif init_type == "lecun":

            key = jax.random.key(seed)

            # Also get distribution type
            distrib = init_scheme.get("distribution", "uniform")

            if distrib is None:
                distrib = "uniform"

            sample_size = init_scheme.get("sample_size", 10000)

            if sample_size is None:
                sample_size = 10000

            # Generate a sample of points
            if distrib == "uniform":
                sample = jax.random.uniform(key, shape=(sample_size,), minval=-1.0, maxval=1.0)
            elif distrib == "normal":
                sample = jax.random.normal(key, shape=(sample_size,))

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
                scale = self.n_in * (self.D + 1)
                # Apply the residual function
                y_res = self.residual(sample)
                # Calculate the average of residual^2(x)
                y_res_sq = y_res**2
                y_res_sq_mean = y_res_sq.mean().item()

                std_res = gain/jnp.sqrt(scale*y_res_sq_mean)
                c_res = nnx.initializers.normal(stddev=std_res)(self.rngs.params(), (self.n_out, self.n_in), jnp.float32)
            
            else:
                # Variance equipartitioned across G+k terms
                scale = self.n_in * self.D

            std_b = gain/jnp.sqrt(scale*y_b_sq_mean)
            c_basis = nnx.initializers.normal(stddev=std_b)(
                self.rngs.params(), (self.n_out, self.n_in, self.D), jnp.float32
            )

        # Glorot-like initialization, where we attempt to balance Var[in] = Var[out] and Var[δin] = Var[δout]
        elif init_type == "glorot":

            key = jax.random.key(seed)

            # Also get distribution type
            distrib = init_scheme.get("distribution", "uniform")

            if distrib is None:
                distrib = "uniform"

            sample_size = init_scheme.get("sample_size", 10000)

            if sample_size is None:
                sample_size = 10000

            # Generate a sample of points
            if distrib == "uniform":
                sample = jax.random.uniform(key, shape=(sample_size,), minval=-1.0, maxval=1.0)
            elif distrib == "normal":
                sample = jax.random.normal(key, shape=(sample_size,))

            # Finally get gain
            gain = init_scheme.get("gain", None)
            if gain is None:
                gain = sample.std().item()

            # Extend the sample to be able to pass through basis
            sample_ext = jnp.tile(sample[:, None], (1, self.n_in))

            # ------------- Basis function gradient ----------------------
            # Define a scalar version of the basis function
            def basis_scalar(x):
                return self.basis(jnp.array([[x]]))[0, 0, :]

            # Create a Jacobian function for the scalar wrapper
            jac_basis = jax.jacobian(basis_scalar)

            num_batches = 20
            batch_size = sample_size // num_batches
            grad_sq_accum = 0.0
            
            for i in range(num_batches):
                batch = sample[i*batch_size:(i+1)*batch_size]
                grad_batch = jax.vmap(jac_basis)(batch)
                grad_sq_accum += (grad_batch**2).sum()

            # Calculate E[B'_m^2(x)]
            grad_b_sq_mean = grad_sq_accum / (sample_size * self.D)
            # ------------------------------------------------------------
            
            # Calculate E[B_m^2(x)]
            y_b = self.basis(sample_ext)
            y_b_sq = y_b**2
            y_b_sq_mean = y_b_sq.mean().item()
            
            # Deal with residual if available
            if self.residual is not None:
                # Variance equipartitioned across all terms
                scale_in = self.n_in * (self.D + 1)
                scale_out = self.n_out * (self.D + 1)

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
                scale_in = self.n_in * self.D
                scale_out = self.n_out * self.D

            std_b = gain*jnp.sqrt(2.0 / (scale_in*y_b_sq_mean + scale_out*grad_b_sq_mean))
            c_basis = nnx.initializers.normal(stddev=std_b)(
                self.rngs.params(), (self.n_out, self.n_in, self.D), jnp.float32
            )

        # Glorot-like initialization as presented in the paper "Towards Deep Physics-Informed Kolmogorov-Arnold Networks"
        # The main difference is that we do not aggregate over all sigmas, each mode has its own, hence "fine grained"
        elif init_type == "glorot_fine":

            key = jax.random.key(seed)

            # Also get distribution type
            distrib = init_scheme.get("distribution", "uniform")

            if distrib is None:
                distrib = "uniform"

            sample_size = init_scheme.get("sample_size", 10000)

            if sample_size is None:
                sample_size = 10000

            # Generate a sample of points
            if distrib == "uniform":
                sample = jax.random.uniform(key, shape=(sample_size,), minval=-1.0, maxval=1.0)
            elif distrib == "normal":
                sample = jax.random.normal(key, shape=(sample_size,))

            # Finally get gain
            gain = init_scheme.get("gain", None)
            if gain is None:
                gain = sample.std().item()

            # Extend the sample to be able to pass through basis
            sample_ext = jnp.tile(sample[:, None], (1, self.n_in))

            # ------------- Basis functions ------------------------

            # μ⁽0⁾ₘ (⟨B_m²⟩) 
            B      = self.basis(sample_ext)
            mu0    = (B**2).mean(axis=(0, 1))

            # μ⁽1⁾ₘ (⟨B'_m²⟩)            

            # Define a scalar version of the basis function
            basis_scalar = lambda x: self.basis(jnp.array([[x]]))[0, 0, :]

            jac_basis = jax.jacrev(basis_scalar)
            mu1  = (jax.vmap(jac_basis)(sample)**2).mean(axis=0)

            # ------------- Residual function ----------------------
            # Deal with residual if available - same as simple glorot
            if self.residual is not None:
                # Variance equipartitioned across all terms
                scale_in = self.n_in * (self.D + 1)
                scale_out = self.n_out * (self.D + 1)

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
                scale_in = self.n_in * self.D
                scale_out = self.n_out * self.D

            sigma_vec = gain * jnp.sqrt(1.0 / (scale_in*mu0 + scale_out*mu1))

            noise = nnx.initializers.normal(stddev=1.0)(
                self.rngs.params(), (self.n_out, self.n_in, self.D), jnp.float32
            )

            c_basis  = noise * sigma_vec

        # Custom initialization, where the user inputs pre-determined arrays
        elif init_type == "custom":
            
            if self.residual is not None:
                c_res = init_scheme.get("c_res", None)

            c_basis = init_scheme.get("c_basis", None)
            
        else:
            raise ValueError(f"Unknown initialization method: {init_type}")

        return c_res, c_basis


    def update_grid(self, x, D_new):
        """
        Performs a grid update given a new value for D (i.e., D_new) and adapts it to the given data, x. Additionally, re-initializes the c_basis parameters to a better estimate, based on the new grid.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).
            D_new (int):
                New number of spline basis functions.
            
        Example:
            >>> layer = RBFLayer(n_in = 2, n_out = 5, D = 4, kernel = "gaussian",
            >>>                  grid_range = (-2.0, 2.0), grid_e = 1.0, residual = None,
            >>>                  external_weights = False, init_scheme = None, add_bias = True,
            >>>                  seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-2.0, maxval=2.0)
            >>>
            >>> layer.update_grid(x=x_batch, D_new=8)
        """

        # Apply the inputs to the current grid to acquire y = Sum(ciBi(x)), where ci are
        # the current coefficients and Bi(x) are the current Legendre basis functions
        Bi = self.basis(x).transpose(1, 0, 2) # (n_in, batch, D)
        ci = self.c_basis[...].transpose(1, 2, 0) # (n_in, D, n_out)
        ciBi = jnp.einsum('ijk,ikm->ijm', Bi, ci) # (n_in, batch, n_out)

        # Update the grid
        self.grid.update(x, D_new)

        # Get the Bj(x) for the degree order
        Bj = self.basis(x).transpose(1, 0, 2) # (n_in, batch, D_new)

        # Solve for the new coefficients
        cj = solve_full_lstsq(Bj, ciBi) # (n_in, D_new, n_out)
        # Cast into shape (n_out, n_in, D_new+1)
        cj = cj.transpose(2, 0, 1)

        self.c_basis = nnx.Param(cj)
        self.D = D_new


    def __call__(self, x):
        """
        The layer's forward pass.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).

        Returns:
            y (jnp.array):
                Output of the forward pass, shape (batch, n_out).
            
        Example:
            >>> layer = RBFLayer(n_in = 2, n_out = 5, D = 4, kernel = "gaussian",
            >>>                  grid_range = (-2.0, 2.0), grid_e = 1.0, residual = None,
            >>>                  external_weights = False, init_scheme = None, add_bias = True,
            >>>                  seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-2.0, maxval=2.0)
            >>>
            >>> output = layer(x_batch)
        """
        
        batch = x.shape[0]
        
        # Calculate basis activations
        Bi = self.basis(x) # (batch, n_in, D)
        act = Bi.reshape(batch, -1) # (batch, n_in * D)

        # Check if external_weights == True
        if self.c_ext is not None:
            act_w = self.c_basis[...] * self.c_ext[..., None] # (n_out, n_in, D)
        else:
            act_w = self.c_basis[...]
        
        # Calculate coefficients
        act_w = act_w.reshape(self.n_out, -1) # (n_out, n_in * D)

        y = jnp.matmul(act, act_w.T) # (batch, n_out)

        # Check if there is a residual function
        if self.residual is not None:
            # Calculate residual activation
            res = self.residual(x) # (batch, n_in)
            # Multiply by trainable weights
            res_w = self.c_res[...] # (n_out, n_in)
            full_res = jnp.matmul(res, res_w.T) # (batch, n_out)

            y += full_res # (batch, n_out)

        if self.bias is not None:
            y += self.bias[...] # (batch, n_out)
        
        return y
