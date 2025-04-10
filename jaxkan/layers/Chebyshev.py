import jax
import jax.numpy as jnp

from flax import nnx

from typing import Union

from ..utils.polynomials import Cb
from ..utils.general import solve_full_lstsq
        
        
class ChebyshevLayer(nnx.Module):
    """
    ChebyshevLayer class. Corresponds to the Chebyshev version of KANs and comes in three "flavors":
        "default": the version presented in https://arxiv.org/pdf/2405.07200
        "modified": the version presented in https://www.sciencedirect.com/science/article/pii/S0045782524005462
        "exact": uses pre-defined functions for higher efficiency, but cannot scale up to arbitrary degrees

    Attributes:
        n_in (int):
            Number of layer's incoming nodes.
        n_out (int):
            Number of layer's outgoing nodes.
        D (int):
            Degree of Chebyshev polynomial (1st kind).
        flavor (Union[str, None]):
            One of "default", "modified", or "exact" - chooses basis implementation.
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
    
    def __init__(self, n_in: int = 2, n_out: int = 5, D: int = 5, flavor: Union[str, None] = None,
                 residual: Union[nnx.Module, None] = None, external_weights: bool = False,
                 init_scheme: Union[dict, None] = None, add_bias: bool = True, seed: int = 42):
        """
        Initializes a ChebyshevLayer instance.
        
        Args:
            n_in (int):
                Number of layer's incoming nodes.
            n_out (int):
                Number of layer's outgoing nodes.
            D (int):
                Degree of Chebyshev polynomial (1st kind).
            flavor (Union[str, None]):
                One of "default", "modified", or "exact" - chooses basis implementation.
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
            >>> layer = ChebyshevLayer(n_in = 2, n_out = 5, D = 5, flavor = "default", 
            >>>                        residual = None, external_weights = False, init_scheme = None,
            >>>                        add_bias = True, seed = 42)
        """
        if flavor is None:
            flavor = "default"
        elif flavor == "exact":
            max_deg = max(list(Cb.keys()))
            if D > max_deg:
                raise ValueError(f"For method 'exact', the maximum degree cannot exceed {max_deg}.")

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.D = D
        self.flavor = flavor
        self.residual = residual

        # Setup nnx rngs
        self.rngs = nnx.Rngs(seed)

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
        Based on the degree and flavor, the values of the Chebyshev basis functions are calculated on the input.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).

        Returns:
            cheb (jnp.array):
                Chebyshev basis functions applied on inputs, shape (batch, n_in, D+1).
            
        Example:
            >>> layer = ChebyshevLayer(n_in = 2, n_out = 5, D = 5, flavor = "default", 
            >>>                        residual = None, external_weights = False, init_scheme = None,
            >>>                        add_bias = True, seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = layer.basis(x_batch)
        """
        batch = x.shape[0]
        # Apply tanh activation
        x = jnp.tanh(x) # (batch, n_in)

        # Default implementation
        if self.flavor == "default":
            x = jnp.expand_dims(x, axis=-1) # (batch, n_in, 1)
            x = jnp.tile(x, (1, 1, self.D + 1)) # (batch, n_in, D+1)
            x = jnp.arccos(x) # (batch, n_in, D+1)
            x *= jnp.arange(self.D+1) # (batch, n_in, D+1)
            cheb = jnp.cos(x) # (batch, n_in, D+1)
            
        # Modified implementation
        elif self.flavor == "modified":
            # Order 0 is set by default, since we initialize at 1
            cheb = jnp.ones((batch, self.n_in, self.D+1))
            # Set order 1 as well
            cheb = cheb.at[:, :, 1].set(x)
            # Handle higher orders iteratively
            for K in range(2, self.D+1):
                cheb = cheb.at[:, :, K].set(2 * x * cheb[:, :, K - 1] - cheb[:, :, K - 2])

        # Exact calculation of polynomials
        elif self.flavor == "exact":
            cheb = jnp.stack([Cb[i](x) for i in range(self.D + 1)], axis=-1)  # (batch, n_in, D+1)

        # Other flavor
        else:
            raise ValueError(f"Unknown layer flavor: {self.flavor}")

        # Exclude the constant "1" dimension if bias is included
        if self.bias is not None:
            return cheb[:, :, 1:]
        else:
            return cheb


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

        # Find if we have D+1 external dimension (if add_bias = False) or D (if add_bias = True)
        ext_dim = self.D if self.bias is not None else self.D+1

        # Default initialization
        if init_type == "default":

            if self.residual is not None:
                c_res = nnx.initializers.glorot_uniform(in_axis=-1, out_axis=-2)(
                    self.rngs.params(), (self.n_out, self.n_in), jnp.float32
                )
            
            std = 1.0/jnp.sqrt(self.n_in * ext_dim)
            c_basis = nnx.initializers.truncated_normal(stddev=std)(
                self.rngs.params(), (self.n_out, self.n_in, ext_dim), jnp.float32
            )

        # Custom power law initialization
        # c_basis ~ N(0, s_b), s_b = const_b / [ (ext_dim+1)^pow_b1 * n_in^pow_b2 ]
        # c_res ~ N(0, s_r), s_r = const_r / [ (ext_dim+1)^pow_r1 * n_in^pow_r2 ]
        elif init_type == "power":

            const_b = init_scheme.get("const_b", 1.0)
            pow_b1 = init_scheme.get("pow_b1", 0.5)
            pow_b2 = init_scheme.get("pow_b2", 0.5)

            if self.residual is not None:
                basis_term = ext_dim + 1
                
                const_r = init_scheme.get("const_r", 1.0)
                pow_r1 = init_scheme.get("pow_r1", 0.5)
                pow_r2 = init_scheme.get("pow_r2", 0.5)
                
                std_res = const_r / ( (basis_term**pow_r1) * (self.n_in**pow_r2) )
                c_res = nnx.initializers.normal(stddev=std_res)(
                    self.rngs.params(), (self.n_out, self.n_in), jnp.float32
                )                
            else:
                basis_term = ext_dim

            std_b = const_b / ( (basis_term**pow_b1) * (self.n_in**pow_b2) )
            c_basis = nnx.initializers.normal(stddev=std_b)(
                self.rngs.params(), (self.n_out, self.n_in, ext_dim), jnp.float32
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
                scale = self.n_in * (ext_dim + 1)
                # Apply the residual function
                y_res = self.residual(sample)
                # Calculate the average of residual^2(x)
                y_res_sq = y_res**2
                y_res_sq_mean = y_res_sq.mean().item()

                std_res = gain/jnp.sqrt(scale*y_res_sq_mean)
                c_res = nnx.initializers.normal(stddev=std_res)(self.rngs.params(), (self.n_out, self.n_in), jnp.float32)
            
            else:
                # Variance equipartitioned across G+k terms
                scale = self.n_in * ext_dim

            std_b = gain/jnp.sqrt(scale*y_b_sq_mean)
            c_basis = nnx.initializers.normal(stddev=std_b)(
                self.rngs.params(), (self.n_out, self.n_in, ext_dim), jnp.float32
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
            grad_b_sq_mean = grad_sq_accum / (sample_size * ext_dim)
            # ------------------------------------------------------------
            
            # Calculate E[B_m^2(x)]
            y_b = self.basis(sample_ext)
            y_b_sq = y_b**2
            y_b_sq_mean = y_b_sq.mean().item()
            
            # Deal with residual if available
            if self.residual is not None:
                # Variance equipartitioned across all terms
                scale_in = self.n_in * (ext_dim + 1)
                scale_out = self.n_out * (ext_dim + 1)

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
                scale_in = self.n_in * ext_dim
                scale_out = self.n_out * ext_dim

            std_b = gain*jnp.sqrt(2.0 / (scale_in*y_b_sq_mean + scale_out*grad_b_sq_mean))
            c_basis = nnx.initializers.normal(stddev=std_b)(
                self.rngs.params(), (self.n_out, self.n_in, ext_dim), jnp.float32
            )

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
        For the case of ChebyKANs there is no concept of grid. However, a fine-graining approach can be followed by progressively increasing the degree of the polynomials.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).
            D_new (int):
                New Chebyshev polynomial degree.
            
        Example:
            >>> layer = ChebyshevLayer(n_in = 2, n_out = 5, D = 5, flavor = "default", 
            >>>                        residual = None, external_weights = False, init_scheme = None,
            >>>                        add_bias = True, seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> layer.update_grid(x=x_batch, D_new=8)
        """

        # Apply the inputs to the current grid to acquire y = Sum(ciBi(x)), where ci are
        # the current coefficients and Bi(x) are the current Chebyshev basis functions
        Bi = self.basis(x).transpose(1, 0, 2) # (n_in, batch, D+1)
        ci = self.c_basis.value.transpose(1, 2, 0) # (n_in, D+1, n_out)
        ciBi = jnp.einsum('ijk,ikm->ijm', Bi, ci) # (n_in, batch, n_out)

        # Update the degree order
        self.D = D_new

        # Get the Bj(x) for the degree order
        Bj = self.basis(x).transpose(1, 0, 2) # (n_in, batch, D_new+1)

        # Solve for the new coefficients
        cj = solve_full_lstsq(Bj, ciBi) # (n_in, D_new+1, n_out)
        # Cast into shape (n_out, n_in, D_new+1)
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
                Output of the forward pass, shape (batch, n_out).
            
        Example:
            >>> layer = ChebyshevLayer(n_in = 2, n_out = 5, D = 5, flavor = "default", 
            >>>                        residual = None, external_weights = False, init_scheme = None,
            >>>                        add_bias = True, seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = layer(x_batch)
        """
        
        batch = x.shape[0]
        
        # Calculate basis activations
        Bi = self.basis(x) # (batch, n_in, D+1)
        act = Bi.reshape(batch, -1) # (batch, n_in * (D+1))

        # Check if external_weights == True
        if self.c_ext is not None:
            act_w = self.c_basis.value * self.c_ext[..., None] # (n_out, n_in, D+1)
        else:
            act_w = self.c_basis.value
        
        # Calculate coefficients
        act_w = act_w.reshape(self.n_out, -1) # (n_out, n_in * (D+1))

        y = jnp.matmul(act, act_w.T) # (batch, n_out)

        # Check if there is a residual function
        if self.residual is not None:
            # Calculate residual activation
            res = self.residual(x) # (batch, n_in)
            # Multiply by trainable weights
            res_w = self.c_res.value # (n_out, n_in)
            full_res = jnp.matmul(res, res_w.T) # (batch, n_out)

            y += full_res # (batch, n_out)

        if self.bias is not None:
            y += self.bias.value # (batch, n_out)
        
        return y
