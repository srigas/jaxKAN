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
                Chebyshev basis functions applied on inputs, shape (batch, n_in, D).
            
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

        # LeCun-like initialization, where Var[in] = Var[out]
        elif init_type == "lecun":

            # Generate a sample of 10^5 points
            key = jax.random.key(seed)
            sample = jax.random.uniform(key, shape=(100000,), minval=-1.0, maxval=1.0)
            sample_std = sample.std().item()
            
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

                std_res = sample_std/jnp.sqrt(scale*y_res_sq_mean)
                c_res = nnx.initializers.normal(stddev=std_res)(self.rngs.params(), (self.n_out, self.n_in), jnp.float32)
            
            else:
                # Variance equipartitioned across G+k terms
                scale = self.n_in * ext_dim

            std_b = sample_std/jnp.sqrt(scale*y_b_sq_mean)
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

        # Apply the inputs to the current grid to acquire y = Sum(ciTi(x)), where ci are
        # the current coefficients and Ti(x) are the current Chebyshev basis functions
        Ti = self.basis(x).transpose(1, 0, 2) # (n_in, batch, D+1)
        ci = self.c_basis.value.transpose(1, 2, 0) # (n_in, D+1, n_out)
        ciTi = jnp.einsum('ijk,ikm->ijm', Ti, ci) # (n_in, batch, n_out)

        # Update the degree order
        self.D = D_new

        # Get the Tj(x) for the degree order
        Tj = self.basis(x).transpose(1, 0, 2) # (n_in, batch, D_new+1)

        # Solve for the new coefficients
        cj = solve_full_lstsq(Tj, ciTi) # (n_in, D_new+1, n_out)
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
