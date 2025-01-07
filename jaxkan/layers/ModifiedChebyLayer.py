from jax import numpy as jnp

from flax import nnx

from ..utils.general import solve_full_lstsq
        
        
class ModifiedChebyLayer(nnx.Module):
    """
    ModifiedChebyLayer class. Corresponds to the Modified Chebyshev version of KANs, in that the Chebyshev polynomials are calculated recursively and not via the arccos representation. Ref: https://www.sciencedirect.com/science/article/pii/S0045782524005462

    Attributes:
        n_in (int):
            Number of layer's incoming nodes.
        n_out (int):
            Number of layer's outgoing nodes.
        k (int):
            Degree of Chebyshev polynomial (1st kind).
        rngs (nnx.Rngs):
            Random key selection for initializations wherever necessary.
    """
    
    def __init__(self,
                 n_in: int = 2, n_out: int = 5, k: int = 5, rngs: nnx.Rngs = nnx.Rngs(42)
                ):
        """
        Initializes a ModifiedChebyLayer instance.
        
        Args:
            n_in (int):
                Number of layer's incoming nodes.
            n_out (int):
                Number of layer's outgoing nodes.
            k (int):
                Degree of Chebyshev polynomial (1st kind).
            rngs (nnx.Rngs):
                Random key selection for initializations wherever necessary.
            
        Example:
            >>> layer = ModifiedChebyLayer(n_in = 2, n_out = 5, k = 5, rngs = nnx.Rngs(42))
        """

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.k = k

        # Register and initialize the trainable parameters of the layer: c_basis, c_act

        # shape (n_out, n_in, k+1)
        noise_std = 1.0/(self.n_in * (self.k+1))
        self.c_basis = nnx.Param(
            nnx.initializers.normal(stddev=noise_std)(
                rngs.params(), (self.n_out, self.n_in, self.k + 1), jnp.float32)
        )

        # shape (n_out, n_in)
        self.c_act = nnx.Param(jnp.ones((self.n_out, self.n_in)))

    def basis(self, x):
        """
        Based on the degree, the values of the Chebyshev basis functions are calculated on the input.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).

        Returns:
            cheby (jnp.array):
                Chebyshev basis functions applied on inputs, shape (batch, n_in, k+1).
            
        Example:
            >>> layer = ModifiedChebyLayer(n_in = 2, n_out = 5, k = 5, rngs = nnx.Rngs(42))
            >>>
            >>> key = jax.random.PRNGKey(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
            >>>
            >>> output = layer.basis(x_batch)
        """
        
        batch = x.shape[0]
        
        # Apply tanh activation
        x = jnp.tanh(x) # (batch, n_in)
        
        # Order 0 is set by default, since we initialize at 1
        cheby = jnp.ones((batch, self.n_in, self.k+1))
        
        # Set order 1 as well
        cheby = cheby.at[:, :, 1].set(x)
        
        # Handle higher orders iteratively
        for K in range(2, self.k+1):
            cheby = cheby.at[:, :, K].set(2 * x * cheby[:, :, K - 1] - cheby[:, :, K - 2])

        return cheby


    def update_grid(self, x, k_new):
        """
        For the case of ChebyKANs there is no concept of grid. However, a fine-graining approach can be followed by progressively increasing the degree of the polynomials.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).
            k_new (int):
                New Chebyshev polynomial degree
            
        Example:
            >>> layer = ModifiedChebyLayer(n_in = 2, n_out = 5, k = 5, rngs = nnx.Rngs(42))
            >>>
            >>> key = jax.random.PRNGKey(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
            >>>
            >>> layer.update_grid(x=x_batch, k_new=7)
        """

        # Apply the inputs to the current grid to acquire y = Sum(ciTi(x)), where ci are
        # the current coefficients and Ti(x) are the current Chebyshev basis functions
        Ti = self.basis(x).transpose(1, 0, 2) # (n_in, batch, k+1)
        ci = self.c_basis.value.transpose(1, 2, 0) # (n_in, k+1, n_out)
        ciTi = jnp.einsum('ijk,ikm->ijm', Ti, ci) # (n_in, batch, n_out)

        # Update the degree order
        self.k = k_new

        # Get the Tj(x) for the degree order
        Tj = self.basis(x).transpose(1, 0, 2) # (n_in, batch, k_new+1)

        # Solve for the new coefficients
        cj = solve_full_lstsq(Tj, ciTi) # (n_in, k_new+1, n_out)
        # Cast into shape (n_out, n_in, k_new+1)
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
            >>> layer = ModifiedChebyLayer(n_in = 2, n_out = 5, k = 5, rngs = nnx.Rngs(42))
            >>>
            >>> key = jax.random.PRNGKey(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
            >>>
            >>> output = layer(x_batch)
        """
        
        batch = x.shape[0]
        
        # Calculate Chebyshev basis activations
        Ti = self.basis(x) # (batch, n_in, k+1)
        cheb = Ti.reshape(batch, -1) # (batch, n_in * (k+1))
        
        # Calculate coefficients
        cheb_w = self.c_basis.value * self.c_act[..., None] # (n_out, n_in, k+1)
        cheb_w = cheb_w.reshape(self.n_out, -1) # (n_out, n_in * (k+1))

        y = jnp.matmul(cheb, cheb_w.T) # (batch, n_out)
        
        return y