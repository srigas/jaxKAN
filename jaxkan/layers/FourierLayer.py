from jax import numpy as jnp

from flax import nnx

from ..utils.general import solve_full_lstsq
        
        
class FourierLayer(nnx.Module):
    """
        FourierLayer class. Corresponds to the Fourier-based version of KANs (FourierKAN).
        Ref: https://github.com/GistNoesis/FourierKAN

        Args:
        -----
            n_in (int): number of layer's incoming nodes.
            n_out (int): number of layer's outgoing nodes.
            k (int): degree of Chebyshev polynomial (1st kind).
            smooth_init (bool): whether to initialize Fourier coefficients with smoothening.
            rngs (nnx.Rngs): random key selection for initializations wherever necessary.
            
        Example Usage:
        --------------
            layer = FourierLayer(n_in = 2, n_out = 5, k = 5, smooth_init = True, rngs = nnx.Rngs(42))
    """
    
    def __init__(self,
                 n_in: int = 2, n_out: int = 5, k: int = 5, smooth_init: bool = True, rngs: nnx.Rngs = nnx.Rngs(42)
                ):

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        
        # Fourier coefficient normalization
        norm_factor = jnp.arange(1, self.k + 1) ** 2 if smooth_init else jnp.sqrt(self.k)

        # Register and initialize the trainable parameters of the layer: c_sin, c_cos
        
        
        # Initialize with σ = 1/sqrt(n_in)
        inits = nnx.initializers.normal(stddev=1.0/jnp.sqrt(self.n_in))(
                        rngs.params(), (2, self.n_out, self.n_in, self.k), jnp.float32)
                        
        # Divide by norm_factor, which is either sqrt(k), or the k-dependent smoothening array
        inits /= norm_factor
        
        # Split to sine and cosine terms
        self.c_cos = nnx.Param(inits[0,:,:,:]) # shape (n_out, n_in, k)
        self.c_sin = nnx.Param(inits[1,:,:,:]) # shape (n_out, n_in, k)

    def basis(self, x):
        """
            Calculate the con/sin activations on the input x.

            Args:
            -----
                x (jnp.array): inputs
                    shape (batch, n_in)

            Returns:
            --------
                c (jnp.array): Cosines applied on inputs
                    shape (batch, n_in, k)
                s (jnp.array): Sines applied on inputs
                    shape (batch, n_in, k)
                
            Example Usage:
            --------------
                layer = FourierLayer(n_in = 2, n_out = 5, k = 5, smooth_init = True, rngs = nnx.Rngs(42))
                              
                key = jax.random.PRNGKey(42)
                x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
                
                output_1, output_2 = layer.basis(x_batch)
        """
        # Expand x to an extra dim for broadcasting
        x = jnp.expand_dims(x, axis=-1) # (batch, n_in, 1)
    
        # Broadcast [1, 2, ..., k] for multiplication
        k_array = jnp.arange(1, self.k + 1).reshape(1, 1, self.k)
        
        # cos/sin terms
        kx = k_array * x # (batch, n_in, k)
        c, s = jnp.cos(kx), jnp.sin(kx) # (batch, n_in, k)

        return c, s


    def update_grid(self, x, k_new):
        """
            For the case of FourierKAN there is no concept of grid. However, a fine-graining approach
            can be followed by progressively increasing the number of summands.

            Args:
            -----
                x (jnp.array): inputs
                    shape (batch, n_in)
                k_new (int): new value for the fourier sum's order
                
            Example Usage:
            --------------
                layer = FourierLayer(n_in = 2, n_out = 5, k = 5, smooth_init = True, rngs = nnx.Rngs(42))
                              
                key = jax.random.PRNGKey(42)
                x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
                
                layer.update_grid(x=x_batch, k_new=7)
        """

        # Apply the inputs to the current "grid" to acquire the cosine and sine terms
        ci, si = self.basis(x) # (batch, n_in, k)
        ci, si = ci.transpose(1, 0, 2), si.transpose(1, 0, 2) # (n_in, batch, k)
        cos_w = self.c_cos.value.transpose(1, 2, 0) # (n_in, k, n_out)
        sin_w = self.c_sin.value.transpose(1, 2, 0) # (n_in, k, n_out)
        
        cosines = jnp.einsum('ijk,ikm->ijm', ci, cos_w) # (n_in, batch, n_out)
        sines = jnp.einsum('ijk,ikm->ijm', si, sin_w) # (n_in, batch, n_out)

        # Update the degree order
        self.k = k_new

        # Get the new fourier activations
        cj, sj = self.basis(x) # (batch, n_in, k_new)
        cj, sj = cj.transpose(1, 0, 2), sj.transpose(1, 0, 2) # (n_in, batch, k_new)

        # Solve for the new cosine coefficients
        new_cos_w = solve_full_lstsq(cj, cosines) # (n_in, k_new, n_out)
        
        # Solve for the new sine coefficients
        new_sin_w = solve_full_lstsq(sj, sines) # (n_in, k_new, n_out)
        
        # Cast into shape (n_out, n_in, k_new)
        new_cos_w = new_cos_w.transpose(2, 0, 1)
        new_sin_w = new_sin_w.transpose(2, 0, 1)

        self.c_cos = nnx.Param(new_cos_w)
        self.c_sin = nnx.Param(new_sin_w)


    def __call__(self, x):
        """
            The layer's forward pass.

            Args:
            -----
                x (jnp.array): inputs
                    shape (batch, n_in)

            Returns:
            --------
                y (jnp.array): output of the forward pass
                    shape (batch, n_out)
                
            Example Usage:
            --------------
                layer = FourierLayer(n_in = 2, n_out = 5, k = 5, smooth_init = True, rngs = nnx.Rngs(42))
                              
                key = jax.random.PRNGKey(42)
                x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
                
                output = layer(x_batch)
        """
        
        batch = x.shape[0]
        
        # Calculate Fourier basis activations
        ci, si = self.basis(x) # each has shape (batch, n_in, k)
        cosines, sines = ci.reshape(batch, -1), si.reshape(batch, -1) # each has shape (batch, n_in * k)
        
        # Reshape factors
        cos_w = self.c_cos.value.reshape(self.n_out, -1) # (n_out, n_in * k)
        sin_w = self.c_sin.value.reshape(self.n_out, -1) # (n_out, n_in * k)
        
        # Get inner products        
        y = jnp.matmul(cosines, cos_w.T) # (batch, n_out)
        y += jnp.matmul(sines, sin_w.T) # (batch, n_out)
        
        return y