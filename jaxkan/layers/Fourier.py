import jax
import jax.numpy as jnp

from flax import nnx

from typing import Union

from .utils import solve_full_lstsq
        
        
class FourierLayer(nnx.Module):
    """
    FourierLayer class. Corresponds to the Fourier-based version of KANs (FourierKAN). Ref: https://github.com/GistNoesis/FourierKAN

    Attributes:
        n_in (int):
            Number of layer's incoming nodes.
        n_out (int):
            Number of layer's outgoing nodes.
        D (int):
            Order of Fourier sum.
        smooth_init (bool):
            Whether to initialize Fourier coefficients with smoothening.
        add_bias (bool):
            Boolean that controls wether bias terms are also included during the forward pass or not.
        seed (int):
            Random key selection for initializations wherever necessary.
    """
    
    def __init__(self, n_in: int = 2, n_out: int = 5, D: int = 5,
                 smooth_init: bool = True, add_bias: bool = True, seed: int = 42):
        """
        Initializes a FourierLayer instance.
        
        Args:
            n_in (int):
                Number of layer's incoming nodes.
            n_out (int):
                Number of layer's outgoing nodes.
            D (int):
                Order of Fourier sum.
            smooth_init (bool):
                Whether to initialize Fourier coefficients with smoothening.
            add_bias (bool):
                Boolean that controls wether bias terms are also included during the forward pass or not.
            seed (int):
                Random key selection for initializations wherever necessary.
            
        Example:
            >>> layer = FourierLayer(n_in = 2, n_out = 5, D = 5, smooth_init = True, add_bias = True, seed = 42)
        """

        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.D = D

        # Setup nnx rngs
        rngs = nnx.Rngs(seed)

        # Add bias
        if add_bias == True:
            self.bias = nnx.Param(jnp.zeros((n_out,)))
        else:
            self.bias = None
        
        # Fourier coefficient normalization
        norm_factor = jnp.arange(1, self.D + 1) ** 2 if smooth_init else jnp.sqrt(self.D)

        # Register and initialize the trainable parameters of the layer: c_sin, c_cos
        
        # Initialize with σ = 1/sqrt(n_in)
        inits = nnx.initializers.normal(stddev=1.0/jnp.sqrt(self.n_in))(
                        rngs.params(), (2, self.n_out, self.n_in, self.D), jnp.float32)
                        
        # Divide by norm_factor, which is either sqrt(k), or the k-dependent smoothening array
        inits /= norm_factor
        
        # Split to sine and cosine terms
        self.c_cos = nnx.Param(inits[0,:,:,:]) # shape (n_out, n_in, D)
        self.c_sin = nnx.Param(inits[1,:,:,:]) # shape (n_out, n_in, D)

    
    def basis(self, x):
        """
        Calculates the con/sin activations on the input x.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).

        Returns:
            c, s (tuple):
                Cosines, sines applied on inputs, shape (batch, n_in, D).
            
        Example:
            >>> layer = FourierLayer(n_in = 2, n_out = 5, D = 5, smooth_init = True, add_bias = True, seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output_1, output_2 = layer.basis(x_batch)
        """
        
        # Expand x to an extra dim for broadcasting
        x = jnp.expand_dims(x, axis=-1) # (batch, n_in, 1)
    
        # Broadcast [1, 2, ..., D] for multiplication
        D_array = jnp.arange(1, self.D + 1).reshape(1, 1, self.D)
        
        # cos/sin terms
        Dx = D_array * x # (batch, n_in, D)
        c, s = jnp.cos(Dx), jnp.sin(Dx) # (batch, n_in, D)

        return c, s


    def update_grid(self, x, D_new):
        """
        For the case of FourierKAN there is no concept of grid. However, a fine-graining approach can be followed by progressively increasing the number of summands.

        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).
            D_new (int):
                New value for the fourier sum's order.
            
        Example:
            >>> layer = FourierLayer(n_in = 2, n_out = 5, D = 5, smooth_init = True, add_bias = True, seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> layer.update_grid(x=x_batch, D_new=8)
        """

        # Apply the inputs to the current "grid" to acquire the cosine and sine terms
        ci, si = self.basis(x) # (batch, n_in, k)
        ci, si = ci.transpose(1, 0, 2), si.transpose(1, 0, 2) # (n_in, batch, D)
        cos_w = self.c_cos[...].transpose(1, 2, 0) # (n_in, D, n_out)
        sin_w = self.c_sin[...].transpose(1, 2, 0) # (n_in, D, n_out)
        
        cosines = jnp.einsum('ijk,ikm->ijm', ci, cos_w) # (n_in, batch, n_out)
        sines = jnp.einsum('ijk,ikm->ijm', si, sin_w) # (n_in, batch, n_out)

        # Update the degree order
        self.D = D_new

        # Get the new fourier activations
        cj, sj = self.basis(x) # (batch, n_in, D_new)
        cj, sj = cj.transpose(1, 0, 2), sj.transpose(1, 0, 2) # (n_in, batch, D_new)

        # Solve for the new cosine coefficients
        new_cos_w = solve_full_lstsq(cj, cosines) # (n_in, D_new, n_out)
        
        # Solve for the new sine coefficients
        new_sin_w = solve_full_lstsq(sj, sines) # (n_in, D_new, n_out)
        
        # Cast into shape (n_out, n_in, D_new)
        new_cos_w = new_cos_w.transpose(2, 0, 1)
        new_sin_w = new_sin_w.transpose(2, 0, 1)

        self.c_cos = nnx.Param(new_cos_w)
        self.c_sin = nnx.Param(new_sin_w)


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
            >>> layer = FourierLayer(n_in = 2, n_out = 5, D = 5, smooth_init = True, add_bias = True, seed = 42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = layer(x_batch)
        """
        
        batch = x.shape[0]
        
        # Calculate Fourier basis activations
        ci, si = self.basis(x) # each has shape (batch, n_in, D)
        cosines, sines = ci.reshape(batch, -1), si.reshape(batch, -1) # each has shape (batch, n_in * D)
        
        # Reshape factors
        cos_w = self.c_cos[...].reshape(self.n_out, -1) # (n_out, n_in * D)
        sin_w = self.c_sin[...].reshape(self.n_out, -1) # (n_out, n_in * D)
        
        # Get inner products        
        y = jnp.matmul(cosines, cos_w.T) # (batch, n_out)
        y += jnp.matmul(sines, sin_w.T) # (batch, n_out)

        if self.bias is not None:
            y += self.bias[...] # (batch, n_out)
        
        return y
        