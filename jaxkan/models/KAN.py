from typing import List
from jax import numpy as jnp

from flax import nnx

from .KANLayer import KANLayer


class KAN(nnx.Module):
    """
    KAN class, corresponding to a network of KANLayers.

    Args:
    -----------
        layer_dims (List[int]): defines the network in terms of nodes. E.g. [4,5,1] is a network with 2 layers: one with n_in=4 and n_out=5 and one with n_in=5 and n_out = 1.
        add_bias (bool): boolean that controls wether bias terms are also included during the forward pass or not. Default: True
        k (int): input for KANLayer class - see KANLayer.py
        const_spl (float/bool): input for KANLayer class - see KANLayer.py
        const_res (float/bool): input for KANLayer class - see KANLayer.py
        residual (nn.Module): input for KANLayer class - see KANLayer.py
        noise_std (float): input for KANLayer class - see KANLayer.py
        grid_e (float): input for KANLayer class - see KANLayer.py
    """
    def __init__(self,
                 layer_dims: List[int], add_bias: bool = True, k: int = 3,
                 G: int = 3, grid_range: tuple = (-1,1), grid_e: float = 0.05,
                 residual: nnx.Module = nnx.silu, noise_std: float = 0.15,
                 rngs: nnx.Rngs = nnx.Rngs(42)
                ):

        # Initialize KAN layers based on layer_dims
        self.layers = [
                KANLayer(
                    n_in=layer_dims[i],
                    n_out=layer_dims[i + 1],
                    k=k,
                    G=G,
                    grid_range=grid_range,
                    grid_e=grid_e,
                    residual=residual,
                    noise_std=noise_std,
                    rngs=rngs
                )
                for i in range(len(layer_dims) - 1)
            ]
    
        if add_bias:
            self.biases = [
                nnx.Param(jnp.zeros((dim,))) for dim in layer_dims[1:]
            ]
        else:
            self.biases = [
                jnp.zeros((dim,)) for dim in layer_dims[1:]
            ]
    
    def update_grids(self, x, G_new):
        """
        Performs the grid update for each layer of the KAN architecture.

        Args:
        -----
            x (jnp.array): inputs for the first layer
                shape (batch, self.layers[0])
            G_new (int): Size of the new grid (in terms of intervals)

        """

        # Loop over each layer
        for i, layer in enumerate(self.layers):
            
            # Update the grid for the current layer
            layer.grid.update(x, G_new)

            # Perform a forward pass to get the input for the next layer
            x, _ = layer(x)
            x += self.biases[i].value

    
    def __call__(self, x):
        """
        Equivalent to the network's forward pass.

        Args:
        -----
            x (jnp.array): inputs for the first layer
                shape (batch, self.layers[0])

        Returns:
        --------
            x (jnp.array): network output
                shape (batch, self.layers[-1])
            spl_regs (List[jnp.array]): a list of spl_reg arrays to be used for the loss function's regularization term.
                size len(layer_dims)-1
        """
        spl_regs = []

        # Pass through each layer of the KAN
        for i, layer in enumerate(self.layers):
            x, spl_reg = layer(x)
            x += self.biases[i].value

            # Append the regularization terms per layer in a list
            spl_regs.append(spl_reg)

        # Return the total output and the regularization terms
        return x, spl_regs
