from typing import List
from jax import numpy as jnp

from flax import nnx

from .layers import get_layer


class KAN(nnx.Module):
    """
        KAN class, corresponding to a network of KAN Layers.

        Args:
        -----
            layer_dims (List[int]): defines the network in terms of nodes. E.g. [4,5,1] is a network with 2 layers: one with n_in=4 and n_out=5 and one with n_in=5 and n_out = 1.
            layer_type (str): type of layer to use (e.g., 'base').
            required_parameters (dict): dictionary containing parameters required for the chosen layer type.
            add_bias (bool): boolean that controls wether bias terms are also included during the forward pass or not.
            rngs (nnx.Rngs): random key selection for initializations wherever necessary.
            
        Example Usage:
        --------------
            req_params = {'k': 3, 'G': 3, 'grid_range': (-1,1), 'grid_e': 0.05, 'residual': nnx.silu, 'noise_std': 0.1}
            model = KAN(layer_dims = [2,5,1], layer_type='base', required_parameters=req_params,
                        add_bias = True, rngs = nnx.Rngs(42))
    """
    
    def __init__(self,
                 layer_dims: List[int], layer_type: str = "base", required_parameters: dict = {}, 
                 add_bias: bool = True, rngs: nnx.Rngs = nnx.Rngs(42)
                ):
                
        self.add_bias = add_bias
        
        # Get the corresponding layer class based on layer_type
        LayerClass = get_layer(layer_type.lower())
            
        if required_parameters is None:
            raise ValueError("required_parameters must be provided as a dictionary for the selected layer_type.")
        
        self.layers = [
                LayerClass(
                    n_in=layer_dims[i],
                    n_out=layer_dims[i + 1],
                    **required_parameters,
                    rngs=rngs
                )
                for i in range(len(layer_dims) - 1)
            ]
    
        if self.add_bias:
            self.biases = [
                nnx.Param(jnp.zeros((dim,))) for dim in layer_dims[1:]
            ]
    
    def update_grids(self, x, G_new):
        """
            Performs the grid update for each layer of the KAN architecture.

            Args:
            -----
                x (jnp.array): inputs for the first layer
                    shape (batch, self.layers[0])
                G_new (int): Size of the new grid (in terms of intervals)
                
            Example Usage:
            --------------
                req_params = {'k': 3, 'G': 3, 'grid_range': (-1,1), 'grid_e': 0.05, 'residual': nnx.silu, 'noise_std': 0.1}
                model = KAN(layer_dims = [2,5,1], layer_type='base', required_parameters=req_params,
                            add_bias = True, rngs = nnx.Rngs(42))
                              
                key = jax.random.PRNGKey(42)
                x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
                
                model.update_grids(x=x_batch, G_new=5)
        """

        # Loop over each layer
        for i, layer in enumerate(self.layers):
            
            # Update the grid for the current layer
            layer.update_grid(x, G_new)

            # Perform a forward pass to get the input for the next layer
            x = layer(x)
            
            if self.add_bias:
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
                
            Example Usage:
            --------------
                req_params = {'k': 3, 'G': 3, 'grid_range': (-1,1), 'grid_e': 0.05, 'residual': nnx.silu, 'noise_std': 0.1}
                model = KAN(layer_dims = [2,5,1], layer_type='base', required_parameters=req_params,
                            add_bias = True, rngs = nnx.Rngs(42))
                              
                key = jax.random.PRNGKey(42)
                x_batch = jax.random.uniform(key, shape=(100, 2), minval=-4.0, maxval=4.0)
                
                output = model(x_batch)
        """

        # Pass through each layer of the KAN
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.add_bias:
                x += self.biases[i].value

        return x
