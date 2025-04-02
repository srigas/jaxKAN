from typing import List
from jax import numpy as jnp

from flax import nnx

from .layers import get_layer

from typing import Union


class KAN(nnx.Module):
    """
    KAN class, corresponding to a network of KAN Layers.

    Attributes:
        layer_dims (List[int]):
            Defines the network in terms of nodes. E.g. [4,5,1] is a network with 2 layers: one with n_in=4 and n_out=5 and one with n_in=5 and n_out = 1.
        layer_type (str):
            Type of layer to use (e.g., 'base').
        required_parameters (dict):
            Dictionary containing parameters required for the chosen layer type.
        seed (int):
            Random key selection for initializations wherever necessary.
    """
    
    def __init__(self, layer_dims: List[int], layer_type: str = "base",
                 required_parameters: Union[None, dict] = None, seed: int = 42
                ):
        """
        Initializes a KAN model.

        Args:
            layer_dims (List[int]):
                Defines the network in terms of nodes. E.g. [4,5,1] is a network with 2 layers: one with n_in=4 and n_out=5 and one with n_in=5 and n_out = 1.
            layer_type (str):
                Type of layer to use (e.g., 'base').
            required_parameters (dict):
                Dictionary containing parameters required for the chosen layer type.
            add_bias (bool):
                Boolean that controls wether bias terms are also included during the forward pass or not.
            seed (int):
                Random key selection for initializations wherever necessary.
                
        Example:
            >>> req_params = {'k': 3, 'G': 5}
            >>> model = KAN(layer_dims = [2,5,1], layer_type='base', required_parameters=req_params, seed=42)
        """
        self.layer_type = layer_type.lower()
        
        # Get the corresponding layer class based on layer_type
        LayerClass = get_layer(self.layer_type)
            
        if required_parameters is None:
            raise ValueError("required_parameters must be provided as a dictionary for the selected layer_type.")
        
        self.layers = [
                LayerClass(
                    n_in=layer_dims[i],
                    n_out=layer_dims[i + 1],
                    **required_parameters,
                    seed=seed
                )
                for i in range(len(layer_dims) - 1)
            ]
    
    def update_grids(self, x, G_new):
        """
        Performs the grid update for each layer of the KAN architecture.

        Args:
            x (jnp.array):
                Inputs for the first layer, shape (batch, self.layers[0]).
            G_new (int):
                Size of the new grid (in terms of intervals).
            
        Example:
            >>> req_params = {'k': 3, 'G': 5}
            >>> model = KAN(layer_dims = [2,5,1], layer_type='base', required_parameters=req_params, seed=42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> model.update_grids(x=x_batch, G_new=10)
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
            x (jnp.array):
                Inputs for the first layer, shape (batch, self.layers[0]).

        Returns:
            x (jnp.array):
                Network output, shape (batch, self.layers[-1]).
            
        Example:
            >>> req_params = {'k': 3, 'G': 5}
            >>> model = KAN(layer_dims = [2,5,1], layer_type='base', required_parameters=req_params, seed=42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = model(x_batch)
        """

        # Pass through each layer of the KAN
        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x
