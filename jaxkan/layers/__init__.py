from .Spline import BaseLayer, SplineLayer
from .Chebyshev import ChebyLayer, ModifiedChebyLayer
from .Fourier import FourierLayer


def get_layer(layer_type: str):
    """
    Helper method that creates a mapping between layer type codes and the actual classes.

    Args:
        layer_type (str):
            Code of layer to be used.
            
    Returns:
        layer (jaxkan.layers.Layer):
            A jaxkan.layers layer class instance to be used as the building block of a KAN.
            
    Example:
        >>> LayerClass = get_layer("base")
    """
    layer_map = {
        "base": BaseLayer,
        "spline": SplineLayer,
        "cheby": ChebyLayer,
        "mod-cheby": ModifiedChebyLayer,
        "fourier": FourierLayer
    }
    
    if layer_type not in layer_map:
        raise ValueError(f"Unknown layer type: {layer_type}. Available types: {list(layer_map.keys())}")
        
    LayerClass = layer_map[layer_type]
        
    return LayerClass