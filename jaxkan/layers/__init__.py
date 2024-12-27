from .BaseLayer import BaseLayer
from .SplineLayer import SplineLayer
from .ChebyLayer import ChebyLayer
from .ModifiedChebyLayer import ModifiedChebyLayer
from .FourierLayer import FourierLayer

def get_layer(layer_type: str):
    layer_map = {
        "base": BaseLayer,
        "spline": SplineLayer,
        "cheby": ChebyLayer,
        "mod-cheby": ModifiedChebyLayer,
        "fourier": FourierLayer
    }
    
    if layer_type not in layer_map:
        raise ValueError(f"Unknown layer type: {layer_type}. Available types: {list(layer_map.keys())}")
        
    return layer_map[layer_type]