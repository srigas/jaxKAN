from .BaseLayer import BaseLayer
from .SplineLayer import SplineLayer

def get_layer(layer_type: str):
    layer_map = {
        "base": BaseLayer,
        "spline": SplineLayer
    }
    
    if layer_type not in layer_map:
        raise ValueError(f"Unknown layer type: {layer_type}. Available types: {list(layer_map.keys())}")
        
    return layer_map[layer_type]