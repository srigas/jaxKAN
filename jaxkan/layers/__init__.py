from .BaseLayer import BaseLayer
#from .EfficientLayer import EfficientLayer

def get_layer(layer_type: str):
    layer_map = {
        "base": BaseLayer,
        #"efficient": EfficientLayer
    }
    
    if layer_type not in layer_map:
        raise ValueError(f"Unknown layer type: {layer_type}. Available types: {list(layer_map.keys())}")
        
    return layer_map[layer_type]