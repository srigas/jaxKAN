from KANLayer import KANLayer

from flax import linen as nn

class KAN(nn.Module):
    layers: list
    
    def setup(self):
        self.kan_layers = [KANLayer(n_in=2, n_out=5) for _ in self.layers]
    
    def __call__(self, x):
        for layer in self.kan_layers:
            x = layer(x)
        return x