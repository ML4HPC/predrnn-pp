#__author__ = 'yunbo'

# import tensorflow as tf
# from layers.TensorLayerNorm import tensor_layer_norm
import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm


#  LayerNorm
## def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):

##class GHU():
class GHU(nn.Module):
    def __init__(self, filter_size, num_features, tln=False, initializer=0.001):
        """Initialize the Gradient Highway Unit.
        """
        super(GHU, self).__init__()
        self.filter_size = filter_size
        self.num_features = num_features
        self.layer_norm = tln
        self.conv2d_z = nn.Conv2d(num_features, num_features*2, filter_size, 1,
                                  filter_size//2, padding_mode='replicate')
        self.conv2d_x = nn.Conv2d(num_features, num_features*2, filter_size, 1,
                                  filter_size//2, padding_mode='replicate')
        self.layernorm = nn.LayerNorm(num_features*2)
        if initializer == -1:
            self.initializer = None
        else:
            nn.init.uniform_(self.conv2d_z.weights, -initializer, initializer)
            nn.init.uniform_(self.conv2d_x.weights, -initializer, initializer)

    def init_state(self, inputs, num_features):
        dims = inputs.shape
        if dims == 4:
            batch = inputs.shape[0]
            height = inputs.shape[1]
            width = inputs.shape[2]
        else:
            raise ValueError('input tensor should be rank 4.')
        return torch.zeros([batch, height, width, num_features],
                           dtype=inputs.dtype, device=inputs.device)

    def forward(self, x, z):
        if z is None:
            z = self.init_state(x, self.num_features)
        z_concat = self.conv2d_z(z)
        x_concat = self.conv2d_x(x)
        if self.layer_norm:
            x_concat = self.layernorm(x_concat.permute(
                0, 2, 3, 1)).permute(0, -1, 1, 2)
            z_concat = self.layernorm(z_concat.permute(
                0, 2, 3, 1)).permute(0, -1, 1, 2)
        gates = x_concat + z_concat
        p, u = torch.split(gates, 2, 3)  # into 2 parts at axis=3
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1-u) * z
        return z_new
