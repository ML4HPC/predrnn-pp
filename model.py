import torch
from torch import nn
import pytest

from layers import CausalLSTMStack  # noqa


class Model(nn.Module):
    def __init__(self, filter_size, num_dims, channels):
        super(Model, self).__init__()

        self.csltm = CausalLSTMStack(filter_size, num_dims, channels)


def test_model():
    batch = 3
    seq_length = 20
    dims = [5, 6, 7]

    filter_size = 3
    num_dims = 3
    channels = [8, 6, 6, 1]

    node = CausalLSTMStack(filter_size, num_dims, channels)

    data = torch.rand([batch, seq_length, *dims])

    h, c, m, z = None, None, None, None
    for t in range(seq_length):
        h, c, m, z = node(data[:, [t]], h, c, m, z)

    conv1 = nn.Conv3d(1, 5, 3, 1)
    conv2 = nn.Conv3d(5, 1, 3, 1)

    cc = conv1(h[-1])
    cc = conv2(cc)

    cc_flat = torch.flatten(cc, 1, 4)

    lin = nn.Linear(6, 1)

    y = lin(cc_flat)

    print(y.shape)


if __name__ == '__main__':
    # pytest.main([__file__])
    test_model()
