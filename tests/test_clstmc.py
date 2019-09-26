import sys
import torch
from pathlib import Path
import pytest

sys.path.append(str(Path.cwd()))

from layers import CausalLSTMCell2d, CausalLSTMCell3d  # noqa


def test_clstmc_2d():
    batch = 2
    dims = [5, 6]

    filter_size = 3
    in_channels, out_channels = 3, 4

    node = CausalLSTMCell2d(filter_size, in_channels, out_channels)

    x = torch.rand([batch, in_channels, *dims])
    h, c, m = node(x)

    m = torch.rand([batch, in_channels, *dims])
    h, c, m = node(x, h, c, m)


def test_clstmc_3d():
    batch = 2
    dims = [5, 6, 7]

    filter_size = 3
    in_channels, out_channels = 3, 4

    node = CausalLSTMCell3d(filter_size, in_channels, out_channels)

    x = torch.rand([batch, in_channels, *dims])
    h, c, m = node(x)

    m = torch.rand([batch, in_channels, *dims])
    h, c, m = node(x, h, c, m)


if __name__ == '__main__':
    pytest.main([__file__])
