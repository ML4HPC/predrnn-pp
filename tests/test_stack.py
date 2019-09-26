import sys
import torch
from pathlib import Path
import pytest

sys.path.append(str(Path.cwd()))

from layers import CausalLSTMStack  # noqa


def test_stack_2d():
    batch = 2
    dims = [5, 6]

    filter_size = 3
    num_dims = 2
    channels = [8, 6, 6, 6]

    node = CausalLSTMStack(filter_size, num_dims, channels)

    x = torch.rand([batch, channels[-1], *dims])

    h, c, m, z = node(x)
    h, c, m, z = node(x, h, c, m, z)


def test_stack_3d():
    batch = 3
    dims = [5, 6, 7]

    filter_size = 3
    num_dims = 3
    channels = [8, 6, 6, 6]

    node = CausalLSTMStack(filter_size, num_dims, channels)

    x = torch.rand([batch, channels[-1], *dims])

    h, c, m, z = node(x)
    h, c, m, z = node(x, h, c, m, z)


if __name__ == '__main__':
    pytest.main([__file__])
