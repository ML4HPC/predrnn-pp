import sys
import torch
from pathlib import Path
import pytest

sys.path.append(str(Path.cwd()))

from layers import GHU2d, GHU3d  # noqa


def test_ghu_2d():
    b, c, h, w = 2, 3, 5, 6
    filter_size = 3

    m = GHU2d(filter_size, c)
    x = torch.rand([b, c, h, w])

    z = m(x)
    z = m(x, z)


def test_ghu_3d():
    b, c, d1, d2, d3 = 2, 3, 5, 6, 7
    filter_size = 3

    m = GHU3d(filter_size, c)
    x = torch.rand([b, c, d1, d2, d3])

    z = m(x)
    z = m(x, z)


if __name__ == '__main__':
    # test_ghu_2d()
    pytest.main([__file__])
