import torch
from torch import nn

from layers import CausalLSTMCell, GHU


class CausalLSTMStack(nn.Module):
    def __init__(self,
                 num_batch,
                 num_height,
                 num_width,
                 nums_hidden,
                 filter_size,
                 layer_norm=True,
                 ):
        super(CausalLSTMStack, self).__init__()

        self.num_batch = num_batch
        self.num_height = num_height
        self.num_width = num_width

        self.filter_size = filter_size
        self.num_layers = len(nums_hidden) - 1
        self.nums_hidden = nums_hidden

        assert self.num_layers >= 2

        self.lstms = []
        for i, n_hidden in enumerate(nums_hidden):
            n_hid_in, n_hid_out = nums_hidden[i:i+1]

            cell = CausalLSTMCell(filter_size,
                                  n_hid_in,
                                  n_hid_out,
                                  self.num_batch,
                                  self.num_height,
                                  self.num_width,
                                  layer_norm=layer_norm)

            self.lstms.append(cell)

        self.ghu = GHU(filter_size, nums_hidden[0],
                       tln=layer_norm)

    def forward(self, x, h_prev=None, c_prev=None, m_prev=None, z_prev=None):
        if h_prev is None:
            h_prev = [None] * self.num_layers
        if c_prev is None:
            c_prev = [None] * self.num_layers

        h_new = [None] * self.num_layers
        c_new = [None] * self.num_layers

        h, c, m = self.lstms[0](x, h_prev[0], c_prev[0], m_prev)
        h_new[0], c_new[0] = h, c

        z = self.ghu(h, z_prev)

        h, c, m = self.lstms[1](z, h_prev[1], c_prev[1], m)
        h_new[1], c_new[1] = h, c

        for k in range(2, self.num_layers):
            h, c, m = self.lstms[k](h_new[k - 1], h_prev[k], c_prev[k], m)
            h_new[k], c_new[k] = h, c

        return h_new, c_new, m, z
