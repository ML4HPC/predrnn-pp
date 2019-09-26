import torch
from torch import nn


class CausalLSTMCell(nn.Module):
    def __init__(self, filter_size, num_hidden_in, num_hidden_out,
                 num_batch, num_height, num_width,
                 forget_bias=1.0, layer_norm=False):
        """
        Initialize the Causal LSTM cell.

        Parameters
        ==========
        filter_size:
            int tuple thats the height and width of the filter.
        num_hidden_in:
            number of units for input tensor.
        num_hidden_out:
            number of units for output tensor.
        seq_shape:
            shape of a sequence.
        forget_bias: float
            The bias added to forget gates.
        layer_norm:
            whether to apply tensor layer normalization
        """
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.num_hidden_out = num_hidden_out

        self.batch = num_batch
        self.height = num_height
        self.width = num_width

        self.layer_norm = layer_norm
        self._forget_bias = forget_bias

        self.conv_h = \
            nn.Conv2d(num_hidden_out, num_hidden_out * 4, filter_size,
                      stride=1, padding=1, padding_mod='replicate')
        self.conv_c = \
            nn.Conv2d(num_hidden_out, num_hidden_out * 3, filter_size,
                      stride=1, padding=1, padding_mod='replicate')
        self.conv_m = \
            nn.Conv2d(num_hidden_in, num_hidden_out * 3, filter_size,
                      stride=1, padding=1, padding_mod='replicate')
        self.conv_x = \
            nn.Conv2d(num_hidden_in, num_hidden_out * 7, filter_size,
                      stride=1, padding=1, padding_mod='replicate')

        self.conv_c2 = \
            nn.Conv2d(num_hidden_out, num_hidden_out * 4, filter_size,
                      stride=1, padding=1, padding_mod='replicate')
        self.conv_o = \
            nn.Conv2d(num_hidden_out, num_hidden_out, filter_size,
                      stride=1, padding=1, padding_mod='replicate')
        self.conv_h2 = \
            nn.Conv2d(num_hidden_out, num_hidden_out, 1,
                      stride=1, padding=1, padding_mod='replicate')

        self.ln_h = nn.LayerNorm(num_hidden_out * 4, elementwise_affine=True)
        self.ln_c = nn.LayerNorm(num_hidden_out * 3, elementwise_affine=True)
        self.ln_m = nn.LayerNorm(num_hidden_out * 3, elementwise_affine=True)
        self.ln_x = nn.LayerNorm(num_hidden_out * 7, elementwise_affine=True)
        self.ln_c2 = nn.LayerNorm(num_hidden_out * 4, elementwise_affine=True)
        self.ln_o = nn.LayerNorm(num_hidden_out, elementwise_affine=True)

    def forward(self, x, h=None, c=None, m=None):
        def run_layer_norm(x, ln):
            return ln(x.permute(0, 2, 3, 1)).permute(0, -1, 1, 2)

        if h is None:
            dim_h = [self.batch, self.num_hidden_out, self.height, self.width]
            h = torch.zeros(dim_h, dtype=x.dtype, device=x.device)

        if c is None:
            dim_c = [self.batch, self.num_hidden_out, self.height, self.width]
            c = torch.zeros(dim_c, dtype=x.dtype, device=x.device)

        if m is None:
            dim_m = [self.batch, self.num_hidden_in, self.height, self.width]
            m = torch.zeros(dim_m, dtype=x.dtype, device=x.device)

        h_cc = self.conv_h(h)
        c_cc = self.conv_c(c)
        m_cc = self.conv_m(m)

        if self.layer_norm:
            h_cc = run_layer_norm(h_cc, self.ln_h)
            c_cc = run_layer_norm(c_cc, self.ln_c)
            m_cc = run_layer_norm(m_cc, self.ln_m)

        i_h, g_h, f_h, o_h = torch.split(h_cc, 4, 1)
        i_c, g_c, f_c = torch.split(c_cc, 3, 1)
        i_m, f_m, m_m = torch.split(m_cc, 3, 1)

        if x is None:
            i = torch.sigmoid(i_h + i_c)
            f = torch.sigmoid(f_h + f_c + self._forget_bias)
            g = torch.tanh(g_h + g_c)
        else:
            x_cc = self.conv_x(x)
            if self.layer_norm:
                x_cc = run_layer_norm(x_cc, self.ln_x)

            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = torch.split(x_cc, 7, 1)

            i = torch.sigmoid(i_x + i_h + i_c)
            f = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
            g = torch.tanh(g_x + g_h + g_c)

        c_new = f * c + i * g

        c2m = self.conv_c2(c_new)
        if self.layer_norm:
            c2m = run_layer_norm(c2m, self.ln_c2)

        i_c, g_c, f_c, o_c = torch.split(c2m, 4, 1)

        if x is None:
            ii = torch.sigmoid(i_c + i_m)
            ff = torch.sigmoid(f_c + f_m + self._forget_bias)
            gg = torch.tanh(g_c)
        else:
            ii = torch.sigmoid(i_c + i_x_ + i_m)
            ff = torch.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
            gg = torch.tanh(g_c + g_x_)

        m_new = ff * torch.tanh(m_m) + ii * gg

        o_m = self.conv_o(m_new)
        if self.layer_norm:
            o_m = run_layer_norm(o_m, self.ln_o)

        if x is None:
            o = torch.tanh(o_h + o_c + o_m)
        else:
            o = torch.tanh(o_x + o_h + o_c + o_m)

        cell = torch.cat([c_new, m_new], -1)
        cell = self.conv_h2(cell)

        h_new = o * torch.tanh(cell)

        return h_new, c_new, m_new
