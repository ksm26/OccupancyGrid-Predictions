import torch 
import torch.nn as nn 

# Code source: https://github.com/thuml/predrnn-pytorch/blob/master/core/layers/SpatioTemporalLSTMCell.py

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size, stride):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=num_hidden*7, kernel_size=filter_size, stride=stride,padding=self.padding ,bias=False)
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden*4, kernel_size=filter_size, stride=stride,padding=self.padding ,bias=False)
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden*3, kernel_size=filter_size, stride=stride,padding=self.padding ,bias=False)
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden*2, out_channels=num_hidden, kernel_size=filter_size, stride=stride,padding=self.padding ,bias=False)
        )
        self.conv_last = nn.Conv2d(num_hidden*2, num_hidden, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t
        
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + c_new + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


# Code source: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=False):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int   Number of channels of input tensor.
        hidden_dim: int  Number of channels of hidden state.
        kernel_size: (int, int)  Size of the convolutional kernel.
        bias: bool       Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = False

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, h_cur, c_cur):

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
