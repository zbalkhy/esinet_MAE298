from torch import nn

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, in_channels, height, width) '''
        batch_size, time_steps, C, H, W = x.size()
        c_in = x.view(batch_size * time_steps, C, H, W)
        c_out = self.module(c_in)
        r_in = c_out.view(batch_size, time_steps, -1)
        if self.batch_first is False:
            r_in = r_in.permute(1, 0, 2)
        return r_in
    
class TimeDistributedLinear(nn.Module):
    def __init__(self, input_size, output_size, batch_first=False, batch_norm=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_first = batch_first
        self.batch_norm = batch_norm

        self.linear = nn.Linear(input_size, output_size)
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_size)
    
    def forward(self, x):
        ''' x size: (batch_size, time_steps, features) '''
        batch_size, time_steps, L = x.size()
        c_in = x.reshape(batch_size * time_steps, L)
        c_out = self.linear(c_in)
        if self.batch_norm:
            c_out = self.bn(c_out)
        r_in = c_out.view(batch_size, time_steps, -1)
        if self.batch_first is False:
            r_in = r_in.permute(1, 0, 2)
        return r_in