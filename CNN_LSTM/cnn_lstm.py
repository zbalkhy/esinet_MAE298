import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
from convnet import ConvDipNet
from timeDistributed import TimeDistributed, TimeDistributedLinear

class CNNLSTM(nn.Module):
    def __init__(self, in_channels, im_shape, n_filters,
                 kernel_size, activation=F.relu,
                 stride=1, fc1_size=512, fc2_size=5124, 
                 hidden_features=1024, num_lstm_layers=1, 
                 fc3_size=512, out_size=5124):
        super().__init__()
        
        self.activation = activation
        self.fc2_size=fc2_size
        self.fc3_size = fc3_size
        self.hidden_features = hidden_features
        self.out_size = out_size
        self.num_lstm_layers= num_lstm_layers
        
        self.convdip = ConvDipNet(in_channels, im_shape, n_filters, kernel_size, 
                                  activation, stride, fc1_size, self.fc2_size)
        self.time_distributed_conv = TimeDistributed(self.convdip, batch_first=True)
        self.time_distributed_linear = TimeDistributedLinear(
            nn.Linear(self.fc2_size, self.fc3_size), 
            batch_first=True)
        self.lstm = nn.LSTM(self.fc3_size, self.hidden_features, 
                            self.num_lstm_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hidden_features)
        self.time_distributed_linear_2 = TimeDistributedLinear(
            nn.Linear(self.hidden_features, self.out_size), 
            batch_first=True)
    
    def forward(self, x, h0=None, c0=None):
        # initialize hidden and cell states
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_lstm_layers, x.size(
                0), self.hidden_features).to(x.device)
            c0 = torch.zeros(self.num_lstm_layers, x.size(
                0), self.hidden_features).to(x.device)
        
        x = self.time_distributed_conv(x) # no need for activation after this layer
        x = self.activation(self.time_distributed_linear(x))
        x, (h_n, c_n) = self.lstm(x, (h0, c0))
        x = self.layer_norm(x)
        x = self.activation(self.time_distributed_linear_2(x))
        return x

