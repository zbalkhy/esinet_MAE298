import torch.nn as nn
import torch.nn.functional as F
from torch import flatten

class ConvDipNet(nn.Module):
    def __init__(self, in_channels, height, width, n_filters, 
                 kernel_size, activation=F.relu, 
                 stride=1, fc_size=512, output_size=5124):
        super().__init__()
        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.stride = stride
        self.fc_size = fc_size

        self.conv1 = nn.Conv2d(self.in_channels, self.n_filters, 
                               self.kernel_size, stride = self.stride, padding='same')
        self.hidden_layer = nn.Linear(self.height*self.width*self.n_filters, self.fc_size)
        self.output_layer = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = self.activation(self.hidden_layer(x))
        x = self.activation(self.output_layer(x))
        return x