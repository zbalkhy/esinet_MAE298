import torch.nn as nn
import torch.nn.functional as F
from torch import flatten

class ConvDipNet(nn.Module):
    def __init__(self, in_channels, im_shape, n_filters, 
                 kernel_size, activation=F.relu, 
                 stride=1, fc_size=512, output_size=5124, final_batch_norm=False):
        super().__init__()
        self.height, self.width = im_shape
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.stride = stride
        self.fc_size = fc_size
        self.output_size = output_size
        self.final_batch_norm = final_batch_norm

        self.conv1 = nn.Conv2d(self.in_channels, self.n_filters, 
                               self.kernel_size, stride = self.stride, padding='same')
        self.bn1 = nn.BatchNorm2d(self.n_filters)
        self.conv2 = nn.Conv2d(self.n_filters, self.n_filters, 
                               self.kernel_size, stride = self.stride, padding='same')
        self.bn2 = nn.BatchNorm2d(self.n_filters)
        self.dropout = nn.Dropout(0.3)
        self.hidden_layer = nn.Linear(self.height*self.width*self.n_filters, self.fc_size)
        self.bn3 = nn.BatchNorm1d(self.fc_size)
        self.output_layer = nn.Linear(self.fc_size, self.output_size)
        if final_batch_norm:
            self.bn4 = nn.BatchNorm1d(self.output_size)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = flatten(x, 1) # flatten all dimensions except batch
        x = self.activation(self.bn3(self.hidden_layer(x)))

        if self.final_batch_norm:
            x = self.bn4(self.output_layer(x)) #self.activation(self.bn3(self.output_layer(x)))
        else:
            x = self.output_layer(x) #self.activation(self.output_layer(x))
        return x