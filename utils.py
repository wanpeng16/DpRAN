import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, n_layers=2):
        super().__init__()
        self.double_conv = DenseBlock(in_channels, out_channels, n_layers=n_layers)
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True)
                                      )

    def forward(self, x):
        return self.shortcut(x) + self.double_conv(x)


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=2):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i * out_channels, out_channels)
            for i in range(n_layers)])

    def forward(self, x):
        out = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.cat([x, out], dim=1)
            out = self.layers[i](x)
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        )

    def forward(self, x):
        return self.conv(x)
