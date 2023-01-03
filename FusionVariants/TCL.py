import torch
import torch.nn as nn
import torch.nn.functional as F


class TC_Late(nn.Module):
    """Channel-wise Concatenation after Temporal Convolution"""

    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super().__init__()

        if in_channels_2:
            self.up = nn.ConvTranspose2d(in_channels_2, in_channels_2, kernel_size=2, stride=2)

        self.tc = nn.Sequential(
            nn.Conv3d(in_channels_1, in_channels_1, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(in_channels_1, in_channels_1, kernel_size=(3, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.AvgPool3d((2, 1, 1))
        )
        self.conv = nn.Conv2d(in_channels_1 * 2 + in_channels_2, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2, x3):
        """
        :param x1: dynamics map [T, C, H, W]
        :param x2:  appearance map [T, C, H, W]
        :param x3: deconvolution map
        :return:
        """
        # channel-wise concatenation
        x1 = x1.permute(0, 2, 1, 3, 4)
        x1 = torch.squeeze(self.tc(x1), dim=2)
        x2 = x2.permute(0, 2, 1, 3, 4)
        x2 = torch.squeeze(self.tc(x2), dim=2)
        out = torch.cat((x1, x2), dim=1)
        if x3 is not None:
            x3 = self.up(x3)
            # input is CHW
            diffY = out.size()[2] - x3.size()[2]
            diffX = out.size()[3] - x3.size()[3]

            x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            out = torch.cat([out, x3], dim=1)  # merge and refine

        return self.conv(out)


if __name__ == "__main__":
    x1 = torch.randn(1, 7, 32, 168, 184)
    x2 = torch.randn(1, 7, 32, 168, 184)
    x3 = torch.randn(1, 64, 84, 92)
    m = TC_Late(in_channels_1=32, in_channels_2=64, out_channels=32)
    print(m(x1, x2, x3).shape)
