import torch
import torch.nn as nn
from utils import OutConv
import torch.nn.functional as F


class HyperDenseNet_TC(nn.Module):
    def __init__(self, nClasses, layers=(32, 32, 64, 128)):
        super(HyperDenseNet_TC, self).__init__()
        in_channels_1 = layers[0]
        self.tc = nn.Sequential(
            nn.Conv3d(in_channels_1 * 2, in_channels_1, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(in_channels_1, in_channels_1, kernel_size=(3, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.AvgPool3d((2, 1, 1))
        )

        self.outc = OutConv(in_channels_1, nClasses)

        self.conv1_top = nn.Sequential(
            nn.Conv2d(layers[0] * 2, layers[1], kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )
        self.conv1_bottom = nn.Sequential(
            nn.Conv2d(layers[0] * 2, layers[1], kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )
        self.conv2_top = nn.Sequential(
            nn.Conv2d(layers[0] * 2 + layers[1] * 4, layers[2], kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )
        self.conv2_bottom = nn.Sequential(
            nn.Conv2d(layers[0] * 2 + layers[1] * 4, layers[2], kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )

        self.conv3_top = nn.Sequential(
            nn.Conv2d(layers[0] * 2 + layers[1] * 4 + layers[2] * 4, layers[3], kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )
        self.conv3_bottom = nn.Sequential(
            nn.Conv2d(layers[0] * 2 + layers[1] * 4 + layers[2] * 4, layers[3], kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )
        self.conv4_top = nn.Sequential(
            nn.Conv2d(layers[0] * 2 + layers[1] * 4 + layers[2] * 4 + layers[3] * 4, layers[0], kernel_size=3,
                      padding=1, bias=True),
            nn.ReLU()
        )
        self.conv4_bottom = nn.Sequential(
            nn.Conv2d(layers[0] * 2 + layers[1] * 4 + layers[2] * 4 + layers[3] * 4, layers[0], kernel_size=3,
                      padding=1, bias=True),
            nn.ReLU()
        )

    def hyperdense(self, f4, f3, f2, f1, x4, x3, x2, x1):
        x2 = F.interpolate(x2, scale_factor=2)
        f2 = F.interpolate(f2, scale_factor=2)
        x3 = F.interpolate(x3, scale_factor=4)
        f3 = F.interpolate(f3, scale_factor=4)
        x4 = F.interpolate(x4, scale_factor=8)
        f4 = F.interpolate(f4, scale_factor=8)

        # concatenate
        y2t_i = torch.cat((x1, f1), dim=1)  # layer[0] * 2
        y2b_i = torch.cat((f1, x1), dim=1)

        y2t_o = self.conv1_top(y2t_i)  # layer[1]
        y2b_o = self.conv1_bottom(y2b_i)

        # concatenate
        y3t_i = torch.cat((y2t_i, y2t_o, y2b_o, x2, f2), dim=1)  # layer[0] * 2 + layer[1]*4
        y3b_i = torch.cat((y2b_i, y2b_o, y2t_o, f2, x2), dim=1)

        y3t_o = self.conv2_top(y3t_i)  # layer[2]
        y3b_o = self.conv2_bottom(y3b_i)

        y4t_i = torch.cat((y3t_i, y3t_o, y3b_o, x3, f3), dim=1)  # layer[0] * 2 + layer[1]*4 + layer[2]*4
        y4b_i = torch.cat((y3b_i, y3b_o, y3t_o, f3, x3), dim=1)

        y4t_o = self.conv3_top(y4t_i)  # layer[3]
        y4b_o = self.conv3_bottom(y4b_i)

        y5t_i = torch.cat((y4t_i, y4t_o, y4b_o, x4, f4), dim=1)
        y5b_i = torch.cat((y4b_i, y4b_o, y4t_o, f4, x4), dim=1)

        y5t_o = self.conv4_top(y5t_i)
        y5b_o = self.conv4_bottom(y5b_i)

        return torch.cat((y5t_o, y5b_o), dim=1)

    def forward(self, f4, f3, f2, f1, x4, x3, x2, x1):
        # ----- First layer ------ #

        x = []
        T = x1.shape[1]
        for t in range(T):
            x.append(self.hyperdense(f4[:, t], f3[:, t], f2[:, t], f1[:, t], x4[:, t], x3[:, t], x2[:, t], x1[:, t]))

        x = torch.stack(x)
        out = x.permute(1, 2, 0, 3, 4)
        out = torch.squeeze(self.tc(out), dim=2)
        pred = F.interpolate(self.outc(out), scale_factor=2)

        return pred


if __name__ == "__main__":
    x1 = torch.randn(1, 7, 32, 168, 184)
    x2 = torch.randn(1, 7, 32, 84, 92)
    x3 = torch.randn(1, 7, 64, 42, 46)
    x4 = torch.randn(1, 7, 128, 21, 23)
    f1 = torch.randn(1, 7, 32, 168, 184)
    f2 = torch.randn(1, 7, 32, 84, 92)
    f3 = torch.randn(1, 7, 64, 42, 46)
    f4 = torch.randn(1, 7, 128, 21, 23)
    m = HyperDenseNet_TC(nClasses=1)
    print(m(f4, f3, f2, f1, x4, x3, x2, x1).shape)
