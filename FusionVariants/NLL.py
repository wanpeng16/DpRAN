import torch
import torch.nn as nn
import torch.nn.functional as F


class Non_Local_Late(nn.Module):
    """Channel-wise Concatenation after Non-local TA"""

    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super().__init__()

        if in_channels_2:
            self.up = nn.ConvTranspose2d(in_channels_2, in_channels_2, kernel_size=2, stride=2)

        self.conv = nn.Conv2d(in_channels_1 * 2 + in_channels_2, out_channels, kernel_size=3, padding=1)

        self.fc_q = nn.Linear(in_channels_1, in_channels_1, bias=False)
        self.fc_k = nn.Linear(in_channels_1, in_channels_1, bias=False)
        self.fc_v = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def non_local(self, x):
        """
        :param x: [B, T, C, H, W]
        :return:
        """
        T = x.shape[1]
        x = x.view((-1,)+x.shape[2:])
        query_i = F.adaptive_avg_pool2d(x, (1, 1))  # [T, C, 1, 1]
        query = self.fc_q(query_i.view((query_i.shape[0], query_i.shape[1])))

        key_i = F.adaptive_avg_pool2d(x, (1, 1))
        key = self.fc_k(key_i.view((key_i.shape[0], key_i.shape[1])))

        score = torch.mm(query, torch.transpose(key, 0, 1))
        score = F.softmax(score, dim=1)  # [T, T]

        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        val = self.fc_v(x)
        fused_x = torch.mm(score, val.view((T, -1)))
        fused_x = fused_x.view((T, C, H, W)) + x

        fused_x = torch.mean(fused_x, dim=0)

        return fused_x.unsqueeze(0)

    def forward(self, x1, x2, x3):
        """
        :param x1: dynamics map [T, C, H, W]
        :param x2:  appearance map [T, C, H, W]
        :param x3: deconvolution map
        :return:
        """
        # channel-wise concatenation
        x1 = self.non_local(x1)
        x2 = self.non_local(x2)
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
    m = Non_Local_Late(in_channels_1=32, in_channels_2=64, out_channels=32)
    print(m(x1, x2, x3).shape)
