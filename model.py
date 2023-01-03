import torch
import torch.nn as nn
import torch.nn.functional as F
from function import PerfusionExcitation
from function import AFF
from function import FrameSelection
from utils import Down, DoubleConv, OutConv


class Encoder_Unet(nn.Module):
    def __init__(self, n_channels, layers=(32, 64, 128, 256)):
        super(Encoder_Unet, self).__init__()
        mid_channels = layers[0] // 2
        out_channels = layers[0]
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.down1 = Down(layers[0], layers[0])
        self.down2 = Down(layers[0], layers[1])
        self.down3 = Down(layers[1], layers[2])
        self.down4 = Down(layers[2], layers[3])

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return x4, x3, x2, x1


# frame-level segmentation at Stage I
class Pretrain_SegmentationNet(nn.Module):
    def __init__(self, n_channels, n_classes, layers=(16, 32, 64, 64)):
        super(Pretrain_SegmentationNet, self).__init__()
        self.encoder = Encoder_Unet(n_channels, layers=layers)
        self.outc = OutConv(layers[-1], n_classes)

    def forward(self, x):  # [T, C, H, W]
        _, _, h, w = x.shape
        x4, _, _, _ = self.encoder(x)
        x5 = self.outc(x4)   # -> 1 channel
        x = F.interpolate(x5, size=(h, w), mode='bilinear', align_corners=False) # up-sample to the full resolution
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels_1, in_channels_2, out_channels, dim, grid=(55, 60),
                 original_scale=(330, 360)):
        super().__init__()

        if in_channels_2:
            self.up = nn.ConvTranspose2d(in_channels_2, in_channels_2, kernel_size=2, stride=2)
        self.coatt = AFF(in_channels_1, dim, grid, original_scale, bias=False, heads=1)
        self.conv = DoubleConv(in_channels_1 + in_channels_2, out_channels)

    def forward(self, x1, x2, x3, m_indices):
        """
        :param x1: dynamics map [T, C, H, W]
        :param x2:  appearance map [T, C, H, W]
        :param x3: deconvolution map
        :return:
        """
        out, _, _ = self.coatt(functionalF=x1, enhancementF=x2, m_indices=m_indices)  # [1, C, H, W]
        if x3 is not None:
            x3 = self.up(x3)
            # input is CHW
            diffY = out.size()[2] - x3.size()[2]
            diffX = out.size()[3] - x3.size()[3]

            x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            out = torch.cat([out, x3], dim=1)   # merge and refine

        return self.conv(out)


class Encoder_PE(nn.Module):
    def __init__(self, layers=(32, 64, 128, 256)):
        super(Encoder_PE, self).__init__()
        self.pe1 = PerfusionExcitation(out_channels=layers[0])
        self.pe2 = PerfusionExcitation(out_channels=layers[1])
        self.pe3 = PerfusionExcitation(out_channels=layers[2])
        self.pe4 = PerfusionExcitation(out_channels=layers[3])

    def forward(self, x4, x3, x2, x1):
        f4 = self.pe4(x4)
        f3 = self.pe3(x3)
        f2 = self.pe2(x2)
        f1 = self.pe1(x1)
        return f4, f3, f2, f1


class Decoder(nn.Module):
    def __init__(self, n_classes, layers=(256, 128, 64, 32)):
        super(Decoder, self).__init__()
        self.up0 = Up(in_channels_1=layers[3], in_channels_2=0, out_channels=layers[2], dim=32,
                      grid=(4, 4), original_scale=(21, 23))

        self.up1 = Up(in_channels_1=layers[2], in_channels_2=layers[2], out_channels=layers[1], dim=32,
                      grid=(4, 4), original_scale=(42, 46))

        self.up2 = Up(in_channels_1=layers[1], in_channels_2=layers[1], out_channels=layers[0], dim=32,
                      grid=(4, 4), original_scale=(84, 92))

        self.up3 = Up(in_channels_1=layers[0], in_channels_2=layers[0], out_channels=layers[0], dim=32,
                      grid=(4, 4), original_scale=(168, 184))

        self.up5 = nn.ConvTranspose2d(layers[0], layers[0] // 2, kernel_size=2, stride=2)

        self.outc = OutConv(layers[0] // 2, n_classes)
        self.frameselect = FrameSelection()

    def forward(self, f4, f3, f2, f1, x4, x3, x2, x1, preds):
        m_index = self.frameselect(preds)
        x = self.up0(f4, x4, None, m_index)
        x = self.up1(f3, x3, x, m_index)
        x = self.up2(f2, x2, x, m_index)
        x = self.up3(f1, x1, x, m_index)
        x = self.up5(x)
        logits = self.outc(x)
        return logits


class DPRAN(nn.Module):
    def __init__(self, n_channels, n_classes=1, layers=(16, 32, 64, 64)):
        super(DPRAN, self).__init__()
        self.encoder_ceus = Encoder_Unet(n_channels, layers=layers)
        self.encoder_pe = Encoder_PE(layers=layers)
        self.decoder = Decoder(n_classes, layers=layers)

    def forward(self, x, preds):
        bs, T = x.shape[0], x.shape[1]
        x = x.reshape((-1,) + x.shape[2:]).unsqueeze(1)
        x4, x3, x2, x1 = self.encoder_ceus(x)
        x4 = x4.reshape((bs, T) + x4.shape[1:])
        x3 = x3.reshape((bs, T) + x3.shape[1:])
        x2 = x2.reshape((bs, T) + x2.shape[1:])
        x1 = x1.reshape((bs, T) + x1.shape[1:])
        f4, f3, f2, f1 = self.encoder_pe(x4, x3, x2, x1)
        logits = self.decoder(f4, f3, f2, f1, x4, x3, x2, x1, preds)
        return logits


if __name__ == "__main__":
    DPRAN = DPRAN(n_channels=1, n_classes=1)
    DPRAN.cuda()
    X = torch.rand((16, 1, 330, 360))
    print(DPRAN(X.to('cuda:0')).shape)
