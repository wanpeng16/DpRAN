import torch
import torch.nn as nn
from model import Encoder_PE, Encoder_Unet, Decoder
from utils import OutConv
from TCE import TC_Early
from TCL import TC_Late
from THD import HyperDenseNet_TC
from TCC import TC_Co
from NLE import Non_Local_Early
from NLL import Non_Local_Late
from NLC import Non_Local_Co
from NLH import HyperDenseNet_NL


class DPRAN_V(nn.Module):
    def __init__(self, n_channels, n_classes=1, layers=(16, 32, 64, 64), name='DpRAN'):
        super(DPRAN_V, self).__init__()
        self.encoder_ceus = Encoder_Unet(n_channels, layers=layers)
        self.encoder_pe = Encoder_PE(layers=layers)
        self.name = name
        if self.name == 'DpRAN':
            self.decoder = Decoder(n_classes, layers=layers)
        if self.name == 'THD':
            self.decoder = HyperDenseNet_TC(n_classes, layers=layers)
        if self.name == 'TCE':
            self.decoder = Decoder_V(n_classes, TC_Early, layers=layers)
        if self.name == 'TCL':
            self.decoder = Decoder_V(n_classes, TC_Late, layers=layers)
        if self.name == 'TCC':
            self.decoder = Decoder_V(n_classes, TC_Co, layers=layers)
        if self.name == 'NLE':
            self.decoder = Decoder_V(n_classes, Non_Local_Early, layers=layers)
        if self.name == 'NLL':
            self.decoder = Decoder_V(n_classes, Non_Local_Late, layers=layers)
        if self.name == 'NLC':
            self.decoder = Decoder_V(n_classes, Non_Local_Co, layers=layers)
        if self.name == 'NLH':
            self.decoder = HyperDenseNet_NL(n_classes, layers=layers)

    def forward(self, x, preds):
        bs, T = x.shape[0], x.shape[1]
        x = x.reshape((-1,) + x.shape[2:]).unsqueeze(1)
        x4, x3, x2, x1 = self.encoder_ceus(x)
        x4 = x4.reshape((bs, T) + x4.shape[1:])
        x3 = x3.reshape((bs, T) + x3.shape[1:])
        x2 = x2.reshape((bs, T) + x2.shape[1:])
        x1 = x1.reshape((bs, T) + x1.shape[1:])
        f4, f3, f2, f1 = self.encoder_pe(x4, x3, x2, x1)

        if self.name == 'DpRAN':
            logits = self.decoder(f4, f3, f2, f1, x4, x3, x2, x1, preds)
        else:
            logits = self.decoder(f4, f3, f2, f1, x4, x3, x2, x1)

        return logits


class Decoder_V(nn.Module):
    def __init__(self, n_classes, Up, layers=(256, 128, 64, 32)):
        super(Decoder_V, self).__init__()
        self.up0 = Up(in_channels_1=layers[3], in_channels_2=0, out_channels=layers[2])

        self.up1 = Up(in_channels_1=layers[2], in_channels_2=layers[2], out_channels=layers[1])

        self.up2 = Up(in_channels_1=layers[1], in_channels_2=layers[1], out_channels=layers[0])

        self.up3 = Up(in_channels_1=layers[0], in_channels_2=layers[0], out_channels=layers[0])

        self.up5 = nn.ConvTranspose2d(layers[0], layers[0] // 2, kernel_size=2, stride=2)

        self.outc = OutConv(layers[0] // 2, n_classes)

    def forward(self, f4, f3, f2, f1, x4, x3, x2, x1):
        x = self.up0(f4, x4, None)
        x = self.up1(f3, x3, x)
        x = self.up2(f2, x2, x)
        x = self.up3(f1, x1, x)
        x = self.up5(x)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    x = torch.randn(1, 7, 336, 368)
    model = DPRAN_V(n_channels=1, n_classes=1, layers=(32, 32, 64, 128), name='NLH')
    model.cuda()
    print(model(x.cuda(), None).shape)
