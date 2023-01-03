import copy

import torch
import torch.nn as nn
from spatial_correlation_sampler import SpatialCorrelationSampler
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import cv2
from scipy.stats import entropy
import torch.nn.functional as F
import math


class PerfusionExcitation(nn.Module):
    def __init__(self, out_channels, patch_size=21, kernel_size=1, stride=1, pad=0, dilation=1, dilation_patch=2):
        super(PerfusionExcitation, self).__init__()
        self.corr = SpatialCorrelationSampler(kernel_size=kernel_size, patch_size=patch_size, stride=stride,
                                              padding=pad, dilation=dilation, dilation_patch=dilation_patch)

        self.conv = nn.Sequential(nn.BatchNorm2d(patch_size * patch_size),
                                  nn.Conv2d(in_channels=patch_size * patch_size, out_channels=out_channels,
                                            kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                                  # linear projection
                                  )

    def forward(self, feats_ts):
        bs = feats_ts.shape[0]
        frame_t_1 = feats_ts[:, :-1]
        frame_t_2 = feats_ts[:, 1:]
        cor_t = [self.corr(frame_t_1[i], frame_t_2[i]) for i in range(bs)]
        cor_t = torch.stack(cor_t)
        B, T, PatchH, PatchW, oH, oW = cor_t.shape
        cor_t = cor_t.reshape(B * T, PatchH * PatchW, oH, oW)
        perfusiondiff = self.conv(cor_t)
        perfusiondiff = perfusiondiff.reshape((B, T) + perfusiondiff.shape[1:])
        perfusiondiff = torch.cat((perfusiondiff, torch.zeros_like(perfusiondiff[:, 1]).unsqueeze(1)),
                                  dim=1)
        return perfusiondiff


class AFF(nn.Module):
    # Cross-attention Temporal Aggregation Module
    def __init__(self, in_channels, out_channels, grid=(55, 60), original_scale=(330, 360),
                 bias=False, heads=1):
        super(AFF, self).__init__()

        self.aff1 = AttentionTF(in_channels, out_channels,
                                grid=grid, original_scale=original_scale, bias=bias)
        self.aff2 = AttentionTF(in_channels, out_channels,
                                grid=grid, original_scale=original_scale, bias=bias)

    def forward(self, functionalF, enhancementF, m_indices, q1=None, q2=None):
        """
        co-attention for modality fusion
        :param functionalF: [B, T, C, H, W]
        :param enhancementF: [B, T, C, H, W]
        :return:
        """
        if q1 is None:
            q1 = enhancementF[:, m_indices[0]]
        if q2 is None:
            q2 = functionalF[:, m_indices[0]]

        outputs_f1 = self.aff1(x=functionalF, query_x=q1)  # Morphology-guided dynamics aggregation
        outputs_f2 = self.aff2(x=enhancementF, query_x=q2)  # Dynamics-guided morphology aggregation

        return outputs_f1 + outputs_f2, outputs_f1, outputs_f2


# Standard positional encoding (addition/ concat both are valid)
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=16):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [B, T, C, H, W]
        :param self.pe [1, max_seq_len, d_model]
        :return:
        """
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        batch_size = x.size(0)
        num_feature = x.size(2)
        spatial_h = x.size(3)
        spatial_w = x.size(4)
        z = Variable(self.pe[:, :seq_len], requires_grad=False)
        z = z.unsqueeze(-1).unsqueeze(-1)  # [1, T, d_model, 1, 1]
        z = z.expand(batch_size, seq_len, num_feature, spatial_h, spatial_w)
        x = x + z
        return x


class AttentionTF(nn.Module):
    """
    Self-attention based temporal fusion
    """

    def __init__(self, in_channels, out_channels, grid=(4, 4), original_scale=(330, 360), bias=False):
        super(AttentionTF, self).__init__()

        self.posenc = PositionalEncoder(d_model=in_channels, max_seq_len=16)
        self.out_channels = out_channels
        self.grids = grid
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias)
        self.avg = nn.AdaptiveAvgPool2d(grid)
        self.upsample = nn.Upsample(size=original_scale, mode='nearest')
        self.reset_parameters()

    # standard attention layer
    def attention(self, q, k, v, d_k):
        # [B,T,C,H,W]
        scores = torch.sum(q * k, 2) / math.sqrt(d_k)
        scores = F.softmax(scores, dim=1)
        scores = self.upsample(scores)  # [B,T,1, H,W]
        scores = scores.unsqueeze(2).expand_as(v)
        output = scores * v
        output = torch.sum(output, 1)
        return output

    def forward(self, x, query_x):
        batch, t, channels, height, width = x.size()
        dim = self.out_channels
        DPRANk = query_x
        # value maps [B,T,C,H,W]
        v_out = self.value_conv(x.view((-1,) + x.shape[2:]))
        v = v_out.view((-1, t) + v_out.shape[1:])
        # key
        x = self.avg(x.view((-1,) + x.shape[2:]))
        x = self.posenc(x.view((-1, t) + x.shape[1:]))  # temporal position encoding
        k_out = self.key_conv(x.view((-1,) + x.shape[2:]))
        k = k_out.view((-1, t) + k_out.shape[1:])
        # query [B,C,H,W]
        q_out = self.query_conv(self.avg(query_x))
        q = q_out.unsqueeze(1).expand_as(k)

        out = self.attention(q, k, v, d_k=dim)
        out += DPRANk

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')


class FrameSelection(nn.Module):
    def __init__(self):  # K, L, N
        super(FrameSelection, self).__init__()

    def Confidence_Estimate(self, foreground):
        _, bw = cv2.threshold(foreground, 0.5, 1, cv2.THRESH_BINARY)
        boundary_pred = foreground[np.nonzero(bw)]
        ent = 0.
        for i in range(boundary_pred.shape[0]):
            ent += entropy([boundary_pred[i], 1.0 - boundary_pred[i]])
        if boundary_pred.shape[0] == 0:
            return 1000000
        else:
            return ent / boundary_pred.shape[0]

    def forward(self, preds):
        # estimate prediction confidence
        score = np.empty((preds.shape[0], preds.shape[1]))
        preds = torch.sigmoid(preds)
        for bs in range(preds.shape[0]):
            for i in range(preds.shape[1]):
                score[bs, i] = self.Confidence_Estimate(preds[bs, i].detach().cpu().numpy())
        score = torch.from_numpy(score).cuda()

        # select critical point
        _, m_indices = torch.sort(score, dim=1, descending=False)
        return m_indices[:, 0]
