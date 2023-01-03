import torch
from thop import profile
from FusionVariants.Model_V import DPRAN_V


model_names = ['DpRAN', 'TCE', 'TCL', 'TCC', 'THD', 'NLE', 'NLL', 'NLC', 'NLH']
FLOPs_dict = {}

if __name__ == "__main__":
    images = torch.randn(1, 7, 336, 368)
    for name in model_names:
        if name == 'TCC' or name == 'NLC':
            layers = (24, 32, 64, 128)
        else:
            layers = (32, 32, 64, 128)
        if name == 'THD' or name == 'NLH':
            layers = (8, 16, 16, 24)
        model = DPRAN_V(n_channels=1, n_classes=1, layers=layers, name=name)
        if name == 'DpRAN':
            preds = torch.randn(1, 7, 336, 368)
            flops, params = profile(model.cuda(), inputs=(images.cuda(), preds.cuda()))
        else:
            flops, params = profile(model.cuda(), inputs=(images.cuda(), None))

        FLOPs_dict[name] = 'FLOPs = ' + str(flops / 1000 ** 3) + 'G'

    for key, val in FLOPs_dict.items():
        print(key)
        print(val)

"""
DpRAN
FLOPs = 21.783359706G
TCE
FLOPs = 21.327885096G
TCL
FLOPs = 22.08900036G
TCC
FLOPs = 25.45995396G
THD
FLOPs = 29.649789624G
NLE
FLOPs = 21.643632808G
NLL
FLOPs = 20.868112552G
NLC
FLOPs = 25.66119108G
NLH
FLOPs = 29.64583468G
"""
