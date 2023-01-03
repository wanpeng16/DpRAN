import argparse
import torch
from model import Pretrain_SegmentationNet, DPRAN
from data.dataloader import create_dataloader
from train import eval_model_DPRAN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DPRAN')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_channels', default=1, type=int, help='Dimension of the input CEUS frames')
    parser.add_argument('--dataset', default='data', type=str, help='Dataset folder name')
    args = parser.parse_args()
    Path = "checkpoint/DpRAN.pt"
    Res = torch.load(Path)

    layers = [32, 32, 64, 128]
    # data load and split
    _, val_loader, test_loader = create_dataloader(dataset=args.dataset, batch_size=1, is_pretraining=True)

    # stage 1
    print("Load Pretrain_SegmentationNet Model...")
    net = Pretrain_SegmentationNet(n_channels=args.num_channels, n_classes=args.num_classes, layers=layers)
    net.cuda()
    net.load_state_dict(Res['Pretrain_SegmentationNet'])
    # stage 2
    print("Load DPRAN Model...")
    model = DPRAN(n_channels=args.num_channels, n_classes=args.num_classes, layers=layers)
    model.cuda()
    model.load_state_dict(Res['DpRAN'])
    val_iou, val_dice, val_hd, val_sen, va_spe = eval_model_DPRAN(model, net, val_loader)
    test_iou, test_dice, test_hd, test_sen, test_spe = eval_model_DPRAN(model, net, test_loader)
    print('\r Val iou: %.4f dice: %.4f hd: %.4f \r Test iou: %.4f dice: %.4f hd: %.4f' % (
        val_iou, val_dice, val_hd, test_iou, test_dice, test_hd)
          )
