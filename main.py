import argparse
import torch
from model import Pretrain_SegmentationNet, DPRAN
import os
from data.dataloader import create_dataloader
from train import net_Pretrain, DPRAN_Train
import segmentation_models_pytorch as smp


def main():
    parser = argparse.ArgumentParser(description='DPRAN')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_channels', default=1, type=int, help='Dimension of the input CEUS frames')
    parser.add_argument('--lr_pre', default=0.002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr', default=0.002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [40]')
    parser.add_argument('--num_epochs_pre', default=50, type=int, help='Number of total training epochs [40]')
    parser.add_argument('--dataset', default='data', type=str, help='Dataset folder name')
    args = parser.parse_args()

    save_path = os.path.join('checkpoint')
    os.makedirs(save_path, exist_ok=True)

    layers = [32, 32, 64, 128]
    # data load and split
    train_loader, val_loader, test_loader = create_dataloader(dataset=args.dataset, batch_size=1, is_pretraining=True)

    # stage 1
    net = Pretrain_SegmentationNet(n_channels=args.num_channels, n_classes=args.num_classes, layers=layers)
    net.cuda()
    criterion = smp.losses.DiceLoss('binary', classes=None, log_loss=False, from_logits=True, smooth=0.0,
                                    ignore_index=None, eps=1e-07)
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_pre)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9862, last_epoch=-1)
    # Parameters
    epoch_start = 0
    epoch_end = args.num_epochs_pre
    print("Start net Pre-Training...")
    net = net_Pretrain(net, criterion, optimizer, scheduler, epoch_start, epoch_end, train_loader, val_loader,
                       save_path)
    # stage 2
    print("Start DPRAN Training...")
    model = DPRAN(n_channels=args.num_channels, n_classes=args.num_classes, layers=layers)
    model.encoder_ceus.load_state_dict(net.encoder.state_dict())
    model.cuda()

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9862, last_epoch=-1)
    epoch_end = args.num_epochs
    train_loader.dataset.is_pretraining = False
    val_loader.dataset.is_pretraining = False
    test_loader.dataset.is_pretraining = False
    test_result, trained_model = DPRAN_Train(model, net, criterion, optimizer, scheduler,
                                             epoch_start, epoch_end, train_loader, val_loader,
                                             test_loader)
    torch.save({'test_rec': test_result, 'DpRAN': trained_model, 'Pretrain_SegmentationNet': net.state_dict()},
               os.path.join(save_path, 'DpRAN' + '.pt'))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    SEED = 0
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    main()

