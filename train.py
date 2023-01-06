import copy
import os.path
import numpy as np
import torch
from torch.autograd import Variable
from metric import accuracy_check_for_batch
import sys
import datetime
from torch.nn import functional as F
from torchmetrics import Dice
import segmentation_models_pytorch as smp

save_path = os.path.join('checkpoint', datetime.date.today().strftime("%m%d%Y"))


def train_model(model, data_train, criterion, optimizer):
    # Train
    model.train()
    for batch, (images, masks, _) in enumerate(data_train):
        optimizer.zero_grad()
        images = images.reshape((-1,) + images.shape[2:])
        images = images.unsqueeze(1)
        masks = masks.reshape((-1,) + masks.shape[2:])
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
        outputs = model(images)  # [T, C, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        # Update weights
        optimizer.step()
        sys.stdout.write('\r Training seq [%d/%d] seq loss: %.4f' % (batch, len(data_train), loss.item()))


def train_model_DPRAN(model, net, data_train, criterion, optimizer):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        data_train (DataLoader): training dataset
    """
    model.train()
    net.eval()
    total_loss = 0.
    for batch, (images, masks, _) in enumerate(data_train):
        optimizer.zero_grad()
        bs, T = images.shape[0], images.shape[1]
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
        # obtain initial prediction
        preds = net(images.reshape((-1,) + images.shape[2:]).unsqueeze(1))  # [T, C, H, W]
        preds = preds.reshape((bs, T) + images.shape[2:])  # [B, T, H, W]
        # fuse feature and feed into Decoder
        outputs = model(images, preds)
        loss = criterion(outputs, masks)
        loss.backward()
        # Update weights
        optimizer.step()
        sys.stdout.write('\r Training seq [%d/%d] seq loss: %.4f' % (batch, len(data_train), loss.item()))
        total_loss = total_loss + loss

    return total_loss / (batch + 1)


def eval_model(model, data_train):
    model.eval()
    total_acc = 0

    for batch, (images, masks, _) in enumerate(data_train):
        images = images.reshape((-1,) + images.shape[2:])
        images = images.unsqueeze(1)
        masks = masks.reshape((-1,) + masks.shape[2:]).long()
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
        preds = model(images)  # [T, C, H, W]  [T, H, W]
        preds = torch.nn.Sigmoid()(preds)
        tp, fp, fn, tn = smp.metrics.get_stats(preds, masks, mode='binary', threshold=0.5)
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        total_acc = total_acc + accuracy

    return total_acc / (batch + 1)


def eval_model_DPRAN(model, net, data_loader):
    model.eval()
    net.eval()
    total_iou = []
    total_dice = []
    total_hd = []
    total_sensitivity = []
    total_specificity = []
    dice = Dice(average='micro').cuda()

    for batch, (images, masks, _) in enumerate(data_loader):
        with torch.no_grad():
            bs, T = images.shape[0], images.shape[1]
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            masks = masks.long()
            # obtain initial prediction
            preds = net(images.reshape((-1,) + images.shape[2:]).unsqueeze(1))  # [T, C, H, W]
            preds = preds.reshape((bs, T) + images.shape[2:])  # [B, T, H, W]
            outputs = F.logsigmoid(model(images, preds)).exp()
            tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks, mode='binary', threshold=0.5)
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
            total_iou = total_iou + [iou.detach().cpu()]
            sensitivity = smp.metrics.functional.sensitivity(tp, fp, fn, tn, reduction="macro")
            total_sensitivity = total_sensitivity + [sensitivity.detach().cpu()]
            specificity = smp.metrics.functional.specificity(tp, fp, fn, tn, reduction="macro")
            total_specificity = total_specificity + [specificity.detach().cpu()]
            total_dice = total_dice + [dice(outputs, masks).detach().cpu()]
            outputs = torch.where(outputs >= 0.5, 1, 0)
            hd = accuracy_check_for_batch(masks[:, 0].detach().cpu(), outputs[:, 0].detach().cpu(),
                                          batch_size=outputs.shape[0], type='hd')
            total_hd = total_hd + [hd]

    total_iou, total_dice, total_hd, total_sensitivity, total_specificity = np.array(total_iou), np.array(
        total_dice), np.array(total_hd), np.array(total_sensitivity), np.array(total_specificity)

    return total_iou.mean(), total_dice.mean(), total_hd.mean(), total_sensitivity.mean(), total_specificity.mean()


def net_Pretrain(net, criterion, optimizer, scheduler, epoch_start, epoch_end, train_loader, val_loader, save_path):
    if os.path.exists(os.path.join(save_path, 'net.pt')):
        net.load_state_dict(torch.load(os.path.join(save_path, 'net.pt')))
        print('Pre-trained net is loaded')
    else:
        best_val_model = None
        best_val = 0
        for epoch_cur in range(epoch_start, epoch_end):
            # train the model
            train_model(net, train_loader, criterion, optimizer)
            scheduler.step()
            if epoch_cur >= epoch_end // 2:
                val_acc = eval_model(net, val_loader)  # val the model
                if val_acc > best_val:
                    best_val = val_acc
                    best_val_model = copy.deepcopy(net.state_dict())
                    print(
                        '\r Epoch [%d/%d] Val acc: %.4f' % (epoch_cur, epoch_end, val_acc))

        net.load_state_dict(best_val_model)
        torch.save(best_val_model, os.path.join(save_path, 'net.pt'))
    return net


def DPRAN_Train(model, net, criterion, optimizer, scheduler, epoch_start, epoch_end, train_loader, val_loader,
                test_loader):
    # Train
    best_val = 0
    best_val_model = None

    for i in range(epoch_start, epoch_end):
        train_loss = train_model_DPRAN(model, net, train_loader, criterion, optimizer)
        scheduler.step()
        sys.stdout.write(
            '\r Epoch [%d/%d] Train loss: %.4f' % (
                i, epoch_end, train_loss)
        )
        # Validation every epoch
        if i >= 1:
            val_iou, val_dice, val_hd, _, _, = eval_model_DPRAN(model, net, val_loader)

            sys.stdout.write('\r Epoch [%d/%d] Val iou: %.4f dice: %.4f hd: %.4f' % (
                i, epoch_end, val_iou, val_dice, val_hd)
                             )
            if val_dice > best_val:
                best_val = val_dice
                sys.stdout.write(
                    '\r Best Val Epoch [%d/%d] Val dice: %.4f' % (i, epoch_end, best_val))
                best_val_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_val_model)
    # Validation best epoch
    test_iou, test_dice, test_hd, test_sen, test_spe = eval_model_DPRAN(model, net, test_loader)
    test_rec = [test_sen, test_spe, test_dice, test_iou, test_hd]

    return test_rec, best_val_model
