import numpy as np
from numpy.core.umath_tests import inner1d
from scipy import ndimage


def dice_check(mask, prediction):
    np_ims = [mask.numpy(), prediction]
    smooth = 1
    y_true_f = np_ims[0].flatten()
    y_pred_f = np_ims[1].flatten()
    intersection = np.logical_and(np_ims[0], np_ims[1]).sum()
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return dice


def ModHausdorffDist_check(mask, prediction):
    np_ims = [mask.numpy(), prediction]
    # Find pairwise distance
    D_mat = np.sqrt(inner1d(np_ims[0], np_ims[0])[np.newaxis].T + inner1d(np_ims[1], np_ims[1]) - 2 * (
        np.dot(np_ims[0], np_ims[1].T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat, axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat, axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))
    return MHD


# Hausdorff Distance
def HausdorffDist_check(mask, prediction):
    np_ims = [mask.numpy(), prediction]
    # Find pairwise distance
    D_mat = np.sqrt(inner1d(np_ims[0], np_ims[0])[np.newaxis].T + inner1d(np_ims[1], np_ims[1]) - 2 * (
        np.dot(np_ims[0], np_ims[1].T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))
    return dH


def meanIOU_check(mask, prediction):
    np_ims = [mask.numpy(), prediction]
    intersection = np.logical_and(np_ims[0], np_ims[1]).sum()
    union = np.logical_or(np_ims[0], np_ims[1]).sum()
    if union == 0:
        iou_score = 0
    else:
        iou_score = intersection / union
    return iou_score


def accuracy_check(mask, prediction):
    np_ims = [mask.numpy(), prediction]
    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)
    return accuracy / len(np_ims[0].flatten())


def accuracy_check_for_batch(masks, predictions, batch_size, type='acc'):
    total_acc = 0
    if type == 'acc':
        for index in range(batch_size):
            total_acc += accuracy_check(masks[index], predictions[index].numpy())
    if type == 'dice':
        for index in range(batch_size):
            total_acc += dice_check(masks[index], predictions[index].numpy())
    if type == 'iou':
        for index in range(batch_size):
            total_acc += meanIOU_check(masks[index], predictions[index].numpy())
    if type == 'hd':
        for index in range(batch_size):
            total_acc += HausdorffDist_check(masks[index], predictions[index].numpy())

    return total_acc / batch_size
