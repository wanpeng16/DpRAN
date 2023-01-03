from PIL import Image
from torch.utils import data
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
import re
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, data_list, base_path, is_training=True, is_pretraining=True, is_static=False):
        self.is_training = is_training
        self.data_list = data_list
        self.base_path = base_path
        self.is_pretraining = is_pretraining
        self.is_static = is_static

    def __len__(self):
        return len(self.data_list)

    def PatchFilter(self, dataset, cset, patch='ceus'):
        def PatchSel(x):
            return x.__contains__(patch)

        newlist = filter(PatchSel, cset)
        newlist = sorted(newlist, key=lambda x: int(re.findall('\d+', x)[0]))
        if self.is_pretraining:
            newlist = newlist[1:-1]
        if self.is_static:
            newlist = newlist[1:6:2]
        return [os.path.join(dataset, p) for p in newlist]

    def __getitem__(self, index):
        ceus_id = self.data_list[index]
        imgs = self.PatchFilter(os.path.join(self.base_path, ceus_id),
                                os.listdir(os.path.join(self.base_path, ceus_id)))

        X = [self.transform(Image.open(img)) for img in imgs]
        X = torch.cat(X, dim=0)
        mask = self.transform(Image.open(os.path.join(self.base_path, ceus_id, 'mask.png')))
        if self.is_pretraining:
            mask = mask.unsqueeze(0).repeat_interleave(X.shape[0], dim=0)

        return X, mask.type(torch.float), ceus_id

    def transform(self, img):
        return transforms.Compose([transforms.Grayscale(),
                                   transforms.Resize((336, 368)),
                                   transforms.ToTensor(),
                                   transforms.ConvertImageDtype(torch.float32),
                                   ])(img)


def create_dataloader(dataset, batch_size=4, is_pretraining=True, is_static=False):
    train_path = np.array(os.listdir(os.path.join(dataset, 'train')))
    val_path = np.array(os.listdir(os.path.join(dataset, 'val')))
    test_path = np.array(os.listdir(os.path.join(dataset, 'test')))

    train_loader = torch.utils.data.DataLoader(
        Dataset(data_list=train_path, base_path=os.path.join(dataset, 'train'), is_training=True,
                is_pretraining=is_pretraining,
                is_static=is_static),
        batch_size=batch_size, shuffle=False,
        drop_last=True, pin_memory=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        Dataset(data_list=val_path, base_path=os.path.join(dataset, 'val'), is_training=False,
                is_pretraining=is_pretraining,
                is_static=is_static),
        batch_size=1, shuffle=False,
        drop_last=False, pin_memory=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        Dataset(data_list=test_path, base_path=os.path.join(dataset, 'test'), is_training=False,
                is_pretraining=is_pretraining,
                is_static=is_static),
        batch_size=1, shuffle=False,
        drop_last=False, pin_memory=False, num_workers=4)

    return train_loader, val_loader, test_loader
