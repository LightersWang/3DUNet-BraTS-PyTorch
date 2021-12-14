import sys
sys.path.append("/home/whr/Code/multi_modal_al/seg/")

import os
from os import listdir
from os.path import join

import numpy as np
import torch.utils.data as data

from dataset.dataset_utils import nib_load
from dataset.augment import get_brats2021_train_transform, get_brats2021_val_transform


def process_f32(img_dir):
    """ Set all Voxels that are outside of the brain mask to 0"""
    modalities = ['t2', 't1ce', 'flair', 't1']
    name = os.path.basename(img_dir)
    images = np.stack([
        np.array(nib_load(join(img_dir, name) + '_' + i + '.nii.gz'),
                 dtype='float32', order='C') for i in modalities], -1)  # [240, 240, 155, 4]
    mask = images.sum(-1) > 0       # [240, 240, 155]

    for k in range(len(modalities)):
        x = images[..., k]
        y = x[mask]  # get brain mask

        lower = np.percentile(y, 0.2)
        upper = np.percentile(y, 99.8)

        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper

        y = x[mask]
        x -= y.mean()
        x /= y.std()

        images[..., k] = x

    return images


class BraTS2021Dataset(data.Dataset):
    def __init__(self, data_root:str, split:str='train', case_names:list=[], transforms=None):
        super(BraTS2021Dataset, self).__init__()
        
        self.split = split              # 'train' or 'val'
        self.data_root = data_root
        self.case_names = case_names
        self.transforms = transforms
        
    def __getitem__(self, index:int) -> tuple:
        name = self.case_names[index]                           # BraTS2021_00000
        base_dir = join(self.data_root, name, name)             # seg/data/brats21/BraTS2021_00000/BraTS2021_00000

        flair = np.array(nib_load(base_dir + '_flair.nii.gz'), dtype='float32')
        t1    = np.array(nib_load(base_dir + '_t1.nii.gz'), dtype='float32')
        t1ce  = np.array(nib_load(base_dir + '_t1ce.nii.gz'), dtype='float32')
        t2    = np.array(nib_load(base_dir + '_t2.nii.gz'), dtype='float32')
        gt    = np.array(nib_load(base_dir + '_seg.nii.gz'), dtype='uint8')

        if self.split == 'train':
            item = self.transforms({'flair':flair, 't1':t1, 't1ce':t1ce, 't2':t2, 'label':gt})[0]   # [0] for RandCropByPosNegLabeld
        elif self.split == 'val':
            item = self.transforms({'flair':flair, 't1':t1, 't1ce':t1ce, 't2':t2, 'label':gt})
        else:
            raise NotImplementedError(self.split)

        return item['image'], item['label'], index, name

    def __len__(self):
        return len(self.case_names)


def get_brats2021_train_loader(args, case_names:list):
    train_transform = get_brats2021_train_transform(
        args.patch_size, args.pos_ratio, args.neg_ratio)
    train_dataset = BraTS2021Dataset(
        args.data_root, 'train', case_names, train_transform)

    return data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, 
        drop_last=False, num_workers=args.num_workers, pin_memory=True)


def get_brats2021_val_loader(args, case_names:list):
    valid_transform = get_brats2021_val_transform()
    valid_dataset = BraTS2021Dataset(
        args.data_root, 'val', case_names, valid_transform)

    return data.DataLoader(
        valid_dataset, batch_size=args.eval_batch_size, shuffle=False, 
        drop_last=False, num_workers=args.num_workers, pin_memory=True)


if __name__ == "__main__":
    class ArgsTest():
        def __init__(self, a, b, c, d):
            self.data_root = a
            self.patch_size = b
            self.train_batch_size = c
            self.eval_batch_size = c
            self.num_workers = d
            self.pos_ratio = 1.0
            self.neg_ratio = 1.0

    args = ArgsTest("data/brats21/", 128, 2, 0)
    print(args.data_root, args.patch_size, args.train_batch_size, args.eval_batch_size, args.num_workers)

    case_names = sorted(listdir(args.data_root))
    print(len(case_names), '\n', case_names[:10])

    # train loader 
    train_loader = get_brats2021_train_loader(args, case_names)
    print(len(train_loader))
    item = next(iter(train_loader))
    
    # # val loader
    # val_loader = get_brats2021_val_loader(args, case_names)
    # print(len(val_loader))
    # item = next(iter(val_loader))

    # print(item[-1])
    # for i in range(len(item) - 1):
    #     print(item[i].shape)

    # # import matplotlib
    # # print(matplotlib.get_backend())    
    # import matplotlib.pyplot as plt

    # fig, axes = plt.subplots(5, 3)

    # t1ce = item[0][0, 2].squeeze().cpu().numpy()
    # axes[0, 0].imshow(t1ce[t1ce.shape[0] // 2].T, cmap='gray')
    # axes[0, 1].imshow(t1ce[:, t1ce.shape[1] // 2].T, cmap='gray')
    # axes[0, 2].imshow(t1ce[..., t1ce.shape[2] // 2].T, cmap='gray')

    # flair = item[0][0, 0].squeeze().cpu().numpy()
    # axes[1, 0].imshow(flair[flair.shape[0] // 2].T, cmap='gray')
    # axes[1, 1].imshow(flair[:, flair.shape[1] // 2].T, cmap='gray')
    # axes[1, 2].imshow(flair[..., flair.shape[2] // 2].T, cmap='gray')

    # label1 = item[-2][0, 0].squeeze().cpu().numpy()
    # axes[2, 0].imshow(label1[t1ce.shape[0] // 2].T)
    # axes[2, 1].imshow(label1[:, t1ce.shape[1] // 2].T)
    # axes[2, 2].imshow(label1[..., t1ce.shape[2] // 2].T)

    # label2 = item[-2][0, 1].squeeze().cpu().numpy()
    # axes[3, 0].imshow(label2[t1ce.shape[0] // 2].T)
    # axes[3, 1].imshow(label2[:, t1ce.shape[1] // 2].T)
    # axes[3, 2].imshow(label2[..., t1ce.shape[2] // 2].T)

    # label3 = item[-2][0, 2].squeeze().cpu().numpy()
    # axes[4, 0].imshow(label3[t1ce.shape[0] // 2].T)
    # axes[4, 1].imshow(label3[:, t1ce.shape[1] // 2].T)
    # axes[4, 2].imshow(label3[..., t1ce.shape[2] // 2].T)
    # plt.show()

    # # check shape
    # from dataset.augment import get_brats2021_base_transform
    # import monai.transforms as transforms
    # train_dataset = BraTS2021Dataset(
    #     args.data_root, 'train', case_names, get_brats2021_train_transform())
    # img = train_dataset[0]
    # print(img)