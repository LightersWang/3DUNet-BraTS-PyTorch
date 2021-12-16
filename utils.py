import os
from os.path import join

import nibabel as nib
import numpy as np
import pandas as pd
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CaseSegMetricMeter(object):
    """Stores segmentation metric (dice & hd95) for every case"""
    def __init__(self):
        self.cases = {}
        self.reset()

    def reset(self):
        self.cases = {}

    def update(self, dice, hd95, names, bsz):
        for i in range(bsz):
            self.cases.update({
                names[i]: [dice[i, 1], dice[i, 0], dice[i, 2], 
                           hd95[i, 1], hd95[i, 0], hd95[i, 2]]
            })

    def output(self, epoch, save_epoch_path):
        df = pd.DataFrame(self.cases).T
        df.columns = ['Dice_WT', 'Dice_TC', 'Dice_ET',
                      'HD95_WT', 'HD95_TC', 'HD95_ET']
        df.to_csv(join(save_epoch_path, f"val_metric_epoch{epoch}.csv"))


def load_cases_split(split_path:str):
    df = pd.read_csv(split_path)
    cases_name, cases_split = np.array(df['name']), np.array(df['split'])
    train_cases = list(cases_name[cases_split == 'train'])
    val_cases = list(cases_name[cases_split == 'val'])

    return train_cases, val_cases


def nib_affine(path):
    return nib.load(path).affine


def save_nifti(seg_map:torch.Tensor, names:list, save_epoch_path:str, args):
    """
    Output val seg map in every iteration to save VRAM
    """
    seg_map_numpy = seg_map.cpu().numpy()
    B, _, H, W, D = seg_map_numpy.shape

    # make save folder
    save_epoch_seg_path = join(save_epoch_path, f"val_seg_pred")
    if not os.path.exists(save_epoch_seg_path):
        os.system(f"mkdir -p {save_epoch_seg_path}")

    for b in range(B):
        output = seg_map_numpy[b]
        seg_img = np.zeros((H, W, D), dtype=np.uint8)

        seg_img[np.where(output[1, ...] == 1)] = 2      # WT --> ED
        seg_img[np.where(output[0, ...] == 1)] = 1      # TC --> NCR
        seg_img[np.where(output[2, ...] == 1)] = 4      # ET --> ET

        original_img_path = join(args.data_root, names[b], names[b]+f'_t1.nii.gz')      # random modality is ok
        affine = nib_affine(original_img_path)
        
        nib.save(
            nib.Nifti1Image(seg_img, affine), 
            join(save_epoch_seg_path, names[b]+f'_pred.nii.gz')
        )