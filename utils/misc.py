import os
import sys
import random
import logging
from os.path import join
from collections import OrderedDict

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if logger is None:
            print('\t'.join(entries))
        else:
            logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class CaseSegMetricsMeterBraTS(object):
    """Stores segmentation metric (dice & hd95) for every case"""
    cols = ['Dice_WT', 'Dice_TC', 'Dice_ET', 'HD95_WT', 'HD95_TC', 'HD95_ET']

    def __init__(self):
        self.reset()

    def reset(self):
        self.cases = pd.DataFrame(columns=self.cols)

    def update(self, dice, hd95, names, bsz):
        for i in range(bsz):
            self.cases.loc[names[i]] = [
                dice[i, 1], dice[i, 0], dice[i, 2], 
                hd95[i, 1], hd95[i, 0], hd95[i, 2],
            ]

    def mean(self):
        return self.cases.mean(0).to_dict()

    def output(self, save_epoch_path):
        # all cases csv
        self.cases.to_csv(join(save_epoch_path, "case_metrics.csv"))
        # summary txt
        self.cases.mean(0).to_csv(join(save_epoch_path, "case_metrics_summary.txt"), sep='\t')


class LeaderboardBraTS(object):
    """Model selection using leaderboard. """
    cols = ['Dice_WT', 'Dice_TC', 'Dice_ET', 'HD95_WT', 'HD95_TC', 'HD95_ET']

    def __init__(self):
        self.reset()

    def reset(self):
        self.cases = pd.DataFrame(columns=self.cols)
        self.case_rank = None

    def update(self, epoch, metrics):
        df = pd.DataFrame(data=metrics, index=[epoch])
        self.cases = pd.concat([self.cases, df], axis=0)

    def rank(self):
        dice_rank = self.cases.iloc[:, :3].rank('index', method='min', ascending=False)
        hd95_rank = self.cases.iloc[:, 3:].rank('index', method='min', ascending=True)
        self.case_rank = pd.concat([dice_rank, hd95_rank], axis=1)
    
    def get_best_epoch(self):
        self.rank()
        return self.case_rank.mean(1).idxmin()
    
    def output(self, dir_path):
        # only run once
        self.rank()
        self.cases.to_csv(join(dir_path, "final_leaderboard.csv"))
        self.case_rank['Mean_Rank'] = self.case_rank.mean(1)
        self.case_rank.to_csv(join(dir_path, "final_leaderboard_rank.csv"))


def load_cases_split(split_path:str):
    df = pd.read_csv(split_path)
    cases_name, cases_split = np.array(df['name']), np.array(df['split'])
    train_cases = list(cases_name[cases_split == 'train'])
    val_cases   = list(cases_name[cases_split == 'val'])
    test_cases  = list(cases_name[cases_split == 'test'])

    return train_cases, val_cases, test_cases


def nib_affine(path):
    return nib.load(path).affine


def save_brats_nifti(seg_map:Tensor, names:list, mode:str, data_root:str, save_epoch_path:str):
    """
    Output val seg map in every iteration to save VRAM
    """
    seg_map_numpy = seg_map.cpu().numpy()
    B, _, H, W, D = seg_map_numpy.shape

    # make save folder
    save_epoch_seg_path = join(save_epoch_path, f"{mode}_seg_pred")
    if not os.path.exists(save_epoch_seg_path):
        os.system(f"mkdir -p {save_epoch_seg_path}")

    for b in range(B):
        output = seg_map_numpy[b]
        seg_img = np.zeros((H, W, D), dtype=np.uint8)

        seg_img[np.where(output[1, ...] == 1)] = 2      # WT --> ED
        seg_img[np.where(output[0, ...] == 1)] = 1      # TC --> NCR
        seg_img[np.where(output[2, ...] == 1)] = 4      # ET --> ET

        # random modality is ok
        original_img_path = join(data_root, 'brats2021', names[b], names[b]+f'_t1.nii.gz')
        affine = nib_affine(original_img_path)
        
        nib.save(
            nib.Nifti1Image(seg_img, affine), 
            join(save_epoch_seg_path, names[b]+f'_pred.nii.gz')
        )


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True


def initialize_logging(exp_dir):
    # mkdir
    log_fname = os.path.join(exp_dir, 'log.log')
    LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    DATE_FORMAT = '%Y%m%d %H:%M:%S'

    logger = logging.getLogger("contrastive_pretrain")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(fmt=LOGGING_FORMAT, datefmt=DATE_FORMAT)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        fh = logging.FileHandler(log_fname)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


def initialization(args):
    # set random seed
    seed_everything(args.seed)

    # make exp dir 
    writer = SummaryWriter(args.exp_dir)

    # init logger & save args
    logger = initialize_logging(args.exp_dir)
    logger.info(f"{'-'*20} New Experiment {'-'*20}")
    logger.info(' '.join(sys.argv))
    logger.info(args)

    return logger, writer


def brats_post_processing(seg_map):
    """ 
        post-processing from brats 2021 1st solution:
        Convert ET into NEC if #ET voxels < 200 (0-TC, 1-WT, 2-ET)
    """
    B, C = seg_map.shape[:2]
    for b in range(B):
        if seg_map[b, 2].sum() < 200:   # ET voxels
            seg_map[b, 2] = 0           # erase all ET voxels
    return seg_map
