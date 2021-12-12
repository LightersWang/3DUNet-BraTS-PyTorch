import numpy as np
import pandas as pd

def load_cases_split(split_path:str):
    df = pd.read_csv(split_path)
    cases_name, cases_split = np.array(df['name']), np.array(df['split'])
    train_cases = list(cases_name[cases_split == 'train'])
    val_cases = list(cases_name[cases_split == 'val'])

    return train_cases, val_cases


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


