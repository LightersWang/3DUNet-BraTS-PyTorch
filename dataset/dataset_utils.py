import os
import numpy as np
import nibabel as nib

from os import listdir
from os.path import join


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


def nib_load(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError
    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data