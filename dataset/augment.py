import numpy as np
import monai.transforms as transforms
from monai.transforms.transform import MapTransform


class RobustZScoreNormalization(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key] > 0

            lower = np.percentile(d[key][mask], 0.2)
            upper = np.percentile(d[key][mask], 99.8)

            d[key][mask & (d[key] < lower)] = lower
            d[key][mask & (d[key] > upper)] = upper

            y = d[key][mask]
            d[key] -= y.mean()
            d[key] /= y.std()

        return d


def get_brats2021_base_transform():
    base_transform = [
        transforms.Orientationd(keys=['flair', 't1', 't1ce', 't2', 'label'], axcodes="RAS"),
        transforms.AddChanneld(keys=['flair', 't1', 't1ce', 't2', 'label']),        # [B, H, W, D] --> [B, C, H, W, D]
        RobustZScoreNormalization(keys=['flair', 't1', 't1ce', 't2']),
        transforms.ConcatItemsd(keys=['flair', 't1', 't1ce', 't2'], name='image', dim=0),
        transforms.DeleteItemsd(keys=['flair', 't1', 't1ce', 't2']),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
    ]

    return base_transform


def get_brats2021_train_transform(patch_size:int=128, pos_ratio=1.0, neg_ratio=1.0):
    base_transform = get_brats2021_base_transform()

    data_aug = [
        # crop
        transforms.RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
            spatial_size=[patch_size]*3, pos=pos_ratio, neg=neg_ratio, num_samples=1),

        # spatial aug
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

        # intensity aug
        transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
        transforms.RandGaussianSmoothd(
            keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),

        # other stuff
        transforms.EnsureTyped(keys=["image", "label"]),
    ]

    return transforms.Compose(base_transform + data_aug)


def get_brats2021_val_transform():
    base_transform = get_brats2021_base_transform()

    val_transform = [transforms.EnsureTyped(keys=["image", "label"])]

    return transforms.Compose(base_transform + val_transform)