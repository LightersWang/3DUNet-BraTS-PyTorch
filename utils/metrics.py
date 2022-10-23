import numpy as np
import torch
from medpy.metric import hd95 as hd95_medpy
from monai.metrics import compute_hausdorff_distance
from torch import Tensor


def dice(output:Tensor, target:Tensor, eps: float=1e-5) -> np.ndarray:
    """calculate multilabel batch dice"""
    target = target.float()
    num = 2 * (output * target).sum(dim=(2,3,4)) + eps
    den = output.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + eps
    dsc = num / den

    return dsc.cpu().numpy()


def hd95(output:Tensor, target:Tensor, spacing=None) -> np.ndarray:
    """ output and target should all be boolean tensors"""
    output = output.bool().cpu().numpy()
    target = target.bool().cpu().numpy()
    
    B, C = target.shape[:2]
    hd95 = np.zeros((B, C), dtype=np.float64)
    for b in range(B):
        for c in range(C):
            pred, gt = output[b, c], target[b, c]

            # reward if gt all background, pred all background
            if (not gt.sum()) and (not pred.sum()):
                hd95[b, c] = 0.0
            # penalize if gt all background, pred has foreground
            elif (not gt.sum()) and (pred.sum()):
                hd95[b, c] = 373.1287
            # penalize if gt has forground, but pred has no prediction
            elif (gt.sum()) and (not pred.sum()):
                hd95[b, c] = 373.1287
            else:
                hd95[b, c] = hd95_medpy(pred, gt, voxelspacing=spacing)
    
    return hd95


def hd95_monai(output:Tensor, target:Tensor):
    B, C = target.shape[:2]
    hd = torch.zeros(B, C).float()
    for i in range(C):
        y_pred = output[:, i].unsqueeze(1)
        y = target[:, i].unsqueeze(1)
        hd[:, i] = compute_hausdorff_distance(
            y_pred, y, include_background=False, percentile=95).squeeze()

    target_sum = target.sum((2, 3, 4))
    output_sum = output.sum((2, 3, 4))
    reward  = (target_sum == 0) & (output_sum == 0)     # no false positive
    penalty = (target_sum == 0) & (output_sum > 0)      # false positive

    hd[reward] = 0.0
    hd[penalty] = 373.1287
    hd = torch.where(torch.isnan(hd), torch.tensor(373.1287), hd)
    
    return hd.cpu().numpy()


if __name__ == "__main__":
    # test dice
    shape = (8, 3, 128, 128, 128)
    x = torch.rand(*shape)
    y = torch.randint(2, shape)     # 2 for [0, 1]
    dsc = dice(x, y)
    print(dsc)

    x = torch.zeros(*shape).float()
    y = torch.zeros(*shape).float()
    # y[:, 1:] = 1
    hdist95 = hd95(x, y)
    print(hdist95)

