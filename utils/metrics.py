import numpy as np
from medpy.metric import hd95 as hd95_medpy
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
