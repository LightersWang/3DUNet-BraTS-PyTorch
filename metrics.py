import torch
from torch import Tensor
from monai.metrics import compute_hausdorff_distance


def dice(output:Tensor, target:Tensor, eps: float=1e-5):
    """calculate multilabel batch dice"""
    target = target.float()
    num = 2 * (output * target).sum(dim=(2,3,4)) + eps
    den = output.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + eps
    dsc = num / den

    return dsc.cpu().numpy()


def hd95(output:Tensor, target:Tensor):
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

