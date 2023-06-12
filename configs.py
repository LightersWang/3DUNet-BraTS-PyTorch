import os
import time
from argparse import ArgumentParser


def parse_seg_args():
    """args of segmentation tasks"""

    parser = ArgumentParser()
    parser.add_argument('--comment', type=str, default='', help='save comment')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers to load data')
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--data_parallel', action='store_true', help='using data parallel')

    # path & dir
    parser.add_argument('--exp_dir', type=str, default='exps', help='experiment dir')
    parser.add_argument('--save_freq', type=int, default=10, help='model save frequency (epoch)')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequency (iteration)')

    # data
    parser.add_argument('--dataset', type=str, default='brats2021', help='dataset hint', 
        choices=['brats2021', 'brats2018'])
    parser.add_argument('--data_root', type=str, default='data/', help='root dir of dataset')
    parser.add_argument('--cases_split', type=str, help='name & split')
    parser.add_argument('--input_channels', '--n_views', type=int, default=4, 
        help="#channels of input data, equal to #encoders in multiencoder unet and" \
             "#view in multiview contrastive learning")

    # data augmentation
    parser.add_argument('--patch_size', type=int, default=128, help='patch size')
    parser.add_argument('--pos_ratio', type=float, default=1.0, 
        help="prob of picking positive patch (center in foreground)")
    parser.add_argument('--neg_ratio', type=float, default=1.0, 
        help="prob of picking negative patch (center in background)")

    # optimize
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--optim', type=str, default='adamw', help='optimizer', 
        choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--beta1', default=0.9, type=float, metavar='M', 
        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, metavar='M', help='beta2 for adam')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--scheduler', type=str, default='none', help='scheduler',
        choices=['warmup_cosine', 'cosine', 'step', 'poly', 'none'])
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warm up epochs')
    parser.add_argument('--milestones', type=int, nargs="+", default=[60, 80], 
        help='milestones for multistep decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1, 
        help='decay factor for multistep decay')
    parser.add_argument('--clip_grad', action='store_true', help='whether to clip gradient')

    # u-net
    parser.add_argument('--unet_arch', type=str, default='unet', 
        choices=['unet', 'multiencoder_unet', 'unetr'], help='Architecuture of the U-Net')
    parser.add_argument('--block', type=str, default='plain', choices=['plain', 'res'],
        help='Type of convolution block')
    parser.add_argument('--channels_list', nargs='+', type=int, default=[32, 64, 128, 256, 320, 320],
        help="#channels of every levels of decoder in a top-down order")
    parser.add_argument('--kernel_size', type=int, default=3, help="size of conv kernels")
    parser.add_argument('--dropout_prob', type=float, default=0.0, help="prob of dropout")
    parser.add_argument('--norm', type=str, default='instance', 
        choices=['instance', 'batch', 'group'], help='type of norm')
    parser.add_argument('--num_classes', type=int, default=3, help='number of predicted classs')
    parser.add_argument('--weight_path', type=str, default=None, 
        help='path to pretrained encoder or decoder weight, None for train-from-scratch')
    parser.add_argument('--deep_supervision', action='store_true', 
        help='whether use deep supervision')
    parser.add_argument('--ds_layer', type=int, default=4, 
        help='last n layer to use deep supervision')

    # eval
    parser.add_argument('--save_model', action='store_true', default=False, 
        help='whether save model state')
    parser.add_argument('--save_pred', action='store_true', default=False, 
        help='whether save individual prediction')
    parser.add_argument('--eval_freq', type=int, default=10, help='eval frequency')
    parser.add_argument('--infer_batch_size', type=int, default=4, help='batchsize for inference')
    parser.add_argument('--patch_overlap', type=float, default=0.5, 
        help="overlap ratio between patches")
    parser.add_argument('--sw_batch_size', type=int, default=2, help="sliding window batch size")
    parser.add_argument('--sliding_window_mode', type=str, default='constant', 
        choices=['constant', 'gaussian'], help='sliding window importance map mode')

    args = parser.parse_args()

    # generate save path
    exp_dir_name = [
        args.comment, 
        args.dataset,
        args.unet_arch,
        args.optim,
        args.scheduler,
        f"pos{args.pos_ratio}",
        f"neg{args.neg_ratio}",
    ]
    exp_dir_name.append(time.strftime("%m%d_%H%M%S", time.localtime()))
    exp_dir_name = "_".join(exp_dir_name)
    args.exp_dir += exp_dir_name

    return args
