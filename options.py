from copy import deepcopy
import os
import time
import pickle
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers to load data')
    # parser.add_argument('--save_prefix', type=str, default='pref', help='some comment for model or test result dir')

    # checkpoint
    parser.add_argument('--load_path', type=str, default='', help='model folder path for pretrain or test')
    parser.add_argument('--load_checkpoint', default=False, help='if load from checkpoint then change later')
    parser.add_argument('--load_model_path', default='', help='if load from checkpoint then change later')

    # data
    parser.add_argument('--data_type', type=str, default='brats2021', choices=['brats2021'], help='dataset hint')
    parser.add_argument('--num_modality', type=int, default=4, help='#modality == #encoder')
    parser.add_argument('--data_root', type=str, default='data/brats21/', help='root dir of dataset')
    parser.add_argument('--cases_split', type=str, help='name & split')
    parser.add_argument('--patch_size', type=int, default=128, help='patch size')
    parser.add_argument('--pos_ratio', type=float, default=1.0, help="prob of picking positive patch (center in foreground)")
    parser.add_argument('--neg_ratio', type=float, default=1.0, help="prob of picking negative patch (center not in foreground)")

    # model
    parser.add_argument('--model_type', type=str, default='mencoder_plain_unet', help='used in model_entry.py')
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=3, help='number of predicted classs')
    parser.add_argument('--num_downsample', type=int, default=5, help='#downsample == #level')
    parser.add_argument('--blocks_per_stage', type=int, default=2, help='blocks per stage')
    parser.add_argument('--deep_supervision', action='store_true', help='deep supervision for biggest 4 level')
    parser.add_argument('--dropout_prob', type=float, default=0.0, help="prob of dropout")
    parser.add_argument('--norm', type=str, default='instance', 
                        choices=['instance', 'batch', 'group'], help='type of norm')

    # train 
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--save_root', type=str, default='data/experiments', help='enter save root path')
    parser.add_argument('--val_save_freq', type=int, default=10, help='validation&save frequency')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency iteration')

    # eval
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--patch_overlap', type=float, default=0.5, help="overlap ratio between patches")
    parser.add_argument('--sliding_window_mode', type=str, default='constant', choices=['constant', 'gaussian'], 
        help='sliding window importance map mode')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    # parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O1', choices=['O1', 'O2'])

    # scheduler
    parser.add_argument('--scheduler', type=str, default='warmup_cosine', help='scheduler',
                        choices=['warmup_cosine', 'cosine', 'step', 'poly'])
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warm up epochs')
    parser.add_argument('--milestones', type=list, default=[60, 80], help='milestones for multistep decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='decay factor for multistep decay')
    
    args = parser.parse_args()
    return args


def get_save_path(args):
    # time as folder name
    str_time = time.strftime("%m%d_%H%M%S", time.localtime())
    save_folder_name = "_".join([str_time, args.data_type, args.model_type])
    save_folder_name = "{}_batchsize{}_epochs{}_{}_lr{}".format(
        save_folder_name, args.train_batch_size, args.epochs, args.optim, args.lr)

    # mkdir savepath
    save_path = os.path.join(args.save_root, save_folder_name)
    if not os.path.exists(save_path):
        os.system('mkdir -p ' + save_path)
    args.save_path = save_path


def save_args(args):
    # save args as txt file & pickle file
    args_path = os.path.join(args.save_path, 'args')
    with open(args_path+'.txt', 'w') as f:
        f.write(str(args).replace(', ', ',\n'))     # write as txt file
    with open(args_path+'.pickle', 'wb') as f:
        pickle.dump(args, f)        # write as pickle file


def load_checkpoint_args(args):
    """
    If training resume from a checkpoint, then use previous args

    Example args.load_path: data/experiments/1216_104930_xxx/epoch_x
    """
    load_path = args.load_path      # old args load path is empty or out-of-date, so save it for now
    ckp_args_path = os.path.join(os.path.dirname(load_path), 'args.pickle')
    with open(ckp_args_path, 'rb') as f:
        loaded_args = f.load(f)

    args = deepcopy(loaded_args)
    args.load_path = load_path
    args.load_checkpoint = True
    epoch = int(args.load_path.split("_")[-1])
    args.load_model_path = os.path.join(args.load_path, f"checkpoint_{epoch:02d}.pth")


def prepare_train_args():
    args = parse_args()
    if args.load_path != '':
        load_checkpoint_args(args)
    else:
        get_save_path(args)
        save_args(args)
    return args


if __name__ == '__main__':
    train_args = prepare_train_args()
    print(train_args)
