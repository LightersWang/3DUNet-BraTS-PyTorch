import os
import time
from os.path import join
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference

import utils.misc as utils
import utils.metrics as metrics
from dataset import brats2021
from configs import parse_seg_args
from models import get_unet
from utils.optim import get_optimizer
from utils.scheduler import get_scheduler
from utils.loss import SoftDiceBCEWithLogitsLoss


def train(args, epoch, model:nn.Module, train_loader, loss_fn, optimizer, scheduler, scaler):
    model.train()

    data_time = utils.AverageMeter('Data', ':6.3f')
    batch_time = utils.AverageMeter('Time', ':6.3f')
    bce_loss_meter = utils.AverageMeter('BCE', ':.4f')
    dsc_loss_meter = utils.AverageMeter('Dice', ':.4f')
    total_loss_meter = utils.AverageMeter('Loss', ':.4f')
    progress = utils.ProgressMeter(
        len(train_loader), 
        [batch_time, data_time, bce_loss_meter, dsc_loss_meter, total_loss_meter],
        prefix=f"Train: [{epoch}]")

    end = time.time()
    for i, (image, label, _, _) in enumerate(train_loader):
        # init
        bce_loss = torch.tensor(0.0, requires_grad=True).cuda()
        dsc_loss = torch.tensor(0.0, requires_grad=True).cuda()
        image, label = image.cuda(), label.float().cuda()
        bsz = image.size(0)
        data_time.update(time.time() - end)

        with autocast((args.amp) and (scaler is not None)):
            # forward
            preds = model(image)

            # calc loss weighting factor, works for both w/ or w/o deep supervision
            weights = np.array([1 / (2 ** j) for j in range(len(preds))])
            weights /= weights.sum()

            # calc losses
            for j in range(len(preds)):
                bce, dsc = loss_fn(preds[j], label)
                bce_loss += weights[j] * bce
                dsc_loss += weights[j] * dsc
            total_loss = bce_loss + dsc_loss

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        if args.amp and scaler is not None:
            scaler.scale(total_loss).backward()
            if args.clip_grad:
                scaler.unscale_(optimizer)  # enable grad clipping
                nn.utils.clip_grad_value_(model.parameters(), 40)
            # FIXME 'No inf checks were recorded for this optimizer' when using half precision
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_value_(model.parameters(), 40)
            optimizer.step()

        # logging
        torch.cuda.synchronize()
        bce_loss_meter.update(bce_loss.item(), bsz)
        dsc_loss_meter.update(dsc_loss.item(), bsz)
        total_loss_meter.update(total_loss.item(), bsz)
        batch_time.update(time.time() - end)

        # monitor training progress
        if (i == 0) or (i + 1) % args.print_freq == 0:
            progress.display(i+1)

        end = time.time()

    if scheduler is not None:
        scheduler.step()

    return {
        'bce_loss': bce_loss_meter.avg, 
        'dsc_loss': dsc_loss_meter.avg, 
        'total_loss': total_loss_meter.avg, 
        'lr': optimizer.state_dict()['param_groups'][0]['lr'],
    }


def infer(args, epoch, model:nn.Module, infer_loader, mode:str, save_pred:bool=False):
    model.eval()

    data_time = utils.AverageMeter('Data', ':6.3f')
    batch_time = utils.AverageMeter('Time', ':6.3f')
    case_metrics_meter = utils.CaseSegMetricsMeterBraTS()
    
    # make save epoch folder
    folder_dir = f"{mode}" if epoch is None else f"{mode}_epoch_{epoch:02d}"
    save_path = join(args.save_dir, folder_dir)
    if not os.path.exists(save_path):
        os.system(f"mkdir -p {save_path}")

    with torch.no_grad():
        end = time.time()
        for i, (image, label, _, brats_names) in enumerate(infer_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            image, label = image.cuda(), label.bool().cuda()
            bsz = image.size(0)

            # get seg map
            seg_map = sliding_window_inference(
                inputs=image, 
                predictor=model,
                roi_size=args.patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.patch_overlap,
                mode=args.sliding_window_mode
            )

            # discrete
            seg_map = torch.where(seg_map > 0.5, True, False)

            # post-processing
            seg_map = utils.brats_post_processing(seg_map)

            # calc metric 
            dice = metrics.dice(seg_map, label)
            hd95 = metrics.hd95(seg_map, label)

            # output seg map
            if save_pred:
                utils.save_brats_nifti(seg_map, brats_names, mode, args.data_root, save_path)

            # logging
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            case_metrics_meter.update(dice, hd95, brats_names, bsz)

            # monitor training progress
            if (i == 0) or (i + 1) % args.print_freq == 0:
                mean_metrics = case_metrics_meter.mean()
                print("\t".join([
                    f'{mode.capitalize()}: [{epoch}][{i+1}/{len(infer_loader)}]',
                    str(batch_time), str(data_time),
                    f"Dice_WT {dice[:, 1].mean():.3f} ({mean_metrics['Dice_WT']:.3f})",
                    f"Dice_TC {dice[:, 0].mean():.3f} ({mean_metrics['Dice_TC']:.3f})",
                    f"Dice_ET {dice[:, 2].mean():.3f} ({mean_metrics['Dice_ET']:.3f})",
                    f"HD95_WT {hd95[:, 1].mean():7.3f} ({mean_metrics['HD95_WT']:7.3f})",
                    f"HD95_TC {hd95[:, 0].mean():7.3f} ({mean_metrics['HD95_TC']:7.3f})",
                    f"HD95_ET {hd95[:, 2].mean():7.3f} ({mean_metrics['HD95_ET']:7.3f})",
                ]))

            end = time.time()

        # output case metric csv
        case_metrics_meter.output(save_path)

    return case_metrics_meter.mean()


def main():
    args = parse_seg_args()
    print("==>", args.comment)
    tb_logger = SummaryWriter(args.save_dir)
    utils.seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    # dataloaders
    train_cases, val_cases, test_cases = utils.load_cases_split(args.cases_split)
    train_loader = brats2021.get_train_loader(args, train_cases)
    val_loader   = brats2021.get_infer_loader(args, val_cases)
    test_loader  = brats2021.get_infer_loader(args, test_cases)

    # model & stuff
    model = get_unet(args).cuda()
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    loss = SoftDiceBCEWithLogitsLoss().cuda()
    scaler = GradScaler() if args.amp else None # auto mixed precision

    # train & val
    print("==> Training starts...")
    best_model = {}
    val_leaderboard = utils.LeaderboardBraTS()
    for epoch in range(args.epochs):
        train_tb = train(args, epoch, model, train_loader, loss, optimizer, scheduler, scaler)
        for key, value in train_tb.items():
            tb_logger.add_scalar("train/" + key, value, epoch)
        
        # validation
        if ((epoch + 1) % args.eval_freq == 0):
            print(f"\n==> Validation starts...")
            # inference on validation set
            val_metrics = infer(args, epoch, model, val_loader, mode='val')
            for key, value in val_metrics.items():
                tb_logger.add_scalar("val/" + key, value, epoch)
            
            # model selection
            val_leaderboard.update(epoch, val_metrics)
            best_model.update({epoch: deepcopy(model.state_dict())})                
            print(f"==> Validation ends...\n")
        
        torch.cuda.empty_cache()
    
    # ouput final leaderboard and its rank
    val_leaderboard.output(args.save_dir)

    # test
    print("\n==> Testing starts...")
    best_epoch = val_leaderboard.get_best_epoch()
    best_model = best_model[best_epoch]
    model.load_state_dict(best_model)
    test_metrics = infer(
        args, best_epoch, model, test_loader, mode='test', save_pred=args.save_pred)
    for key, value in test_metrics.items():
        tb_logger.add_scalar("test/" + key, value, best_epoch)

    # save the best model on validation set
    if args.save_model:
        print("==> Saving...")
        state = {'model': best_model, 'epoch': best_epoch, 'args':args}
        torch.save(state, os.path.join(
            args.save_dir, f"test_epoch_{best_epoch:02d}", f'best_ckpt.pth'))
    
    print("==> Testing ends...\n")


if __name__ == '__main__':
    main()