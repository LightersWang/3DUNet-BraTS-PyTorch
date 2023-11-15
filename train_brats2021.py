import os
import time
import warnings
from copy import deepcopy
from os.path import join

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast

import utils.metrics as metrics
from configs import parse_seg_args
from dataset import brats2021
from models import get_unet
from utils.loss import SoftDiceBCEWithLogitsLoss
from utils.misc import (AverageMeter, CaseSegMetricsMeterBraTS, ProgressMeter, LeaderboardBraTS,
                        brats_post_processing, initialization, load_cases_split, save_brats_nifti)
from utils.optim import get_optimizer
from utils.scheduler import get_scheduler


def train(args, epoch, model, train_loader, loss_fn, optimizer, scheduler, scaler, writer, logger):
    model.train()

    data_time = AverageMeter('Data', ':6.3f')
    batch_time = AverageMeter('Time', ':6.3f')
    bce_meter = AverageMeter('BCE', ':.4f')
    dsc_meter = AverageMeter('Dice', ':.4f')
    loss_meter = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader), 
        [batch_time, data_time, bce_meter, dsc_meter, loss_meter],
        prefix=f"Train: [{epoch}]")

    end = time.time()
    for i, (image, label, _, _) in enumerate(train_loader):
        # init
        image, label = image.cuda(), label.float().cuda()
        bsz = image.size(0)
        data_time.update(time.time() - end)

        with autocast((args.amp) and (scaler is not None)):
            # forward
            # TODO: adapt to deep supervision
            preds = model(image)
            bce_loss, dsc_loss = loss_fn(preds, label)
            loss = bce_loss + dsc_loss

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        if args.amp and scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad:
                scaler.unscale_(optimizer)  # enable grad clipping
                nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        # logging
        torch.cuda.synchronize()
        bce_meter.update(bce_loss.item(), bsz)
        dsc_meter.update(dsc_loss.item(), bsz)
        loss_meter.update(loss.item(), bsz)
        batch_time.update(time.time() - end)

        # monitor training progress
        if (i == 0) or (i + 1) % args.print_freq == 0:
            progress.display(i+1, logger)

        end = time.time()

    if scheduler is not None:
        scheduler.step()

    train_tb = {
        'bce_loss': bce_meter.avg, 
        'dsc_loss': dsc_meter.avg, 
        'total_loss': loss_meter.avg, 
        'lr': optimizer.state_dict()['param_groups'][0]['lr'],
    }

    for key, value in train_tb.items():
        writer.add_scalar(f"train/{key}", value, epoch)


def infer(args, epoch, model:nn.Module, infer_loader, writer, logger, mode:str, save_pred:bool=False):
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    case_metrics_meter = CaseSegMetricsMeterBraTS()
    
    # make save epoch folder
    folder_dir = mode if epoch is None else f"{mode}_epoch_{epoch:02d}"
    save_path = join(args.exp_dir, folder_dir)
    if not os.path.exists(save_path):
        os.system(f"mkdir -p {save_path}")

    with torch.no_grad():
        end = time.time()
        for i, (image, label, _, brats_names) in enumerate(infer_loader):
            # get data
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
            seg_map = brats_post_processing(seg_map)

            # calc metric 
            dice = metrics.dice(seg_map, label)
            hd95 = metrics.hd95(seg_map, label)

            # output seg map
            if save_pred:
                save_brats_nifti(seg_map, brats_names, mode, args.data_root, save_path)

            # logging
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            case_metrics_meter.update(dice, hd95, brats_names, bsz)

            # monitor training progress
            if (i == 0) or (i + 1) % args.print_freq == 0:
                mean_metrics = case_metrics_meter.mean()
                logger.info("\t".join([
                    f'{mode.capitalize()}: [{epoch}][{i+1}/{len(infer_loader)}]', str(batch_time), 
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

    # get validation metrics and log to tensorboard
    infer_metrics = case_metrics_meter.mean()
    for key, value in infer_metrics.items():
        writer.add_scalar(f"{mode}/{key}", value, epoch)
    
    return infer_metrics


def main():
    args = parse_seg_args()
    logger, writer = initialization(args)

    # dataloaders
    train_cases, val_cases, test_cases = load_cases_split(args.cases_split)
    train_loader = brats2021.get_train_loader(args, train_cases)
    val_loader   = brats2021.get_infer_loader(args, val_cases)
    test_loader  = brats2021.get_infer_loader(args, test_cases)

    # model & stuff
    model = get_unet(args).cuda()
    if args.data_parallel:
        model = nn.DataParallel(model).cuda()
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    loss = SoftDiceBCEWithLogitsLoss().cuda()
    if args.amp:
        scaler = GradScaler()
        logger.info("==> Using AMP (Auto Mixed Precision)")
    else:
        scaler = None

    # load model
    if args.weight_path is not None:
        logger.info("==> Loading pretrain model...")
        assert args.weight_path.endswith(".pth")
        model_state = torch.load(args.weight_path)['model']
        model.load_state_dict(model_state)

    # train & val
    logger.info("==> Training starts...")
    best_model = {}
    val_leaderboard = LeaderboardBraTS()
    for epoch in range(args.epochs):
        train(args, epoch, model, train_loader, loss, optimizer, scheduler, scaler, writer, logger)
        
        # validation
        if ((epoch + 1) % args.eval_freq == 0):
            logger.info(f"==> Validation starts...")
            # inference on validation set
            val_metrics = infer(args, epoch, model, val_loader, writer, logger, mode='val')
            
            # model selection
            val_leaderboard.update(epoch, val_metrics)
            best_model.update({epoch: deepcopy(model.state_dict())})
            logger.info(f"==> Validation ends...")
        
        torch.cuda.empty_cache()
    
    # ouput final leaderboard and its rank
    val_leaderboard.output(args.exp_dir)

    # test
    logger.info("==> Testing starts...")
    best_epoch = val_leaderboard.get_best_epoch()
    best_model = best_model[best_epoch]
    model.load_state_dict(best_model)
    infer(args, best_epoch, model, test_loader, writer, logger, mode='test', save_pred=args.save_pred)

    # save the best model on validation set
    if args.save_model:
        logger.info("==> Saving...")
        state = {'model': best_model, 'epoch': best_epoch, 'args':args}
        torch.save(state, os.path.join(
            args.exp_dir, f"test_epoch_{best_epoch:02d}", f'best_ckpt.pth'))
    
    logger.info("==> Testing ends...")


if __name__ == '__main__':
    main()