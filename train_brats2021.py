import os
from os.path import join
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter

try:
    from apex import amp, optimizers
except ImportError:
    pass

import metrics
from dataset.brats2021 import get_brats2021_train_loader, get_brats2021_val_loader
from loss import SoftDiceBCEWithLogitsLoss
from model.model_entry import DeepSupervisionUNetEval, select_model
from optim import get_optimizer
from options import prepare_train_args
from scheduler import get_scheduler
from utils import AverageMeter, CaseSegMetricMeter, load_cases_split, save_nifti


def train(args, epoch, model, train_loader, loss, optimizer, scheduler):
    # switch to train mode
    model.train()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    bce_loss_meter = AverageMeter()
    dsc_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    end = time.time()
    for iter, (image, label, _, _) in enumerate(train_loader):
        # init
        bce_loss = torch.tensor(0.).cuda()
        dsc_loss = torch.tensor(0.).cuda()
        image, label = image.cuda(), label.float().cuda()
        bsz = image.size(0)
        data_time.update(time.time() - end) 

        # forward
        pred = model(image)

        # calc loss weighting factor, work for both w/ or w/o deep supervision
        weights = np.array([1 / (2 ** j) for j in range(len(pred))])
        weights /= weights.sum()

        # calc losses
        for j in range(len(pred)):
            bce, dsc = loss(pred[j], label)
            bce_loss += weights[j] * bce
            dsc_loss += weights[j] * dsc
        total_loss = bce_loss + dsc_loss

        # compute gradient and do Adam step
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        # prevent gradient explosion
        nn.utils.clip_grad_value_(model.parameters(), 40)
        optimizer.step() 

        # logging losses
        bce_loss_meter.update(bce_loss.item(), bsz)
        dsc_loss_meter.update(dsc_loss.item(), bsz)
        total_loss_meter.update(total_loss.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)

        # monitor training progress
        if iter % args.print_freq == 0:
            print(f'Train: [{epoch}][{iter + 1}/{len(train_loader)}]\t'
                f'BT {batch_time.val:.3f} ({batch_time.avg:.3f}) \t'
                f'DT {data_time.val:.3f} ({data_time.avg:.3f}) \t'
                f'bce_loss {bce_loss_meter.val:.3f} ({bce_loss_meter.avg:.3f}) \t'
                f'dsc_loss {dsc_loss_meter.val:.3f} ({dsc_loss_meter.avg:.3f}) \t'
                f'total_loss {total_loss_meter.val:.3f} ({total_loss_meter.avg:.3f}) \t')

        end = time.time()

    scheduler.step()

    return {
        'bce_loss': bce_loss_meter.avg, 
        'dsc_loss': dsc_loss_meter.avg, 
        'total_loss': total_loss_meter.avg, 
        'lr': optimizer.state_dict()['param_groups'][0]['lr'],
    }


def val(args, epoch, model:nn.Module, val_loader, save_epoch_path:str):
    model.eval()

    batch_time = AverageMeter()
    wt_dice_meter = AverageMeter()
    tc_dice_meter = AverageMeter()
    et_dice_meter = AverageMeter()
    wt_hd95_meter = AverageMeter()
    tc_hd95_meter = AverageMeter()
    et_hd95_meter = AverageMeter()
    case_metric_meter = CaseSegMetricMeter()

    with torch.no_grad():
        end = time.time()
        for iter, (image, label, _, brats_names) in enumerate(val_loader):
            image, label = image.cuda(), label.float().cuda()
            bsz = image.size(0)

            # get seg map
            model_eval = DeepSupervisionUNetEval(model)     # wrapper for infer
            seg_map = sliding_window_inference(
                inputs=image, 
                roi_size=args.patch_size,
                sw_batch_size=1,
                predictor=model_eval,
                overlap=args.patch_overlap,
                mode=args.sliding_window_mode
            )

            # discrete
            seg_map = torch.where(seg_map > 0.5, 1.0, 0.0)

            # calc metric 
            dice = metrics.dice(seg_map, label)
            hd95 = metrics.hd95(seg_map, label)

            # output seg map
            save_nifti(seg_map, brats_names, save_epoch_path, args)

            # logging
            wt_dice_meter.update(dice[:, 1].mean(), bsz)
            tc_dice_meter.update(dice[:, 0].mean(), bsz)
            et_dice_meter.update(dice[:, 2].mean(), bsz)
            wt_hd95_meter.update(hd95[:, 1].mean(), bsz)
            tc_hd95_meter.update(hd95[:, 0].mean(), bsz)
            et_hd95_meter.update(hd95[:, 2].mean(), bsz)
            case_metric_meter.update(dice, hd95, brats_names, bsz)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            # monitor training progress
            if iter % args.print_freq == 0:
                print(f'Val: [{iter + 1}/{len(val_loader)}]\t'
                    f'BT {batch_time.val:.3f} ({batch_time.avg:.3f}) \t'
                    f'Dice_WT {wt_dice_meter.val:.3f} ({wt_dice_meter.avg:.3f}) \t'
                    f'Dice_TC {tc_dice_meter.val:.3f} ({tc_dice_meter.avg:.3f}) \t'
                    f'Dice_ET {et_dice_meter.val:.3f} ({et_dice_meter.avg:.3f}) \t'
                    f'HD95_WT {wt_hd95_meter.val:.3f} ({wt_hd95_meter.avg:.3f}) \t'
                    f'HD95_TC {tc_hd95_meter.val:.3f} ({tc_hd95_meter.avg:.3f}) \t'
                    f'HD95_ET {et_hd95_meter.val:.3f} ({et_hd95_meter.avg:.3f}) \t')

            end = time.time()
    
    # output case metric csv
    case_metric_meter.output(epoch, save_epoch_path)

    return {
        "Dice_WT": wt_dice_meter.avg,
        "Dice_TC": tc_dice_meter.avg,
        "Dice_ET": et_dice_meter.avg,
        "HD95_WT": wt_hd95_meter.avg,
        "HD95_TC": tc_hd95_meter.avg,
        "HD95_ET": et_hd95_meter.avg,
    }


def main():
    args = prepare_train_args()
    tb_logger = SummaryWriter(args.save_path)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # dataloader
    train_cases, val_cases = load_cases_split(args.cases_split)
    train_loader = get_brats2021_train_loader(args, train_cases)
    val_loader = get_brats2021_val_loader(args, val_cases)

    # model & stuff
    model = select_model(args).cuda()
    loss = SoftDiceBCEWithLogitsLoss().cuda()
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    # auto mixed precision
    if args.amp:
        amp.register_float_function(torch, 'sigmoid')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    model = torch.nn.DataParallel(model, device_ids=args.gpus)

    # load model & resume training
    if args.load_model_path != '':
        print(f"==> loading checkpoint ...")
        state = torch.load(args.load_model_path)
        sratr_epoch = state['epoch'] + 1
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if args.amp and 'amp' in state:
            print('=> resuming amp state_dict')
            amp.load_state_dict(state['amp'])
        
        print(f"==> loaded checkpoint from epoch {state['epoch']}")
        del state
        torch.cuda.empty_cache()
    else:
        sratr_epoch = 0

    # train & val
    for epoch in range(sratr_epoch, args.epochs):
        print("==> training...")
        train_tb = train(args, epoch, model, train_loader, loss, optimizer, scheduler)

        for key in train_tb.keys():
            tb_logger.add_scalar(key, train_tb[key], epoch)
        
        if ((epoch + 1) % args.val_save_freq == 0) or (epoch == 1):
            print("==> testing...")

            # make save epoch folder
            save_epoch_path = join(args.save_path, f"epoch_{epoch:02d}")
            if not os.path.exists(save_epoch_path):
                os.system(f"mkdir -p {save_epoch_path}")

            # actual validation & tensorboard logging
            val_tb = val(args, epoch, model, val_loader, save_epoch_path)
            for key in val_tb.keys():
                tb_logger.add_scalar(key, val_tb[key], epoch)

            # saving model & metrics for every case
            print("==> saving...")
            state = {
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            torch.save(state, os.path.join(save_epoch_path, f'checkpoint_{epoch:02d}.pth'))

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
