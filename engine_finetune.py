# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, pdb
import math
from typing import Iterable, Optional, Dict, Tuple

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from utils import adjust_learning_rate, NativeScalerWithGradNormCount as NativeScaler
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score, 
    accuracy_score,
    log_loss,
    roc_auc_score,
    precision_recall_curve,
    auc,
)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    lr_schedule: list, num_training_steps_per_epoch: int,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    update_freq = args.update_freq if args is not None else 1
    use_amp = args.use_amp if args is not None else False
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq=100, header=header)):

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue

        # we use a per iteration (instead of per epoch) lr scheduler
        if step % update_freq == 0:
            # Assign learning rate
            global_step = epoch * num_training_steps_per_epoch + step
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[global_step]
        
        # Handle tuple of images
        if isinstance(samples, (list, tuple)):
            # The custom model architecture with dual inputs does not support mixup
            if mixup_fn is not None:
                raise ValueError("Mixup is not supported for models with dual-image inputs.")
            samples = [s.to(device, non_blocking=True) for s in samples]
        else:
        samples = samples.to(device, non_blocking=True)

        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None and not isinstance(samples, list):
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        
        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None and (data_iter_step + 1) % update_freq == 0:
            log_writer.update(loss=loss_value, head="train")
            if class_acc is not None:
                log_writer.update(class_acc=class_acc, head="train")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, val=None, use_amp=False) -> Tuple[Dict[str, float], float, float]:
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    all_targets = []
    all_probs = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        # Handle tuple of images
        if isinstance(images, (list, tuple)):
            images_for_size = images[0]
            images = [img.to(device, non_blocking=True) for img in images]
        else:
            images_for_size = images
        images = images.to(device, non_blocking=True)
            
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(images)
            loss = criterion(output, target)
        
        all_probs.append(output)
        all_targets.append(target)

        torch.cuda.synchronize()

        acc1, _ = accuracy(output, target, topk=(1, 2))

        batch_size = images_for_size.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item() / 100.0, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.2%} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    output_ddp = [torch.zeros_like(all_probs) for _ in range(utils.get_world_size())]
    dist.all_gather(output_ddp, all_probs)
    labels_ddp = [torch.zeros_like(all_targets) for _ in range(utils.get_world_size())]
    dist.all_gather(labels_ddp, all_targets)

    output_all = torch.cat(output_ddp, dim=0)
    labels_all = torch.cat(labels_ddp, dim=0)

    y_pred = softmax(output_all.detach().cpu().numpy(), axis=1)[:, 1]
    y_true = labels_all.detach().cpu().numpy()
    y_true = y_true.astype(int)
  
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, float(acc), float(ap)