# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from lib.utils import Timer, AverageMeter, precision_at_one, fast_hist, per_class_iu, \
    get_prediction, get_torch_device

import MinkowskiEngine as ME


def print_info(iteration,
               max_iteration,
               data_time,
               iter_time,
               has_gt=False,
               losses=None,
               scores=None,
               ious=None,
               hist=None,
               ap_class=None,
               class_names=None):
  debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
  debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

  if has_gt:
    acc = hist.diagonal() / hist.sum(1) * 100
    debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
        "Score {top1.val:.3f} (AVG: {top1.avg:.3f})\t" \
        "mIOU {mIOU:.3f} mAP {mAP:.3f} mAcc {mAcc:.3f}\n".format(
            loss=losses, top1=scores, mIOU=np.nanmean(ious),
            mAP=np.nanmean(ap_class), mAcc=np.nanmean(acc))
    if class_names is not None:
      debug_str += "\nClasses: " + " ".join(class_names) + '\n'
    debug_str += 'IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n'
    debug_str += 'mAP: ' + ' '.join('{:.03f}'.format(i) for i in ap_class) + '\n'
    debug_str += 'mAcc: ' + ' '.join('{:.03f}'.format(i) for i in acc) + '\n'

  logging.info(debug_str)


def average_precision(prob_np, target_np):
  num_class = prob_np.shape[1]
  label = label_binarize(target_np, classes=list(range(num_class)))
  with np.errstate(divide='ignore', invalid='ignore'):
    return average_precision_score(label, prob_np, None)


def test(model, data_loader, config, transform_data_fn=None, has_gt=True):
  device = get_torch_device(config.is_cuda)
  dataset = data_loader.dataset
  num_labels = dataset.NUM_LABELS
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
  criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)
  losses, scores, ious = AverageMeter(), AverageMeter(), 0
  aps = np.zeros((0, num_labels))
  hist = np.zeros((num_labels, num_labels))

  logging.info('===> Start testing')

  global_timer.tic()
  data_iter = data_loader.__iter__()
  max_iter = len(data_loader)
  max_iter_unique = max_iter

  # Fix batch normalization running mean and std
  model.eval()

  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()

  with torch.no_grad():
    for iteration in range(max_iter):
      data_timer.tic()
      if config.return_transformation:
        coords, input, target, pointcloud, transformation = data_iter.next()
      else:
        coords, input, target = data_iter.next()
      data_time = data_timer.toc(False)

      # Preprocess input
      iter_timer.tic()

      if config.normalize_color:
        input[:, :3] = input[:, :3] / 255. - 0.5
      sinput = ME.SparseTensor(input, coords).to(device)

      # Feed forward
      inputs = (sinput,)
      soutput = model(*inputs)
      output = soutput.F

      pred = get_prediction(dataset, output, target).int()
      iter_time = iter_timer.toc(False)

      if has_gt:
        target_np = target.numpy()
        num_sample = target_np.shape[0]
        target = target.to(device)

        cross_ent = criterion(output, target.long())
        losses.update(float(cross_ent), num_sample)
        scores.update(precision_at_one(pred, target), num_sample)
        hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels)
        ious = per_class_iu(hist) * 100

        prob = torch.nn.functional.softmax(output, dim=1)
        ap = average_precision(prob.cpu().detach().numpy(), target_np)
        aps = np.vstack((aps, ap))
        # Due to heavy bias in class, there exists class with no test label at all
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=RuntimeWarning)
          ap_class = np.nanmean(aps, 0) * 100.

      if iteration % config.test_stat_freq == 0 and iteration > 0:
        reordered_ious = dataset.reorder_result(ious)
        reordered_ap_class = dataset.reorder_result(ap_class)
        class_names = dataset.get_classnames()
        print_info(
            iteration,
            max_iter_unique,
            data_time,
            iter_time,
            has_gt,
            losses,
            scores,
            reordered_ious,
            hist,
            reordered_ap_class,
            class_names=class_names)

      if iteration % config.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

  global_time = global_timer.toc(False)

  reordered_ious = dataset.reorder_result(ious)
  reordered_ap_class = dataset.reorder_result(ap_class)
  class_names = dataset.get_classnames()
  print_info(
      iteration,
      max_iter_unique,
      data_time,
      iter_time,
      has_gt,
      losses,
      scores,
      reordered_ious,
      hist,
      reordered_ap_class,
      class_names=class_names)

  logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

  # Explicit memory cleanup
  if hasattr(data_iter, 'cleanup'):
    data_iter.cleanup()

  return losses.avg, scores.avg, np.nanmean(ap_class), np.nanmean(per_class_iu(hist)) * 100
