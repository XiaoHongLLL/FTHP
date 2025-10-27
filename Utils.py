import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer.Constants as Constants
import numpy as np

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, num_classes, ignore_index=-1):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, output, target):
        """
        Output: (batch_size, num_classes)
        Target: (batch_size)
        """
        # (batch_size * seq_len, num_classes)
        output = output.view(-1, self.num_classes)
        # (batch_size * seq_len)
        target = target.view(-1)

        # get the idx of ignored index
        # (non_pad_prob.shape[0])
        non_pad_mask = (target != self.ignore_index)
        # (non_pad_prob.shape[0], num_classes)
        output = output[non_pad_mask]
        # (non_pad_prob.shape[0])
        target = target[non_pad_mask]

        # smooth the target
        # (non_pad_prob.shape[0], num_classes)
        one_hot = F.one_hot(target, self.num_classes)
        # (non_pad_prob.shape[0], num_classes)
        smooth_target = (1.0 - self.label_smoothing) * one_hot + self.label_smoothing / self.num_classes

        # compute the log prob
        log_prob = F.log_softmax(output, dim=1)

        # compute the loss
        loss = (-log_prob * smooth_target).sum(dim=1)
        return loss.mean()

def type_loss(prediction, types, loss_func):
    """
    Event type loss
    :param prediction: (batch_size, seq_len, num_classes)
    :param types: (batch_size, seq_len)
    :param loss_func:
    :return:
    """
    # (batch_size * seq_len, num_classes)
    prediction = prediction[:, 1:, :].contiguous().view(-1, prediction.size(-1))
    # (batch_size * seq_len)
    types = types[:, 1:].contiguous().view(-1)
    # filter out the padding
    # (batch_size * seq_len)
    non_pad_mask = (types != Constants.PAD)
    # (non_pad_mask.shape[0], num_classes)
    prediction = prediction[non_pad_mask]
    # (non_pad_mask.shape[0])
    types = types[non_pad_mask]

    loss = loss_func(prediction, types)
    return loss

def time_loss(prediction, times):
    """
    Time prediction loss
    :param prediction: (batch_size, seq_len)
    :param times: (batch_size, seq_len)
    :return:
    """
    # (batch_size * seq_len)
    prediction = prediction[:, 1:].contiguous().view(-1)
    # (batch_size * seq_len)
    times = times[:, 1:].contiguous().view(-1)

    # filter out the padding
    # (batch_size * seq_len)
    non_pad_mask = (times != Constants.PAD)
    # (non_pad_mask.shape[0])
    prediction = prediction[non_pad_mask]
    # (non_pad_mask.shape[0])
    times = times[non_pad_mask]

    # (non_pad_mask.shape[0])
    loss = (prediction - times) ** 2
    return loss.mean()

def evaluate_samples(t_sample, gt_t, type_sample, gt_type, opt):
    """
    t_sample: (batch_size, n_samples)
    gt_t: (batch_size, 1)
    type_sample: (batch_size, n_samples)
    gt_type: (batch_size, 1)
    opt:
    """

    # 确保 gt_t 和 gt_type 是 (B, 1)
    if gt_t.ndim == 1:
        gt_t = gt_t.unsqueeze(1)
    if gt_type.ndim == 1:
        gt_type = gt_type.unsqueeze(1)

    # (batch_size, n_samples)
    abs_err = (t_sample - gt_t).abs()
    # (batch_size)
    crps = abs_err.mean(1)
    # (batch_size, n_samples)
    coverage = t_sample.quantile(opt.eval_quantile.reshape(1,-1),dim=1)
    # (batch_size, n_quantile)
    coverage = coverage.permute(1,0)
    # (batch_size, n_quantile)
    coverage = (coverage>gt_t)
    # (n_quantile)
    coverage = coverage.float().mean(0)
    # (n_quantile)
    intlen = t_sample.quantile(1-opt.eval_quantile_step/2,dim=1) - t_sample.quantile(opt.eval_quantile_step/2,dim=1)

    # (batch_size, n_samples)
    corr_type = (type_sample == gt_type)
    # (batch_size)
    corr_type = corr_type.float().mean(1)

    return coverage, intlen, crps.sum(), corr_type.sum()