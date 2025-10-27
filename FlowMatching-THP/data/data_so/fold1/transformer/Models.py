import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import Encoder, get_non_pad_mask
# 引入我们定义
from flow_matching.path import ConditionalFlowMatcher
from flow_matching.loss import FlowMatchingLoss


class VectorField(nn.Module):
    """
    用于预测速度 v_theta(x_t, t, c) 的 MLP。
    """

    def __init__(self, d_in, d_cond, d_hid, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_cond = d_cond
        # 输入: (x_t, t, c)
        self.layer_1 = nn.Linear(d_in + 1 + d_cond, d_hid)
        self.layer_2 = nn.Linear(d_hid, d_hid)
        self.layer_3 = nn.Linear(d_hid, d_out)
        self.relu = nn.ReLU()

    def forward(self, x, t, c):
        """
        x: (B, L, D_in)  (x_t)
        t: (B, L, 1) or (B,) or scalar (flow time)
        c: (B, L, D_cond) (condition)
        """

        # 确保 t 的维度正确 (B, L, 1)
        if t.dim() == 0:
            t = t.view(1, 1, 1).expand(x.shape[0], x.shape[1], 1)
        elif t.dim() == 1:  # (B,)
            t = t.view(x.shape[0], 1, 1).expand(x.shape[0], x.shape[1], 1)
        elif t.dim() == 2 and t.shape[1] == 1:  # (B, 1)
            t = t.expand(x.shape[0], x.shape[1], 1)

        inp = torch.cat([x, t, c], dim=-1)
        h = self.relu(self.layer_1(inp))
        h = self.relu(self.layer_2(h))
        out = self.layer_3(h)
        return out


class FlowMatchingTHP(nn.Module):
    """
    Flow Matching Temporal Hawkes Process
    """

    def __init__(self, num_types, config):
        super().__init__()

        # 1. 历史编码器 (同 SMURF/THP)
        self.encoder = Encoder(
            num_types=num_types,
            d_model=config.d_model,
            d_inner=config.d_inner_hid,
            n_layers=config.n_layers,
            n_head=config.n_head,
            d_k=config.d_k,
            d_v=config.d_v,
            dropout=config.dropout,
        )

        self.num_types = num_types
        self.d_model = config.d_model
        self.config = config

        # 2. IBNN GMM 潜空间
        self.latent_dim = config.d_latent  # GMM 潜变量维度
        # GMM 均值 $\mu_m$ (不包括 padding 和 eos)
        self.gmm_means = nn.Embedding(num_types, self.latent_dim)
        # 假设所有类型共享对角方差
        self.gmm_log_var = nn.Parameter(torch.zeros(self.latent_dim))

        # 目标维度 (时间间隔 + 潜变量)
        self.x_dim = 1 + self.latent_dim

        # 3. Flow Matching 向量场 v_theta
        self.v_field = VectorField(
            d_in=self.x_dim,
            d_cond=self.d_model,
            d_hid=config.d_inner_hid,
            d_out=self.x_dim
        )

        # 4. Flow Matcher & Loss
        self.flow_matcher = ConditionalFlowMatcher(sigma=config.fm_sigma)
        self.fm_loss_func = FlowMatchingLoss()

        # 5. SMURF: 归一化参数 (由 main.py 注入)
        self.normalize = config.normalize
        self.time_mean = 1.0
        self.time_std = 1.0

        # 6. 辅助的类型预测器 (用于辅助 GMM 均值学习)
        self.type_predictor_head = nn.Linear(config.d_model, num_types)
        # (padding=-1, 类型 0~N-1)
        self.type_loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_lambda = config.loss_lambda  # FM loss 和 Type loss 的权重

    def normalize_time(self, time_gap):
        # 避免 log(0)
        time_gap = time_gap.clamp(min=1e-8)

        if self.normalize == 'normal':
            return time_gap / self.time_mean
        elif self.normalize == 'log':
            return (torch.log(time_gap) - self.time_mean) / self.time_std
        else:  # 'None'
            return time_gap

    def denormalize_time(self, time_gap_norm):
        if self.normalize == 'normal':
            return time_gap_norm * self.time_mean
        elif self.normalize == 'log':
            # 确保输出为正
            return torch.exp(time_gap_norm * self.time_std + self.time_mean)
        else:  # 'None'
            return time_gap_norm

    def sample_gmm(self, event_type_idx):
        """ 从 GMM 采样 z_m (用于 x_1) """
        # event_type_idx: (B, L)
        means = self.gmm_means(event_type_idx)  # (B, L, D_latent)
        std = torch.exp(0.5 * self.gmm_log_var)
        z = means + std * torch.randn_like(means)
        return z, means

    def forward(self, event_type, event_time, time_gap):
        """
        event_type: (B, L)
        event_time: (B, L)
        time_gap: (B, L-1) (t_i - t_{i-1})
        """
        non_pad_mask = get_non_pad_mask(event_type)  # (B, L, 1)

        # 1. 编码历史
        enc_output = self.encoder(event_type, event_time, non_pad_mask)  # (B, L, D_model)

        # 历史 c_{i} 用于预测 x_{i+1}
        c = enc_output[:, :-1, :]  # (B, L-1, D_model)

        # 准备 Flow Matching 目标 x1 = (time_norm, z_m)

        # 2. 目标时间 (归一化)
        target_time_gap = time_gap  # (B, L-1)
        target_time_norm = self.normalize_time(target_time_gap).unsqueeze(-1)  # (B, L-1, 1)

        # 3. 目标类型 (GMM 潜变量)
        # (B, L-1), 假设 0=PAD, 1~N 为类型
        target_type_idx = event_type[:, 1:] - 1
        # 掩码掉 PAD (使用索引 0 作为 PAD 类型的均值)
        target_type_idx[target_type_idx < 0] = 0

        z_m, mu_m = self.sample_gmm(target_type_idx)  # (B, L-1, D_latent)

        x_1 = torch.cat([target_time_norm, z_m], dim=-1)  # (B, L-1, D_x)

        # 4. 准备先验 x0
        x_0 = torch.randn_like(x_1)  # (B, L-1, D_x)

        # 5. 采样时间 t
        t = torch.rand(x_1.shape[0], x_1.shape[1], 1, device=x_1.device)  # (B, L-1, 1)

        # 6. 计算路径 x_t 和目标速度 u_t
        x_t, u_t = self.flow_matcher.sample_conditional_path(x_0, x_1, t)

        # 7. 预测速度 v_theta
        v_pred = self.v_field(x_t, t, c)  # (B, L-1, D_x)

        # 8. (可选) 辅助类型预测
        type_logits = self.type_predictor_head(c)  # (B, L-1, NumTypes)

        # 传递给 compute_loss
        prediction = {'v_pred': v_pred, 'u_t': u_t, 'type_logits': type_logits}

        return enc_output, prediction

    def compute_loss(self, enc_out, event_time, time_gap, event_type, prediction, pred_loss_func):
        v_pred = prediction['v_pred']
        u_t = prediction['u_t']
        type_logits = prediction['type_logits']

        # 准备掩码 (B, L-1)
        mask = (event_type[:, 1:] != Constants.PAD).float()

        # 1. Flow Matching Loss
        mask_fm = mask.unsqueeze(-1)  # (B, L-1, 1)
        fm_loss = self.fm_loss_func(v_pred, u_t, mask_fm)

        # 2. 辅助类型 Loss (GMM 均值学习)
        target_type = event_type[:, 1:] - 1  # (B, L-1)
        target_type[target_type < 0] = -1  # 设置为 ignore_index

        type_loss = self.type_loss_func(
            type_logits.view(-1, self.num_types),
            target_type.view(-1)
        )

        total_loss = fm_loss + self.loss_lambda * type_loss

        return total_loss