import torch
import torch.nn as nn
# 定义了 Flow Matching 的 L_2 损失。

class FlowMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 简单的 MSE Loss
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, v_pred, u_t, mask=None):
        """
        v_pred: 预测的速度 v_theta(x_t, t, c) (B, L, D_x)
        u_t: 目标速度 u_t(x_t | x_1) (B, L, D_x)
        mask: 掩码 (B, L, 1)
        """
        loss = self.mse(v_pred, u_t)

        if mask is not None:
            loss = loss * mask
            # 计算带掩码的均值
            return loss.sum() / mask.sum()
        else:
            return loss.mean()