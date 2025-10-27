import torch
# 定义了从 x_0 到 x_1 的概率路径和目标向量场 u_t
class GaussianConditionalProbabilityPath:
    """
    定义了从 x0 到 x1 的直线路径 (ODE)
    p_t(x_t | x_0, x_1) = (1-t) * x_0 + t * x_1
    对应的目标向量场 u_t = x_1 - x_0
    """
    def __init__(self, sigma=1.0):
        # sigma 在这个简单路径中未使用，但保留接口一致性
        self.sigma = sigma

    def sample(self, t, x0, x1):
        """
        采样 x_t 并返回目标向量场 u_t
        t: (B, L, 1)
        x0, x1: (B, L, D_x)
        """
        # 路径上的点
        x_t = (1 - t) * x0 + t * x1
        # 目标向量场 (速度)
        u_t = x1 - x0
        return x_t, u_t

class ConditionalFlowMatcher:
    def __init__(self, sigma=1.0):
        self.path = GaussianConditionalProbabilityPath(sigma)

    def sample_conditional_path(self, x0, x1, t):
        """
        x0: 先验 (B, L, D_x)
        x1: 数据 (B, L, D_x)
        t: 时间 (B, L, 1)
        """
        return self.path.sample(t, x0, x1)