from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math
import torch
import torch.distributions as D
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape ()
        Returns:
            - drift_coefficient: shape (batch_size, dim)
        """
        pass
class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (bs,1)
            - dt: time, shape (bs,1)
        Returns:
            - nxt: state at time t + dt (bs, dim)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (batch_size, dim)
            - ts: timesteps, shape (bs, num_timesteps,1)
        Returns:
            - x_final: final state at time ts[-1], shape (batch_size, dim)
        """
        num_steps = ts.shape[0]
        for t_idx in range(num_steps - 1):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]

            # 扩展 t 和 h 以匹配 x 的批量大小
            t_batch = t.expand(x.shape[0], 1)
            h_batch = h.expand(x.shape[0], 1)

            x = self.step(x, t_batch, h_batch)
        return x
    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (bs, dim)
            - ts: timesteps, shape (bs, num_timesteps, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, num
            _timesteps, dim)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + self.ode.drift_coefficient(xt, t) * h


class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Returns:
            - Dimensionality of the distribution
        """
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, dim)
        """
        pass


class Density(ABC):
    """
    Distribution with tractable density
    """

    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log density at x.
        Args:
            - x: shape (batch_size, dim)
        Returns:
            - log_density: shape (batch_size, 1)
        """
        pass


class Gaussian(torch.nn.Module, Sampleable, Density):
    """
    Multivariate Gaussian distribution
    """

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        mean: shape (dim,)
        cov: shape (dim,dim)
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)


class GaussianMixture(torch.nn.Module, Sampleable, Density):
    """
    Two-dimensional Gaussian mixture model, and is a Density and a Sampleable. Wrapper around torch.distributions.MixtureSameFamily.
    """

    def __init__(
            self,
            means: torch.Tensor,  # nmodes x data_dim
            covs: torch.Tensor,  # nmodes x data_dim x data_dim
            weights: torch.Tensor,  # nmodes
    ):
        """
        means: shape (nmodes, 2)
        covs: shape (nmodes, 2, 2)
        weights: shape (nmodes, 1)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=self.means,
                covariance_matrix=self.covs,
                validate_args=False,
            ),
            validate_args=False,
        )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))


class ConditionalProbabilityPath(torch.nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """

    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple  # 简单初始分布Psimple
        self.p_data = p_data  # 目标数据分布Pdata

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        # Pt(x)随时间 t 从p_simple（t=0）逐渐变化到p_data（t=1）

        num_samples = t.shape[0]  # 样本数量

        # 从Pdata采样条件变量z z~Pdata
        z = self.sample_conditioning_variable(num_samples)  # (num_samples, dim)

        # 从条件分布Pt(x|z)采样x x~Pt(x|z)
        x = self.sample_conditional_path(z, t)  # (num_samples, dim)
        return x  # 从而得到边缘概率路径Pt(X)

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z
        Args:
            - num_samples: the number of samples
        Returns:
            - z: samples from p(z), (num_samples, dim)
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 条件概率路径
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 计算条件向量场u_t(x|z)，即x在t时刻的速度方向
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 计算条件分布p_t(x|z)的分数函数，即log概率的梯度
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_score: conditional score (num_samples, dim)
        """
        pass


# 实现两个关键噪声调度函数

class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        # t=0时，alpha_t必须等于0（初始时刻无数据信号）
        assert torch.allclose(
            self(torch.zeros(1, 1)), torch.zeros(1, 1)
        )
        # Check alpha_1 = 1
        # t=1时，alpha_t必须等于1（最终时刻全是数据信号）
        assert torch.allclose(
            self(torch.ones(1, 1)), torch.ones(1, 1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算alpha_t的值，必须满足：self(0.0)=0.0，self(1.0)=1.0
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算alpha_t对时间t的导数dα_t/dt
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """
        t = t.unsqueeze(1)  # 给t增加一个维度，形状从(样本数,1)→(样本数,1,1)，用于适配jacrev的输入要求
        dt = vmap(jacrev(self))(t)  # 批量计算每个t的导数，jacrev：算导数的函数
        return dt.view(-1, 1)  # 调整形状为(样本数,1)，与输入t的形状一致


class Beta(ABC):
    def __init__(self):
        # t=0时，beta_t必须等于1（初始时刻全是噪声
        assert torch.allclose(
            self(torch.zeros(1, 1)), torch.ones(1, 1)
        )
        # t=1时，beta_t必须等于0（最终时刻无噪声）
        assert torch.allclose(
            self(torch.ones(1, 1)), torch.zeros(1, 1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt beta_t (num_samples, 1)
        """
        t = t.unsqueeze(1)  # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t)  # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)


class LinearAlpha(Alpha):
    """
    实现 alpha_t = t
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算α_t的值，直接返回输入的时间t（因为α_t=t）
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算α_t的导数dα_t/dt
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """
        return torch.ones_like(t)  # 返回和t形状相同的全1张量，因为α_t=t的导数是1


class SquareRootBeta(Beta):
    """
    Implements beta_t = rt(1-t)
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算β_t的值，返回sqrt(1-t)
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """
        return torch.sqrt(1 - t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算β_t的导数dβ_t/dt
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """
        return - 0.5 / (torch.sqrt(1 - t) + 1e-4)  # 1e-4是“数值稳定性项”，避免t接近1时分母sqrt(1-t)趋近于0导致报错


# 实现 Pt(x|z)=N(α_t*z,β_t^2*Id)
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    # 高斯条件概率路径的起点是简单高斯噪声P0(x)=N(0,Id)（即 Psimple），终点是数据分布Pdata(x)，通过 α_t 和 β_t 连接两者。
    def __init__(self, event_dim: int, alpha: Alpha, beta: Beta):
        # 定义初始简单分布Psimple：与Pdata同维度的标准高斯噪声（N(0,I_d)）
        p_simple = Gaussian.isotropic(event_dim, 1.0)
        # 注意：在条件FM中，p_data是隐式的p(z|c)，我们没有一个简单的Sampleable对象
        # 我们可以将p_data设为None，因为我们不会在训练器中直接调用p_data.sample()
        super().__init__(p_simple, p_data=None)
        self.alpha = alpha
        self.beta = beta
        self._dim = event_dim # 存储维度信息

    @property
    def dim(self) -> int:
        return self._dim


    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        从Pt(x|z)=N(α_t z, β_t² I_d)采样x
        Samples from the conditional distribution p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """
        # 计算推动x演化的向量场u_t(x|z)（流模型核心）#
        # 计算当前时间t的α_t、β_t及其导数
        alpha_t = self.alpha(t)  # α_t的值
        beta_t = self.beta(t)  # β_t的值
        dt_alpha_t = self.alpha.dt(t)  # dα_t/dt
        dt_beta_t = self.beta.dt(t)  # dβ_t/dt
        # 向量场公式：u_t(x|z) = (dα_t/dt - (dβ_t/dt/β_t)α_t)z + (dβ_t/dt/β_t)x
        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_score: conditional score (num_samples, dim)
        """
        # 计算logp_t(x|z)的分数函数
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)

        return (z * alpha_t - x) / beta_t ** 2
#定义条件向量场类
class ConditionalVectorFieldODE(ODE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.path = path    # 保存概率路径
        self.z = z          # 保存固定的条件变量z

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z)
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        bs = x.shape[0]  # 获取批量大小（一次处理多少个样本点）
        # 扩展z的形状：从(1, dim)→(bs, dim)（每个样本点共享同一个z）
        z = self.z.expand(bs, *self.z.shape[1:])
        # 调用概率路径的条件向量场计算方法，返回漂移系数
        return self.path.conditional_vector_field(x,z,t)


class LearnedConditionalODE(ODE):
    """
    使用学习到的条件向量场 v_theta(x, t, c) 进行推理的 ODE
    """

    def __init__(self, model: torch.nn.Module, c: torch.Tensor):
        """
        Args:
        - model: 学习到的 v_theta(x, t, c)
        - c: 固定的条件变量, (batch_size, d_model)
        """
        super().__init__()
        self.model = model  # 这是你的 ConditionalMLPVectorField
        self.c = c  # 这是来自 THP 的历史上下文

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        返回 v_theta(x, t, c)
        Args:
            - x: state at time t, shape (bs, event_dim)
            - t: time, shape (bs, 1) or (bs,)
        Returns:
            - v_theta(x, t, c): shape (batch_size, event_dim)
        """
        # 确保 t 的形状正确
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # model 需要 (x, t, c)
        return self.model(x, t, self.c)

class Trainer(ABC):

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model  # 待训练的模型（此处为MLPVectorField）

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        #计算训练损失
        pass

    def get_optimizer(self, lr: float):
        # 获取优化器Adam
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3,** kwargs) -> torch.Tensor:
        # 准备训练：模型移到设备、初始化优化器、设置训练模式
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # 训练循环（带进度条）
        pbar = tqdm(enumerate(range(num_epochs)))  # tqdm用于显示训练进度
        for idx, epoch in pbar:
            opt.zero_grad()  # 清空梯度
            loss = self.get_train_loss(**kwargs)  # 计算损失（子类实现）
            loss.backward()  # 反向传播计算梯度
            opt.step()  # 更新模型参数
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')  # 显示当前损失

        # 训练结束：设置为评估模式
        self.model.eval()


class ConditionalFlowMatchingTrainer(Trainer):
    """
    用于条件流匹配 (Conditional Flow Matching) 的训练器
    """

    def __init__(self, model: torch.nn.Module, path: GaussianConditionalProbabilityPath):
        """
        Args:
            - model: 神经网络 v_theta(x, t, c)
            - path: 定义 p_t(x|z) 和 u_t(x|z) 的概率路径
        """
        super().__init__(model)
        self.path = path

    def get_train_loss(self, c: torch.Tensor, z_target: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算条件流匹配损失
        Args:
            - c: 条件 (来自 THP), shape (batch_size, d_model)
            - z_target: 目标 z (来自 EventMapper), shape (batch_size, event_dim)
        Returns:
            - loss: 标量损失
        """
        num_samples = z_target.shape[0]

        # 1. 采样时间 t ~ U[0, 1]
        # (为了数值稳定性，避免 t=1 时 beta_t=0 导致除零)
        t = torch.rand(num_samples, 1, device=z_target.device) * (1 - 1e-4)

        # 2. 采样 x_t ~ p_t(x|z_target) = N(alpha_t * z, beta_t^2 * I)
        x_t = self.path.sample_conditional_path(z_target, t)

        # 3. 计算目标向量场 u_t(x_t|z_target)
        u_t = self.path.conditional_vector_field(x_t, z_target, t)

        # 4. 计算模型预测 v_theta(x_t, t, c)
        v_t = self.model(x_t, t, c)

        # 5. 计算损失 L = ||v_t - u_t||^2
        loss = torch.mean((v_t - u_t) ** 2)
        return loss

class SinusoidalPosEmb(nn.Module):
    """
    时间 t 的正弦位置编码
    """

    def __init__(self, dim):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension dim ({dim}) must be even.")
        self.dim = dim

    def forward(self, t):
        # t 原始 shape: (batch_size, 1) or (batch_size, )
        t = t.view(-1, 1)  # 确保 t 是 (batch_size, 1)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ==========================================================
# 关键新模块: 事件 <-> 向量 z 的转换器
# ==========================================================

class EventTargetMapper(nn.Module):
    """
    将 (time_gap, event_type) 编码为连续向量 z，并可解码。
    这是 THP 和 FM 之间的关键桥梁。
    """

    def __init__(self, event_dim: int, num_types: int, time_emb_dim: int = 16):
        super().__init__()

        if time_emb_dim >= event_dim:
            raise ValueError("event_dim must be larger than time_emb_dim")

        self.time_emb_dim = time_emb_dim
        self.type_emb_dim = event_dim - time_emb_dim
        self.event_dim = event_dim
        self.num_types = num_types

        # 1. 时间 (time_gap) 编码器
        # 我们使用一个小的 SinusoidalPosEmb
        self.time_encoder = SinusoidalPosEmb(time_emb_dim)

        # 2. 类型 (event_type) 编码器
        # event_type 从 1 开始 (0 是 PAD)，所以 num_types + 1
        self.type_embed = nn.Embedding(num_types + 1, self.type_emb_dim, padding_idx=0)

        # 3. 解码器头部
        # 将 z_time 解码回标量 time_gap (log space)
        self.time_decoder = nn.Sequential(
            nn.Linear(time_emb_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        # 类型的解码器是 type_embed 的权重矩阵转置 (在 decode 方法中实现)

    def encode(self, time_gap: torch.Tensor, event_type: torch.Tensor) -> torch.Tensor:
        """
        编码: (time_gap, event_type) -> z
        Args:
            - time_gap: (batch_size,) - 真实的 time_gap
            - event_type: (batch_size,) - 真实的 event_type (1-based)
        Returns:
            - z: (batch_size, event_dim)
        """
        # (重要) 对 time_gap 进行 log 变换以稳定尺度
        time_gap_log = torch.log1p(time_gap)

        # (batch_size, time_emb_dim)
        time_emb = self.time_encoder(time_gap_log)

        # (batch_size, type_emb_dim)
        type_emb = self.type_embed(event_type)

        # (batch_size, event_dim)
        z = torch.cat([time_emb, type_emb], dim=-1)
        return z

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解码: z -> (time_gap_pred, event_type_logits)
        Args:
            - z: (batch_size, event_dim)
        Returns:
            - time_gap_pred: (batch_size,)
            - type_logits: (batch_size, num_types)
        """
        # 1. 拆分 z
        # (batch_size, time_emb_dim)
        z_time = z[..., :self.time_emb_dim]
        # (batch_size, type_emb_dim)
        z_type = z[..., self.time_emb_dim:]

        # 2. 解码时间
        # (batch_size, 1)
        time_gap_log_pred = self.time_decoder(z_time)
        # (重要) 逆变换
        time_gap_pred = torch.expm1(time_gap_log_pred.squeeze(-1))
        # 防止负时间
        time_gap_pred = F.relu(time_gap_pred)

        # 3. 解码类型
        # (batch_size, type_emb_dim) @ (type_emb_dim, num_types + 1)
        # 我们只关心 1...num_types
        type_weights = self.type_embed.weight[1:, :].t()  # (type_emb_dim, num_types)
        type_logits = torch.matmul(z_type, type_weights)  # (batch_size, num_types)

        return time_gap_pred, type_logits


# ==========================================================
# 关键新模块: 条件向量场 v(x, t, c)
# ==========================================================

class ConditionalMLPVectorField(nn.Module):
    """
    一个条件MLP，用于学习向量场 v(x, t, c)

    输入:
        - x: 状态 (下一个事件 z), shape (batch_size, event_dim)
        - t: 时间, shape (batch_size, 1)
        - c: 条件 (THP历史编码), shape (batch_size, d_model)
    输出:
        - v: 预测的向量场, shape (batch_size, event_dim)
    """

    def __init__(self, event_dim: int, d_model: int, time_emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.event_dim = event_dim
        self.d_model = d_model

        # 1. 流的时间编码 (t \in [0, 1])
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # 2. 主网络
        # 输入维度: event_dim (来自x) + time_emb_dim (来自t) + d_model (来自c)
        input_dim = event_dim + time_emb_dim + d_model

        self.main_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, event_dim)  # 输出维度必须是 event_dim
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # t 可能是 (bs,) or (bs, 1), 确保 (bs, 1)
        t = t.view(-1, 1)

        t_emb = self.time_mlp(t)

        # 拼接 x, t_emb, 和 c
        xtc_emb = torch.cat([x, t_emb, c], dim=1)

        return self.main_mlp(xtc_emb)