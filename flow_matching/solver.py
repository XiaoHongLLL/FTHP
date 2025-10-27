# 该文件内容基于 facebookresearch/flow_matching/flow_matching/solver/ode_solver.py
# 需要: pip install torchdiffeq

import torch
from torchdiffeq import odeint
from typing import Callable, Optional, Sequence, Union
# 用于逆向生成采样的 ODE 求解器

class ODESolver:
    """
    使用 torchdiffeq 包装 ODE 求解器
    """

    def __init__(self, velocity_model: Callable):
        """
        velocity_model: 一个可调用对象, 接受 (x, t, **model_extras)
        """
        super().__init__()
        self.velocity_model = velocity_model

    def sample(
            self,
            x_init: torch.Tensor,
            time_grid: torch.Tensor,
            step_size: Optional[float] = None,
            method: str = "euler",
            atol: float = 1e-5,
            rtol: float = 1e-5,
            enable_grad: bool = False,
            **model_extras,
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        x_init: 初始点 (如 x_0)
        time_grid: 求解的时间点 (如 [0.0, 1.0])
        model_extras: 传递给 velocity_model 的额外参数 (如条件 c)
        """
        time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):
            # model_extras 包含条件 c
            # t 是标量, x 是 (B, ..., D_x)
            return self.velocity_model(x=x, t=t, **model_extras)

        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            sol = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        # sol 是 (Time, B, ..., D_x)
        # 返回最后一个时间点 (t=1) 的样本
        return sol[-1]