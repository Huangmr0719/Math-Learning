"""第 18 章使用的离线二维最小 DDPM。

模型只学习由程序生成的八团二维数据，不下载数据，也不追求图像质量。
它的目的，是让 forward noising、epsilon prediction 与 reverse sampling
在几秒到几十秒的轻量实验中形成闭环。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TorchDiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_previous: torch.Tensor
    posterior_variance: torch.Tensor

    @property
    def steps(self) -> int:
        return int(self.betas.shape[0])


def make_torch_schedule(
    steps: int = 80,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 8e-2,
    device: torch.device | str = "cpu",
) -> TorchDiffusionSchedule:
    if steps < 2:
        raise ValueError("steps 至少为 2。")
    betas = torch.linspace(beta_start, beta_end, steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_previous = torch.cat(
        [torch.ones(1, device=device), alpha_bars[:-1]]
    )
    posterior_variance = (
        betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
    )
    return TorchDiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_previous=alpha_bars_previous,
        posterior_variance=posterior_variance,
    )


def _extract(values: torch.Tensor, t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """取出每个 batch 样本自己的时间系数并扩成可广播 shape。"""

    selected = values.gather(0, t)
    return selected.reshape(t.shape[0], *([1] * (target.ndim - 1)))


def q_sample_torch(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: TorchDiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对应 x_t = sqrt(alpha_bar_t)x_0 + sqrt(1-alpha_bar_t)epsilon。"""

    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_bar = _extract(schedule.alpha_bars.sqrt(), t, x0)
    sqrt_one_minus = _extract((1.0 - schedule.alpha_bars).sqrt(), t, x0)
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise


def sample_eight_gaussians(
    count: int,
    *,
    seed: int = 7,
    radius: float = 2.0,
    cluster_std: float = 0.08,
) -> torch.Tensor:
    """生成八个圆周 Gaussian 团，shape 为 [count, 2]。"""

    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 8, (count,), generator=generator)
    angles = labels.float() * (2.0 * math.pi / 8.0)
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    noise = torch.randn((count, 2), generator=generator) * cluster_std
    return centers + noise


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int = 32):
        super().__init__()
        if dimension % 2 != 0:
            raise ValueError("time embedding dimension 必须为偶数。")
        self.dimension = dimension

    def forward(self, t: torch.Tensor, total_steps: int) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        # 原始整数时间乘不同频率。低频描述缓慢变化，高频区分相邻时间。
        angles = t.float()[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class TinyNoiseMLP(nn.Module):
    def __init__(self, hidden_dim: int = 96, time_dim: int = 32):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.network = nn.Sequential(
            nn.Linear(2 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, xt: torch.Tensor, t: torch.Tensor, total_steps: int) -> torch.Tensor:
        embedded_t = self.time_embedding(t, total_steps)
        return self.network(torch.cat([xt, embedded_t], dim=1))


@dataclass
class DiffusionTrainingResult:
    model: TinyNoiseMLP
    schedule: TorchDiffusionSchedule
    losses: list[float]
    data: torch.Tensor
    samples: torch.Tensor
    trajectory: dict[int, torch.Tensor]
    device: str


def reverse_sample_torch(
    model: TinyNoiseMLP,
    schedule: TorchDiffusionSchedule,
    *,
    count: int = 1000,
    seed: int = 17,
    record_steps: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    device = schedule.betas.device
    torch.manual_seed(seed)
    x = torch.randn((count, 2), device=device)
    trajectory: dict[int, torch.Tensor] = {}
    if record_steps is None:
        record_steps = tuple(
            sorted(
                {
                    schedule.steps - 1,
                    3 * schedule.steps // 4,
                    schedule.steps // 2,
                    schedule.steps // 4,
                    0,
                },
                reverse=True,
            )
        )
    model.eval()

    with torch.no_grad():
        for step in reversed(range(schedule.steps)):
            t = torch.full((count,), step, device=device, dtype=torch.long)
            predicted_noise = model(x, t, schedule.steps)

            beta_t = schedule.betas[step]
            alpha_t = schedule.alphas[step]
            alpha_bar_t = schedule.alpha_bars[step]
            mean = (
                x - beta_t * predicted_noise / torch.sqrt(1.0 - alpha_bar_t)
            ) / torch.sqrt(alpha_t)

            # t=0 时已经到达 x_0，不再注入随机噪声。
            if step > 0:
                x = mean + torch.sqrt(schedule.posterior_variance[step]) * torch.randn_like(x)
            else:
                x = mean

            if step in record_steps:
                trajectory[step] = x.detach().cpu()

    return x.detach().cpu(), trajectory


def train_toy_ddpm(
    *,
    training_steps: int = 600,
    diffusion_steps: int = 80,
    batch_size: int = 256,
    learning_rate: float = 2e-3,
    seed: int = 7,
    device: str | None = None,
) -> DiffusionTrainingResult:
    if training_steps < 1:
        raise ValueError("training_steps 至少为 1。")

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = torch.device(device)

    data_cpu = sample_eight_gaussians(4096, seed=seed)
    data = data_cpu.to(torch_device)
    schedule = make_torch_schedule(diffusion_steps, device=torch_device)
    model = TinyNoiseMLP().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []

    for _ in range(training_steps):
        indices = torch.randint(0, data.shape[0], (batch_size,), device=torch_device)
        x0 = data[indices]
        t = torch.randint(0, schedule.steps, (batch_size,), device=torch_device)
        xt, true_noise = q_sample_torch(x0, t, schedule)
        predicted_noise = model(xt, t, schedule.steps)

        # 对应 L_simple = E ||epsilon - epsilon_theta(x_t,t)||^2。
        # F.mse_loss 默认对 batch 和二维 feature 同时取平均，返回标量。
        loss = F.mse_loss(predicted_noise, true_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    samples, trajectory = reverse_sample_torch(
        model,
        schedule,
        count=1200,
        record_steps=tuple(
            sorted(
                {
                    schedule.steps - 1,
                    3 * schedule.steps // 4,
                    schedule.steps // 2,
                    schedule.steps // 4,
                    0,
                },
                reverse=True,
            )
        ),
    )
    return DiffusionTrainingResult(
        model=model,
        schedule=schedule,
        losses=losses,
        data=data_cpu,
        samples=samples,
        trajectory=trajectory,
        device=str(torch_device),
    )
