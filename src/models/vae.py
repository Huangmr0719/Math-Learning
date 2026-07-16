"""第 7 章使用的离线小型 VAE。

使用 scikit-learn 自带的 8×8 digits 数据，不需要联网下载。模型和训练器
保持刻意简洁，便于把每一步与 VAE 公式对应起来。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class TinyVAE(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, latent_dim: int = 2):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.encoder(x))
        return self.mu_head(hidden), self.logvar_head(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # logvar = log(sigma^2)，所以 0.5 * logvar = log(sigma)。
        std = torch.exp(0.5 * logvar)
        # epsilon 与 std shape 相同：[batch_size, latent_dim]。
        epsilon = torch.randn_like(std)
        # 对应 z = mu + sigma ⊙ epsilon；乘法是逐元素乘法。
        return mu + std * epsilon

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.decoder_hidden(z))
        return torch.sigmoid(self.decoder_output(hidden))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # reduction="sum" 先对像素和 batch 求和，随后除以 batch_size，
    # 因而三个返回值都表示“每个样本的平均损失”。
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction, target, reduction="sum"
    ) / target.shape[0]
    kl_loss = (
        -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
        / target.shape[0]
    )
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss


def load_digits_data() -> tuple[torch.Tensor, torch.Tensor]:
    digits = load_digits()
    images = torch.tensor(digits.data / 16.0, dtype=torch.float32)
    labels = torch.tensor(digits.target, dtype=torch.long)
    return images, labels


@dataclass
class TrainingResult:
    model: TinyVAE
    history: dict[str, list[float]]
    images: torch.Tensor
    labels: torch.Tensor


def train_digits_vae(
    *,
    epochs: int = 12,
    latent_dim: int = 2,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 7,
) -> TrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    images, labels = load_digits_data()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = TinyVAE(latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"total": [], "reconstruction": [], "kl": []}

    for _epoch in range(epochs):
        epoch_total = 0.0
        epoch_reconstruction = 0.0
        epoch_kl = 0.0
        batch_count = 0
        for batch_images, _batch_labels in loader:
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(batch_images)
            total, reconstruction_term, kl_term = vae_loss(
                reconstruction, batch_images, mu, logvar
            )
            total.backward()
            optimizer.step()

            epoch_total += float(total.detach())
            epoch_reconstruction += float(reconstruction_term.detach())
            epoch_kl += float(kl_term.detach())
            batch_count += 1

        history["total"].append(epoch_total / batch_count)
        history["reconstruction"].append(epoch_reconstruction / batch_count)
        history["kl"].append(epoch_kl / batch_count)

    return TrainingResult(model=model, history=history, images=images, labels=labels)

