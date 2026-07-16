"""教材使用的轻量模型。"""

from .diffusion import (
    DiffusionTrainingResult,
    TinyNoiseMLP,
    TorchDiffusionSchedule,
    make_torch_schedule,
    q_sample_torch,
    reverse_sample_torch,
    sample_eight_gaussians,
    train_toy_ddpm,
)
from .vae import TinyVAE, load_digits_data, train_digits_vae, vae_loss

__all__ = [
    "DiffusionTrainingResult",
    "TinyNoiseMLP",
    "TinyVAE",
    "TorchDiffusionSchedule",
    "load_digits_data",
    "make_torch_schedule",
    "q_sample_torch",
    "reverse_sample_torch",
    "sample_eight_gaussians",
    "train_digits_vae",
    "train_toy_ddpm",
    "vae_loss",
]
