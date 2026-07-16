import numpy as np
import torch

from src.math_checks import (
    ddpm_mean_from_noise,
    ddpm_posterior_mean_variance,
    diffusion_coefficients,
    isotropic_gaussian_kl_same_variance,
    q_sample_from_x0,
)
from src.models.diffusion import make_torch_schedule, q_sample_torch


def test_forward_marginal_matches_theoretical_moments():
    betas = np.linspace(1e-4, 0.02, 20)
    _, alpha_bars = diffusion_coefficients(betas)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(100_000)
    samples = q_sample_from_x0(
        np.full_like(noise, 2.0),
        float(alpha_bars[9]),
        noise,
    )
    assert np.isclose(samples.mean(), np.sqrt(alpha_bars[9]) * 2.0, atol=0.01)
    assert np.isclose(samples.var(), 1.0 - alpha_bars[9], atol=0.01)


def test_posterior_mean_equals_exact_noise_parameterization():
    betas = np.linspace(1e-4, 0.02, 20)
    alphas, alpha_bars = diffusion_coefficients(betas)
    index = 8
    x0 = np.array([1.2, -0.7])
    epsilon = np.array([0.3, -1.1])
    xt = q_sample_from_x0(x0, float(alpha_bars[index]), epsilon)
    posterior_mean, _ = ddpm_posterior_mean_variance(
        x0,
        xt,
        alpha_t=float(alphas[index]),
        alpha_bar_t=float(alpha_bars[index]),
        alpha_bar_previous=float(alpha_bars[index - 1]),
        beta_t=float(betas[index]),
    )
    noise_mean = ddpm_mean_from_noise(
        xt,
        epsilon,
        alpha_t=float(alphas[index]),
        alpha_bar_t=float(alpha_bars[index]),
        beta_t=float(betas[index]),
    )
    assert np.allclose(posterior_mean, noise_mean)


def test_equal_variance_gaussian_kl_is_scaled_mean_squared_error():
    mean_q = np.array([0.2, -0.5])
    mean_p = np.array([-0.1, 0.7])
    variance = 0.3
    expected = np.sum((mean_q - mean_p) ** 2) / (2.0 * variance)
    assert np.isclose(
        isotropic_gaussian_kl_same_variance(mean_q, mean_p, variance),
        expected,
    )


def test_torch_q_sample_shapes_and_given_noise():
    schedule = make_torch_schedule(steps=10)
    x0 = torch.ones((4, 2))
    t = torch.tensor([0, 1, 5, 9])
    noise = torch.zeros_like(x0)
    xt, returned_noise = q_sample_torch(x0, t, schedule, noise=noise)
    assert xt.shape == x0.shape
    assert torch.equal(returned_noise, noise)
    assert torch.allclose(xt[:, 0], schedule.alpha_bars[t].sqrt())
