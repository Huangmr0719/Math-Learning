#!/usr/bin/env python3
"""课程数学公式的轻量命令行检查。

这些 demo 用数值实验支持推导，但不替代 notebook 中的严格证明。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.math_checks import (  # noqa: E402
    finite_difference_gradient,
    gaussian_kl_standard_normal,
    ornstein_uhlenbeck_variance,
    simulate_ornstein_uhlenbeck,
)


def gaussian_kl_demo() -> None:
    mu = np.array([0.4, -0.7])
    logvar = np.array([0.2, -0.3])
    analytic = gaussian_kl_standard_normal(mu, logvar)

    rng = np.random.default_rng(7)
    variance = np.exp(logvar)
    samples = mu + np.sqrt(variance) * rng.standard_normal((300_000, 2))
    log_q = -0.5 * np.sum(
        np.log(2.0 * np.pi * variance) + (samples - mu) ** 2 / variance,
        axis=1,
    )
    log_p = -0.5 * np.sum(
        np.log(2.0 * np.pi) + samples**2,
        axis=1,
    )
    numeric = float(np.mean(log_q - log_p))
    print(
        "gaussian-kl:",
        f"analytic={analytic:.6f}",
        f"monte-carlo={numeric:.6f}",
        f"error={abs(analytic - numeric):.3e}",
    )


def softmax_gradient_demo() -> None:
    logits = np.array([0.2, -0.4, 1.1])
    direction = np.array([0.7, -0.2, 0.5])

    def objective(values: np.ndarray) -> float:
        shifted = values - values.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        return float(probabilities @ direction)

    shifted = logits - logits.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    jacobian = np.diag(probabilities) - np.outer(probabilities, probabilities)
    analytic = jacobian @ direction
    numeric = finite_difference_gradient(objective, logits)
    print(
        "softmax-grad:",
        f"max-error={np.max(np.abs(analytic - numeric)):.3e}",
    )


def gaussian_score_demo() -> None:
    mean = np.array([0.3, -0.5])
    variance = np.array([0.7, 1.4])
    point = np.array([1.1, -0.2])

    def log_density(values: np.ndarray) -> float:
        return float(
            -0.5
            * np.sum(
                np.log(2.0 * np.pi * variance)
                + (values - mean) ** 2 / variance
            )
        )

    analytic = -(point - mean) / variance
    numeric = finite_difference_gradient(log_density, point)
    print(
        "score-gaussian:",
        f"max-error={np.max(np.abs(analytic - numeric)):.3e}",
    )


def ou_sde_demo() -> None:
    times, _, sample_variances, _ = simulate_ornstein_uhlenbeck(
        particles=100_000,
        steps=200,
        dt=0.005,
        diffusion=0.8,
        seed=7,
    )
    theory = ornstein_uhlenbeck_variance(times, diffusion=0.8)
    print(
        "ou-sde:",
        f"sample-variance={sample_variances[-1]:.6f}",
        f"theory={theory[-1]:.6f}",
        f"error={abs(sample_variances[-1] - theory[-1]):.3e}",
    )


DEMOS = {
    "gaussian-kl": gaussian_kl_demo,
    "softmax-grad": softmax_gradient_demo,
    "score-gaussian": gaussian_score_demo,
    "ou-sde": ou_sde_demo,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        choices=["all", *DEMOS],
        default="all",
        help="选择一个数值检查；默认运行全部。",
    )
    arguments = parser.parse_args()
    selected = DEMOS.values() if arguments.demo == "all" else [DEMOS[arguments.demo]]
    for demo in selected:
        demo()


if __name__ == "__main__":
    main()
