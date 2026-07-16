"""小型、确定性的数学验证函数。

这些函数用于支持推导，而不是替代证明。每个函数都显式处理定义域，
尤其避免把 KL 中应为无穷大的零概率情况静默过滤掉。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Comparison:
    analytic: float
    numeric: float
    absolute_error: float
    within_tolerance: bool


def normalize_distribution(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("概率向量必须是一维数组。")
    if np.any(values < 0):
        raise ValueError("概率不能为负数。")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("概率总和必须大于 0。")
    return values / total


def kl_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """计算 D_KL(p || q)，正确处理支持集边界。"""

    p = normalize_distribution(p)
    q = normalize_distribution(q)
    if p.shape != q.shape:
        raise ValueError("p 与 q 必须具有相同 shape。")

    # 若 p 在某处认为事件可能发生，而 q 断言该事件概率为 0，
    # log(p / 0) 发散，因此 KL 必须是正无穷。
    if np.any((p > 0) & (q == 0)):
        return float("inf")

    # p=0 的项按极限约定贡献 0；只对 p>0 的位置计算。
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def kl_jensen_support_mass(p: np.ndarray, q: np.ndarray) -> float:
    """返回 Jensen 证明中 E_p[q(X)/p(X)] 的正确取值。

    该期望只会遍历 p 的支持集，因此一般等于
    ``sum(q[p > 0])``，而不一定等于 1。只有 q 在 p 的支持集之外
    没有概率质量时，结果才等于 1。
    """

    p = normalize_distribution(p)
    q = normalize_distribution(q)
    if p.shape != q.shape:
        raise ValueError("p 与 q 必须具有相同 shape。")
    if np.any((p > 0) & (q == 0)):
        raise ValueError("Jensen 比值 q/p 要求 p 的支持集包含在 q 的支持集中。")
    return float(np.sum(q[p > 0]))


def gaussian_kl_standard_normal(mu: np.ndarray, logvar: np.ndarray) -> float:
    """D_KL(N(mu, diag(exp(logvar))) || N(0, I))。"""

    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    if mu.shape != logvar.shape:
        raise ValueError("mu 与 logvar 必须具有相同 shape。")
    return float(-0.5 * np.sum(1.0 + logvar - mu**2 - np.exp(logvar)))


def finite_difference_gradient(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    gradient = np.zeros_like(point)
    for index in np.ndindex(point.shape):
        plus = point.copy()
        minus = point.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (function(plus) - function(minus)) / (2.0 * epsilon)
    return gradient


def compare_values(
    analytic: float,
    numeric: float,
    *,
    tolerance: float = 1e-2,
) -> Comparison:
    error = abs(float(analytic) - float(numeric))
    return Comparison(
        analytic=float(analytic),
        numeric=float(numeric),
        absolute_error=error,
        within_tolerance=error <= tolerance,
    )


def validate_diffusion_schedule(betas: np.ndarray) -> np.ndarray:
    """检查并返回一维 DDPM beta schedule。

    每个 beta_t 都必须位于 (0, 1)，否则一步转移的方差或保留比例
    alpha_t = 1 - beta_t 将失去概率意义。
    """

    betas = np.asarray(betas, dtype=float)
    if betas.ndim != 1 or betas.size == 0:
        raise ValueError("betas 必须是非空一维数组。")
    if np.any((betas <= 0.0) | (betas >= 1.0)):
        raise ValueError("每个 beta_t 都必须严格位于 (0, 1)。")
    return betas


def diffusion_coefficients(
    betas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 alpha_t 与累计乘积 alpha_bar_t。"""

    betas = validate_diffusion_schedule(betas)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return alphas, alpha_bars


def q_sample_from_x0(
    x0: np.ndarray,
    alpha_bar_t: float,
    noise: np.ndarray,
) -> np.ndarray:
    """闭式采样 q(x_t | x_0)。"""

    x0 = np.asarray(x0, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if x0.shape != noise.shape:
        raise ValueError("x0 与 noise 必须具有相同 shape。")
    if not 0.0 < alpha_bar_t <= 1.0:
        raise ValueError("alpha_bar_t 必须位于 (0, 1]。")
    return np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * noise


def ddpm_posterior_mean_variance(
    x0: np.ndarray,
    xt: np.ndarray,
    *,
    alpha_t: float,
    alpha_bar_t: float,
    alpha_bar_previous: float,
    beta_t: float,
) -> tuple[np.ndarray, float]:
    """计算 q(x_{t-1} | x_t, x_0) 的均值与标量方差。"""

    x0 = np.asarray(x0, dtype=float)
    xt = np.asarray(xt, dtype=float)
    if x0.shape != xt.shape:
        raise ValueError("x0 与 xt 必须具有相同 shape。")
    if not 0.0 < alpha_t < 1.0:
        raise ValueError("alpha_t 必须位于 (0, 1)。")
    if not 0.0 < alpha_bar_t < 1.0:
        raise ValueError("alpha_bar_t 必须位于 (0, 1)。")
    if not 0.0 < alpha_bar_previous <= 1.0:
        raise ValueError("alpha_bar_previous 必须位于 (0, 1]。")
    if not 0.0 < beta_t < 1.0:
        raise ValueError("beta_t 必须位于 (0, 1)。")

    denominator = 1.0 - alpha_bar_t
    coefficient_x0 = np.sqrt(alpha_bar_previous) * beta_t / denominator
    coefficient_xt = (
        np.sqrt(alpha_t) * (1.0 - alpha_bar_previous) / denominator
    )
    mean = coefficient_x0 * x0 + coefficient_xt * xt
    variance = beta_t * (1.0 - alpha_bar_previous) / denominator
    return mean, float(variance)


def ddpm_mean_from_noise(
    xt: np.ndarray,
    predicted_noise: np.ndarray,
    *,
    alpha_t: float,
    alpha_bar_t: float,
    beta_t: float,
) -> np.ndarray:
    """由 epsilon 参数化计算 DDPM reverse mean。"""

    xt = np.asarray(xt, dtype=float)
    predicted_noise = np.asarray(predicted_noise, dtype=float)
    if xt.shape != predicted_noise.shape:
        raise ValueError("xt 与 predicted_noise 必须具有相同 shape。")
    if not 0.0 < alpha_t < 1.0:
        raise ValueError("alpha_t 必须位于 (0, 1)。")
    if not 0.0 < alpha_bar_t < 1.0:
        raise ValueError("alpha_bar_t 必须位于 (0, 1)。")
    return (
        xt - beta_t * predicted_noise / np.sqrt(1.0 - alpha_bar_t)
    ) / np.sqrt(alpha_t)


def isotropic_gaussian_kl_same_variance(
    mean_q: np.ndarray,
    mean_p: np.ndarray,
    variance: float,
) -> float:
    """相同各向同性协方差 Gaussian 的 KL。"""

    mean_q = np.asarray(mean_q, dtype=float)
    mean_p = np.asarray(mean_p, dtype=float)
    if mean_q.shape != mean_p.shape:
        raise ValueError("两个均值必须具有相同 shape。")
    if variance <= 0.0:
        raise ValueError("variance 必须大于 0。")
    return float(np.sum((mean_q - mean_p) ** 2) / (2.0 * variance))


def stable_logmeanexp(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """稳定计算 log(mean(exp(values)))。"""

    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.mean(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis)


def velocity_to_x0_epsilon(
    xt: np.ndarray,
    velocity: np.ndarray,
    alpha: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """将 v-prediction 转回 x0 与 epsilon。"""

    xt = np.asarray(xt, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if xt.shape != velocity.shape:
        raise ValueError("xt 与 velocity 必须具有相同 shape。")
    if not np.isclose(alpha**2 + sigma**2, 1.0, atol=1e-6):
        raise ValueError("alpha^2 + sigma^2 必须等于 1。")
    x0 = alpha * xt - sigma * velocity
    epsilon = sigma * xt + alpha * velocity
    return x0, epsilon


def classifier_free_guidance(
    unconditional: np.ndarray,
    conditional: np.ndarray,
    scale: float,
) -> np.ndarray:
    """CFG 线性外推。"""

    unconditional = np.asarray(unconditional, dtype=float)
    conditional = np.asarray(conditional, dtype=float)
    if unconditional.shape != conditional.shape:
        raise ValueError("conditional 与 unconditional shape 必须相同。")
    return unconditional + scale * (conditional - unconditional)


def euler_integrate_scalar(
    derivative: Callable[[float, float], float],
    initial: float,
    start: float,
    end: float,
    steps: int,
) -> float:
    if steps < 1:
        raise ValueError("steps 至少为 1。")
    h = (end - start) / steps
    x = float(initial)
    t = float(start)
    for _ in range(steps):
        x += h * derivative(x, t)
        t += h
    return x


def heun_integrate_scalar(
    derivative: Callable[[float, float], float],
    initial: float,
    start: float,
    end: float,
    steps: int,
) -> float:
    if steps < 1:
        raise ValueError("steps 至少为 1。")
    h = (end - start) / steps
    x = float(initial)
    t = float(start)
    for _ in range(steps):
        k1 = derivative(x, t)
        k2 = derivative(x + h * k1, t + h)
        x += 0.5 * h * (k1 + k2)
        t += h
    return x


def simulate_ornstein_uhlenbeck(
    *,
    particles: int,
    steps: int,
    dt: float,
    diffusion: float,
    seed: int = 7,
    initial: float = 0.0,
    mean_reversion: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用 Euler–Maruyama 模拟一维 Ornstein–Uhlenbeck 过程。

    模型为 ``dX_t = -mean_reversion * X_t dt + diffusion dW_t``。
    返回时间、样本均值、样本方差和最终粒子。时间从完成第一步后的
    ``dt`` 开始，避免把更新后的统计量错误标在 ``t=0``。
    """

    if particles < 1 or steps < 1:
        raise ValueError("particles 与 steps 都必须至少为 1。")
    if dt <= 0.0:
        raise ValueError("dt 必须大于 0。")
    if diffusion < 0.0:
        raise ValueError("diffusion 不能为负。")
    if mean_reversion <= 0.0:
        raise ValueError("mean_reversion 必须大于 0。")

    rng = np.random.default_rng(seed)
    x = np.full(particles, float(initial))
    means = np.empty(steps)
    variances = np.empty(steps)
    for index in range(steps):
        noise = rng.standard_normal(particles)
        x = (
            x
            - mean_reversion * x * dt
            + diffusion * np.sqrt(dt) * noise
        )
        means[index] = x.mean()
        variances[index] = x.var()
    times = np.arange(1, steps + 1, dtype=float) * dt
    return times, means, variances, x


def ornstein_uhlenbeck_variance(
    times: np.ndarray,
    *,
    diffusion: float,
    mean_reversion: float = 0.5,
    initial_variance: float = 0.0,
) -> np.ndarray:
    """返回 Ornstein–Uhlenbeck 过程的解析方差。"""

    times = np.asarray(times, dtype=float)
    if np.any(times < 0.0):
        raise ValueError("times 不能包含负数。")
    if diffusion < 0.0:
        raise ValueError("diffusion 不能为负。")
    if mean_reversion <= 0.0:
        raise ValueError("mean_reversion 必须大于 0。")
    if initial_variance < 0.0:
        raise ValueError("initial_variance 不能为负。")

    decay = np.exp(-2.0 * mean_reversion * times)
    stationary_variance = diffusion**2 / (2.0 * mean_reversion)
    return initial_variance * decay + stationary_variance * (1.0 - decay)


def conditional_squared_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    prediction: float,
) -> float:
    """计算离散条件分布下的期望平方误差。"""

    targets = np.asarray(targets, dtype=float)
    probabilities = normalize_distribution(probabilities)
    if targets.ndim != 1 or targets.shape != probabilities.shape:
        raise ValueError("targets 与 probabilities 必须是 shape 相同的一维数组。")
    return float(np.sum(probabilities * (targets - prediction) ** 2))


def coupling_marginals(coupling: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回二维 coupling 的 source 与 target 边缘分布。"""

    coupling = np.asarray(coupling, dtype=float)
    if coupling.ndim != 2:
        raise ValueError("coupling 必须是二维联合概率表。")
    if np.any(coupling < 0.0):
        raise ValueError("coupling 中的概率不能为负。")
    total = float(coupling.sum())
    if not np.isclose(total, 1.0):
        raise ValueError("coupling 的全部概率必须和为 1。")
    return coupling.sum(axis=1), coupling.sum(axis=0)
