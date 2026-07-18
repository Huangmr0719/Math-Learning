import math

import numpy as np

from src.math_checks import (
    compare_values,
    conditional_squared_error,
    coupling_marginals,
    finite_difference_gradient,
    gaussian_kl_standard_normal,
    kl_discrete,
    kl_jensen_support_mass,
    normalize_distribution,
    ornstein_uhlenbeck_variance,
    simulate_ornstein_uhlenbeck,
    stable_logmeanexp,
)


def test_normalize_distribution():
    actual = normalize_distribution(np.array([2.0, 3.0, 5.0]))
    assert np.allclose(actual, np.array([0.2, 0.3, 0.5]))
    assert np.isclose(actual.sum(), 1.0)


def test_kl_is_zero_for_equal_distributions():
    distribution = np.array([0.2, 0.3, 0.5])
    assert np.isclose(kl_discrete(distribution, distribution), 0.0)


def test_kl_respects_support_boundary():
    value = kl_discrete(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert math.isinf(value)
    assert value > 0


def test_kl_is_generally_asymmetric():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.3, 0.3, 0.4])
    assert not np.isclose(kl_discrete(p, q), kl_discrete(q, p))


def test_kl_jensen_expectation_only_sums_q_on_p_support():
    p = np.array([1.0, 0.0])
    q = np.array([0.5, 0.5])
    support_mass = kl_jensen_support_mass(p, q)
    assert np.isclose(support_mass, 0.5)
    assert kl_discrete(p, q) >= -math.log(support_mass)


def test_gaussian_kl_minimum_is_standard_normal():
    mu = np.zeros(4)
    logvar = np.zeros(4)
    assert np.isclose(gaussian_kl_standard_normal(mu, logvar), 0.0)


def test_finite_difference_gradient_matches_quadratic():
    point = np.array([1.5, -2.0])
    numeric = finite_difference_gradient(lambda value: float(np.sum(value**2)), point)
    assert np.allclose(numeric, 2.0 * point, atol=1e-6)


def test_compare_values_reports_tolerance():
    report = compare_values(1.0, 1.005, tolerance=0.01)
    assert report.within_tolerance
    assert np.isclose(report.absolute_error, 0.005)


def test_iwae_logmeanexp_recovers_elbo_at_one_sample():
    log_weights = np.array([[-2.0, -0.5, 0.7]])  # [K=1, B=3]
    per_example = stable_logmeanexp(log_weights, axis=0)
    assert np.allclose(per_example, log_weights[0])


def test_iwae_toy_bound_tightens_in_monte_carlo_expectation():
    """Numerically support, but do not replace, the IWAE monotonicity proof."""

    rng = np.random.default_rng(19)
    log_weights = rng.normal(-0.5, 1.0, size=(80_000, 32))
    estimates = [
        float(np.mean(stable_logmeanexp(log_weights[:, :k], axis=1)))
        for k in (1, 4, 16, 32)
    ]
    assert estimates[0] < estimates[1] < estimates[2] < estimates[3] < 0.0
    assert estimates[-1] > -0.03


def test_ornstein_uhlenbeck_euler_maruyama_matches_theory():
    times, _, sample_variances, _ = simulate_ornstein_uhlenbeck(
        particles=100_000,
        steps=200,
        dt=0.005,
        diffusion=0.8,
        seed=0,
    )
    theory = ornstein_uhlenbeck_variance(times, diffusion=0.8)
    assert np.isclose(sample_variances[-1], theory[-1], rtol=0.03)


def test_conditional_mean_minimizes_expected_squared_error():
    targets = np.array([-2.0, 4.0])
    probabilities = np.array([0.25, 0.75])
    conditional_mean = float(np.sum(targets * probabilities))
    at_mean = conditional_squared_error(targets, probabilities, conditional_mean)
    assert at_mean < conditional_squared_error(targets, probabilities, 0.0)
    assert at_mean < conditional_squared_error(targets, probabilities, 4.0)


def test_coupling_marginals_are_row_and_column_sums():
    coupling = np.array([[0.4, 0.1], [0.2, 0.3]])
    source, target = coupling_marginals(coupling)
    assert np.allclose(source, np.array([0.5, 0.5]))
    assert np.allclose(target, np.array([0.6, 0.4]))
