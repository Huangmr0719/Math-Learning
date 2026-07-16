import math

import numpy as np

from src.math_checks import (
    classifier_free_guidance,
    euler_integrate_scalar,
    heun_integrate_scalar,
    stable_logmeanexp,
    velocity_to_x0_epsilon,
)


def test_stable_logmeanexp_matches_direct_small_values():
    values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 1.0]])
    direct = np.log(np.mean(np.exp(values), axis=0))
    assert np.allclose(stable_logmeanexp(values, axis=0), direct)


def test_stable_logmeanexp_remains_finite_for_large_values():
    result = stable_logmeanexp(np.array([1000.0, 1001.0]), axis=0)
    assert np.isfinite(result)


def test_v_parameterization_round_trip():
    alpha = 0.8
    sigma = 0.6
    x0 = np.array([1.2, -0.5])
    epsilon = np.array([-0.3, 0.9])
    xt = alpha * x0 + sigma * epsilon
    velocity = alpha * epsilon - sigma * x0
    recovered_x0, recovered_epsilon = velocity_to_x0_epsilon(
        xt, velocity, alpha, sigma
    )
    assert np.allclose(recovered_x0, x0)
    assert np.allclose(recovered_epsilon, epsilon)


def test_cfg_boundary_scales():
    unconditional = np.array([1.0, 2.0])
    conditional = np.array([3.0, 4.0])
    assert np.allclose(
        classifier_free_guidance(unconditional, conditional, 0.0),
        unconditional,
    )
    assert np.allclose(
        classifier_free_guidance(unconditional, conditional, 1.0),
        conditional,
    )


def test_heun_is_more_accurate_than_euler_on_exponential_ode():
    derivative = lambda x, _t: x
    euler = euler_integrate_scalar(derivative, 1.0, 0.0, 1.0, 8)
    heun = heun_integrate_scalar(derivative, 1.0, 0.0, 1.0, 8)
    assert abs(heun - math.e) < abs(euler - math.e)
