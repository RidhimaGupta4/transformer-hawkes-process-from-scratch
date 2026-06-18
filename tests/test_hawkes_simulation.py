"""
Tests for the Hawkes process simulator. Includes statistical
validation: the simulated process's empirical event rate is checked
against the known closed-form stationary rate for a multivariate
Hawkes process, not just shape/type checks. This is the real test
of correctness for a stochastic simulator.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hawkes_simulation import HawkesParams, simulate_hawkes, true_intensity


def test_explosive_process_raises():
    # alpha with spectral radius >= 1 must be rejected at construction,
    # not silently simulated into an exploding/invalid process.
    mu = np.array([0.5])
    alpha = np.array([[1.5]])  # spectral radius 1.5, explosive
    beta = np.array([[1.0]])
    try:
        HawkesParams(mu=mu, alpha=alpha, beta=beta)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "Expected explosive process to be rejected"
    print("[PASS] explosive (non-stationary) Hawkes parameters correctly rejected")


def test_univariate_timestamps_sorted_and_in_range():
    mu = np.array([0.3])
    alpha = np.array([[0.4]])
    beta = np.array([[1.0]])
    params = HawkesParams(mu=mu, alpha=alpha, beta=beta)

    timestamps, types = simulate_hawkes(params, t_max=100.0, seed=42)
    assert len(timestamps) > 0, "Should generate at least some events"
    assert np.all(np.diff(timestamps) > 0), "Timestamps must be strictly increasing"
    assert np.all(timestamps >= 0) and np.all(timestamps <= 100.0)
    assert np.all((types == 0))  # univariate: only type 0 exists
    print(f"[PASS] univariate simulation: {len(timestamps)} events, sorted, in range")


def test_univariate_empirical_rate_matches_theory():
    """
    For a univariate Hawkes process with exponential kernel, the
    theoretical stationary event rate is:

        rate = mu / (1 - alpha)

    (this is the standard branching-process result: each event
    produces on average alpha offspring events, so the total
    expected number of events per immigrant event is 1/(1-alpha),
    and immigrants arrive at rate mu).

    We simulate a long realisation and check the empirical rate
    converges to this theoretical value within statistical error.
    """
    mu_val, alpha_val, beta_val = 0.5, 0.6, 2.0
    mu = np.array([mu_val])
    alpha = np.array([[alpha_val]])
    beta = np.array([[beta_val]])
    params = HawkesParams(mu=mu, alpha=alpha, beta=beta)

    t_max = 20_000.0
    timestamps, _ = simulate_hawkes(params, t_max=t_max, seed=123)

    empirical_rate = len(timestamps) / t_max
    theoretical_rate = mu_val / (1 - alpha_val)

    # With t_max=20000 and rate ~1.25, we expect ~25000 events; the
    # standard error of the rate estimate is small enough that a 10%
    # relative tolerance is a meaningful, non-trivial check.
    relative_error = abs(empirical_rate - theoretical_rate) / theoretical_rate
    assert relative_error < 0.10, (
        f"Empirical rate {empirical_rate:.4f} deviates too far from "
        f"theoretical rate {theoretical_rate:.4f} (rel. error {relative_error:.2%})"
    )
    print(
        f"[PASS] empirical rate {empirical_rate:.4f} matches theoretical "
        f"rate {theoretical_rate:.4f} (rel. error {relative_error:.2%})"
    )


def test_multivariate_empirical_rate_matches_theory():
    """
    For a multivariate Hawkes process, the stationary expected rate
    vector r solves r = mu + alpha @ r, i.e. r = (I - alpha)^{-1} @ mu
    (each type-v event produces, in expectation, alpha[u, v] offspring
    events of type u, since the kernel integral of alpha*beta*exp(-beta*t)
    over (0, inf) equals alpha regardless of beta). This is checked
    against a long simulated realisation, not just shapes/types.
    """
    mu = np.array([0.15, 0.05])
    alpha = np.array([[0.20, 0.00], [0.55, 0.10]])
    beta = np.array([[1.0, 1.0], [1.0, 1.0]])
    params = HawkesParams(mu=mu, alpha=alpha, beta=beta)

    t_max = 30_000.0
    timestamps, types = simulate_hawkes(params, t_max=t_max, seed=99)

    empirical_rates = np.array([
        (types == 0).sum() / t_max,
        (types == 1).sum() / t_max,
    ])
    theoretical_rates = np.linalg.inv(np.eye(2) - alpha) @ mu

    relative_error = np.abs(empirical_rates - theoretical_rates) / theoretical_rates
    assert np.all(relative_error < 0.10), (
        f"Empirical rates {empirical_rates} deviate too far from theoretical "
        f"rates {theoretical_rates} (rel. errors {relative_error})"
    )
    print(
        f"[PASS] multivariate empirical rates {np.round(empirical_rates, 4)} match "
        f"theoretical rates {np.round(theoretical_rates, 4)} (rel. errors {np.round(relative_error, 3)})"
    )


def test_multivariate_shapes_and_cross_excitation():
    # 2-type process where type 0 strongly excites type 1 but not
    # vice versa; check both types appear and timestamps are sorted.
    mu = np.array([0.2, 0.1])
    alpha = np.array([[0.1, 0.0], [0.5, 0.1]])
    beta = np.array([[1.0, 1.0], [1.0, 1.0]])
    params = HawkesParams(mu=mu, alpha=alpha, beta=beta)

    timestamps, types = simulate_hawkes(params, t_max=500.0, seed=7)
    assert np.all(np.diff(timestamps) > 0)
    assert set(np.unique(types)).issubset({0, 1})
    assert len(np.unique(types)) == 2, "Expected both event types to appear"
    print(f"[PASS] multivariate (2-type) simulation: {len(timestamps)} events, both types present")


def test_true_intensity_jumps_at_events_and_decays():
    mu = np.array([0.3])
    alpha = np.array([[0.8]])
    beta = np.array([[2.0]])
    params = HawkesParams(mu=mu, alpha=alpha, beta=beta)

    # Manually place one event at t=5 and check intensity behaviour
    timestamps = np.array([5.0])
    event_types = np.array([0])

    eval_times = np.array([4.9, 5.01, 6.0, 50.0])
    intensities = true_intensity(params, eval_times, timestamps, event_types, target_type=0)

    # Before the event: intensity should equal baseline mu
    assert np.isclose(intensities[0], mu[0], atol=1e-6)
    # Just after the event: intensity should jump up due to excitation
    assert intensities[1] > intensities[0]
    # Intensity should decay over time after the jump
    assert intensities[1] > intensities[2] > intensities[3]
    # Long after the event, intensity should have decayed back near baseline
    assert np.isclose(intensities[3], mu[0], atol=1e-3)
    print("[PASS] true_intensity: correct jump at event time and decay back to baseline")


if __name__ == "__main__":
    test_explosive_process_raises()
    test_univariate_timestamps_sorted_and_in_range()
    test_univariate_empirical_rate_matches_theory()
    test_multivariate_empirical_rate_matches_theory()
    test_multivariate_shapes_and_cross_excitation()
    test_true_intensity_jumps_at_events_and_decays()
    print("\nAll Hawkes simulation tests passed.")
