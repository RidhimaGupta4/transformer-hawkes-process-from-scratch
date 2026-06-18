"""
Simulates a multivariate Hawkes process using Ogata's modified
thinning algorithm (Ogata, 1981). This gives event sequences with a
known ground-truth generative process, which is essential for
validating a temporal point process model: with real data you only
ever see one realisation and never know the true intensity, but with
simulated data you can check whether the trained model recovers
parameters and intensity shapes close to the ones used to generate
the data.

Model:
    For event type u, the conditional intensity is

        lambda_u(t) = mu_u + sum_v sum_{t_j < t, type=v}
                          alpha_{u,v} * beta_{u,v} * exp(-beta_{u,v} (t - t_j))

    i.e. a multivariate Hawkes process with exponential decay
    kernels (Hawkes, 1971), where each past event of type v
    contributes a decaying excitation to the future intensity of
    type u, scaled by alpha_{u,v} and decaying at rate beta_{u,v}.

Reference:
    Ogata, Y. (1981). "On Lewis' simulation method for point
    processes." IEEE Transactions on Information Theory.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class HawkesParams:
    """
    Parameters for a K-type multivariate Hawkes process.

    mu:    base intensities, shape (K,). mu[u] is the background
           rate of events of type u with no excitation.
    alpha: excitation matrix, shape (K, K). alpha[u, v] is how much
           an event of type v excites the intensity of type u.
    beta:  decay matrix, shape (K, K). beta[u, v] is the decay rate
           of that excitation; larger beta means the excitation
           fades faster.
    """

    mu: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray

    def __post_init__(self):
        k = len(self.mu)
        assert self.alpha.shape == (k, k), "alpha must be (K, K)"
        assert self.beta.shape == (k, k), "beta must be (K, K)"
        assert np.all(self.mu > 0), "base intensities must be positive"
        assert np.all(self.beta > 0), "decay rates must be positive"
        # Spectral radius condition for stationarity (branching ratio < 1):
        # the process must not explode in expectation. This checks that
        # the "average number of offspring events per event" is below 1.
        branching_matrix = self.alpha  # since kernel integral = alpha (beta cancels)
        spectral_radius = np.max(np.abs(np.linalg.eigvals(branching_matrix)))
        assert spectral_radius < 1.0, (
            f"Process is non-stationary / explosive: spectral radius of alpha "
            f"is {spectral_radius:.3f}, must be < 1. Reduce alpha values."
        )


def simulate_hawkes(
    params: HawkesParams,
    t_max: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a multivariate Hawkes process on [0, t_max] using
    Ogata's modified thinning algorithm, with an O(1)-per-event
    incremental state update.

    Naively, computing lambda_u(t) requires summing the decayed
    contribution of every past event, which costs O(N) per
    evaluation and O(N^2) overall as the sequence grows -- too slow
    for sequences beyond a few thousand events. This implementation
    exploits the fact that an exponential kernel is Markovian: the
    total decayed excitation S[u, v](t) = sum_{t_j < t, type=v}
    exp(-beta[u,v](t - t_j)) satisfies the recursion

        S[u, v](t) = exp(-beta[u,v] * dt) * (S[u, v](t_prev) + [v fired at t_prev])

    so it can be updated in O(K) per event (K = number of types)
    instead of recomputed from scratch in O(N).

    Args:
        params: HawkesParams defining mu, alpha, beta.
        t_max: simulation horizon.
        seed: random seed for reproducibility.

    Returns:
        timestamps: sorted event times, shape (N,)
        event_types: corresponding event types in [0, K-1], shape (N,)
    """
    rng = np.random.default_rng(seed)
    k = len(params.mu)

    timestamps: list[float] = []
    event_types: list[int] = []

    # S[u, v] = decayed sum of past type-v events' contribution to
    # type-u's intensity, evaluated at the current reference time t_ref.
    S = np.zeros((k, k), dtype=float)
    t_ref = 0.0
    t = 0.0

    def intensities_at(t_query: float) -> np.ndarray:
        """lambda_u(t_query) given S evaluated at t_ref <= t_query."""
        dt_ref = t_query - t_ref
        decay = np.exp(-params.beta * dt_ref)  # (K, K)
        S_at_query = S * decay
        return params.mu + S_at_query.sum(axis=1)

    while t < t_max:
        # Current total intensity is a valid upper bound for the next
        # proposal: between events, intensity only decays (pure
        # exponential decay, no growth), so lambda(s) <= lambda(t)
        # for any s in (t, next_jump].
        lambdas_u = intensities_at(t)
        total_intensity = lambdas_u.sum()

        if total_intensity <= 0:
            break

        dt = rng.exponential(scale=1.0 / total_intensity)
        t_candidate = t + dt

        if t_candidate > t_max:
            break

        lambdas_u_candidate = intensities_at(t_candidate)
        total_candidate = lambdas_u_candidate.sum()

        u = rng.uniform(0.0, 1.0)
        if u <= total_candidate / total_intensity:
            # Accept the candidate event. First, advance S to the
            # candidate time (decay existing state), then record the
            # new event's contribution for future steps.
            dt_ref = t_candidate - t_ref
            S *= np.exp(-params.beta * dt_ref)
            t_ref = t_candidate

            probs = lambdas_u_candidate / total_candidate
            event_type = int(rng.choice(k, p=probs))

            # New event of type `event_type` contributes alpha*beta
            # to each target type's intensity, decaying from now on.
            S[:, event_type] += params.alpha[:, event_type] * params.beta[:, event_type]

            timestamps.append(t_candidate)
            event_types.append(event_type)

        t = t_candidate

    return np.array(timestamps), np.array(event_types)


def true_intensity(
    params: HawkesParams,
    eval_times: np.ndarray,
    timestamps: np.ndarray,
    event_types: np.ndarray,
    target_type: int,
) -> np.ndarray:
    """
    Compute the ground-truth conditional intensity lambda_u(t) for a
    given target event type u, at each time in eval_times, given the
    observed history (timestamps, event_types). Used to compare the
    model's learned intensity against the true generative intensity.

    Args:
        params: the HawkesParams used to generate the data.
        eval_times: times at which to evaluate intensity, shape (M,)
        timestamps: observed event times (history), shape (N,)
        event_types: observed event types (history), shape (N,)
        target_type: which type u to compute lambda_u for.

    Returns:
        intensities, shape (M,)
    """
    intensities = np.full(len(eval_times), params.mu[target_type], dtype=float)
    for tj, vj in zip(timestamps, event_types):
        mask = eval_times > tj  # only past events contribute
        decay = np.exp(-params.beta[target_type, vj] * (eval_times[mask] - tj))
        intensities[mask] += (
            params.alpha[target_type, vj] * params.beta[target_type, vj] * decay
        )
    return intensities
