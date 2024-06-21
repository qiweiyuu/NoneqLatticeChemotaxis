# generic functions for simulating the nonequilibrium Ising model and balancing

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import lax
import numpy as onp
import matplotlib.pyplot as plt
import argparse, os
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

gaussian_func = (
    lambda x, mu, sigma, w: w
    * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    / np.sqrt(2 * np.pi * sigma**2)
)


## set default parameters
parser = argparse.ArgumentParser(description="ising_neq_v1")
parser.add_argument("--N", type=int, default=6, help="lattice size")
parser.add_argument(
    "--activity-threshold",
    type=float,
    default=0.05,
    help="Threshold for balancing the activities",
)
parser.add_argument(
    "--prominence-threshold",
    type=float,
    default=0.01,
    help="prominence threshold for identifying the peaks",
)
parser.add_argument(
    "--step-thermalize",
    type=int,
    default=1000000,
    help="steps to thermalize the system; need to be thrown away for computing the switching time",
)
parser.add_argument(
    "--max-step-balancing",
    type=int,
    default=1000000,
    help="max number of KMC steps for finding the balanced k1 (per step in the binary search)",
)
parser.add_argument(
    "--max-step-evaluating",
    type=int,
    default=5000000,
    help="max number of KMC steps for calculating the final dwell and switching time (per k3 value)",
)
flags_default = parser.parse_args("")


############################################################################################################
def KMC_step(
    a,
    J,
    k1,
    k2,
    k3,
    kn1,
    kn2,
    kn3,
    N=flags_default.N,
    r1=onp.random.rand(),
    r2=onp.random.rand(),
):
    field = J * (
        np.roll(a, 1, axis=0)
        + np.roll(a, -1, axis=0)
        + np.roll(a, 1, axis=1)
        + np.roll(a, -1, axis=1)
    )
    rates = k2 * (a == 1) + k3 * (a == 0) + k1 * (a == -1) * np.exp(field)
    rates_backward = (
        kn1 * (a == 1)
        + kn2 * (a == 0) * np.exp(field / 2)
        + kn3 * (a == -1) * np.exp(field / 2)
    )
    rates_all = np.stack([rates, rates_backward])
    rates_sum = rates_all.ravel().cumsum()
    R = rates_sum[-1]

    # determine the time to the next event
    t = 1 / R * np.log(1 / r1)

    # find the spin to flip
    r = r2 * R
    dir, i, j = np.unravel_index(np.searchsorted(rates_sum, r), (2, N, N))
    a = a.at[i, j].set((a[i, j] + dir * 2) % 3 - 1)
    return a, t


def activity_estimate(
    k1,
    k2,
    k3,
    kn1,
    kn2,
    kn3,
    J,
    N=flags_default.N,
    steps_balance=int(1e4),
    m_threshold=-0.5,
):
    r1, r2 = onp.random.rand(steps_balance), onp.random.rand(steps_balance)
    r1, r2 = np.array(r1), np.array(r2)

    def step_fn(carry, i):
        a, t = carry
        a, dt = KMC_step(a, J, k1, k2, k3, kn1, kn2, kn3, N, r1=r1[i], r2=r2[i])
        t += dt
        return (a, t), (t, a.mean())

    initial_state = (np.array(onp.random.choice([-1, 0, 1], size=(N, N))), 0)
    final_state, (t_trace, m_trace) = lax.scan(
        step_fn, initial_state, np.arange(steps_balance)
    )

    activity_estimate = (
        1 - np.sum(np.diff(t_trace)[m_trace[:-1] < m_threshold]) / t_trace[-1]
    )
    return activity_estimate


def activity_estimate_binary_search(
    a_th,
    k2,
    k3,
    kn2,
    kn3,
    J,
    epsilon,
    N=flags_default.N,
    steps_balance=int(1e4),
    activity_tolerance=0.03,
):
    k1_l, k1_r = k3 / 2, 4.0
    k_tol = 1e-3
    a_l, a_r = (
        activity_estimate(k1_l, k2, k3, k1_l * epsilon, kn2, kn3, J, N, steps_balance),
        activity_estimate(k1_r, k2, k3, k1_r * epsilon, kn2, kn3, J, N, steps_balance),
    )
    while (
        onp.abs(a_l - a_th) > activity_tolerance
        or onp.abs(a_r - a_th) > activity_tolerance
    ):
        k1_m = (k1_l + k1_r) / 2
        a_m = activity_estimate(
            k1_m, k2, k3, k1_m * epsilon, kn2, kn3, J, N, steps_balance
        )
        if a_m < a_th:
            k1_l = k1_m
            a_l = a_m
        else:
            k1_r = k1_m
            a_r = a_m
        if onp.abs(k1_l - k1_r) < k_tol:
            break
    return (k1_l + k1_r) / 2


def calc_dwell_and_switching_times(
    k1,
    k2,
    k3,
    kn1,
    kn2,
    kn3,
    J,
    N=flags_default.N,
    max_steps=flags_default.max_step_evaluating,
    activity_only=False,
    transient_steps=flags_default.step_thermalize,
    dwell_threshold_factor=1.5,
    switching_threshold_factor=1.5,
    prominence_threshold=flags_default.prominence_threshold,
    use_kde=True,
    dataname=None,
    random_seed=None,
    return_thresholds=False,
    m_thresholds=None,
):
    # set random seed for multiple runs
    if random_seed is not None:
        onp.random.seed(random_seed)

    ## define thresholds for dwell and switching events
    dwell_threshold = 1 - onp.exp(-dwell_threshold_factor)
    switching_threshold = 1 - onp.exp(-switching_threshold_factor)

    ## thermalize the system
    r1, r2 = onp.random.rand(transient_steps), onp.random.rand(transient_steps)
    r1, r2 = np.array(r1), np.array(r2)

    def KMC_wrapper(i, a):
        a, t = KMC_step(a, J, k1, k2, k3, kn1, kn2, kn3, r1=r1[i], r2=r2[i])
        return a

    a_init = np.array(onp.random.choice([-1, 0, 1], size=(N, N)))
    a_thermalized = lax.fori_loop(0, transient_steps, KMC_wrapper, a_init)

    ## simulate the trajectory
    r1, r2 = onp.random.rand(max_steps), onp.random.rand(max_steps)
    r1, r2 = np.array(r1), np.array(r2)

    def step_fn(carry, i):
        a, t = carry
        a, dt = KMC_step(a, J, k1, k2, k3, kn1, kn2, kn3, N, r1=r1[i], r2=r2[i])
        t += dt
        return (a, t), (t, a.mean())

    initial_state = (a_thermalized, 0)
    final_state, (t_trace, m_trace) = lax.scan(
        step_fn, initial_state, np.arange(max_steps)
    )

    bin_edges = onp.arange(-N * N - 0.5, N * N + 1.5, 1) / (N * N)
    m_counts, bin_edges = onp.histogram(
        m_trace[:-1], weights=onp.diff(t_trace), bins=bin_edges
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pdf = m_counts / m_counts.sum() / (bin_edges[1] - bin_edges[0])

    if dataname is not None:
        np.savez(dataname + "_debug_hist.npz", m_counts=m_counts, bin_edges=bin_edges)

    if m_thresholds is not None:
        m_min, m_max = m_thresholds[0], m_thresholds[1]
    else:
        # identify the peaks
        smooth_factor = 1.0
        smooth_factor_step = 0.5
        if use_kde:
            m_resample = onp.random.choice(
                bin_centers,
                size=5000000,
                p=m_counts / m_counts.sum(),
            )
            while smooth_factor < 10:
                kde = gaussian_kde(
                    m_resample,
                    bw_method=(bin_edges[1] - bin_edges[0]) * smooth_factor,
                )
                m_mesh = onp.arange(-N * N, N * N + 1, 1) / (N * N)
                p_mesh = kde(m_mesh)
                p_mesh_augmented = onp.concatenate(([p_mesh[1]], p_mesh, [p_mesh[-2]]))
                pm_peaks = (
                    find_peaks(p_mesh_augmented, prominence=prominence_threshold)[0] - 1
                )
                if sum(pm_peaks <= 4) > 1:
                    # if there are still multiples near -1, increase the smooth factor
                    smooth_factor += smooth_factor_step
                else:
                    # done
                    break

            gm = GaussianMixture(n_components=2, random_state=0).fit(
                m_resample.reshape(-1, 1)
            )
        else:
            m_mesh, p_mesh = bin_centers, pdf

        # add a point at the start and at the end to identify the peaks at the boundary
        p_mesh_augmented = onp.concatenate(([p_mesh[1]], p_mesh, [p_mesh[-2]]))
        pm_peaks = find_peaks(p_mesh_augmented, prominence=prominence_threshold)[0] - 1
        prominences = peak_prominences(p_mesh_augmented, pm_peaks + 1)[0]

        if activity_only:
            if len(pm_peaks) == 1 and m_mesh[pm_peaks[0]] < -0.5:
                return 0

        if len(pm_peaks) > 1:
            prominent_peaks = np.argsort(-prominences)[:2]
            m_peaks = np.array([m_mesh[pm_peaks[_]] for _ in prominent_peaks])
            m_min, m_max = m_peaks.min(), m_peaks.max()
        elif use_kde:
            m_min, m_max = gm.means_.min(), gm.means_.max()
        elif len(pm_peaks) == 1:
            m_min = m_mesh[0]
            m_max = m_mesh[pm_peaks[0]]
        else:
            m_min = m_mesh[0]
            m_max = m_mesh[-1]

    # print("k1 = {:.6f}, m_min = {:.6f}, m_max = {:.6f}".format(k1, m_min, m_max))

    m_trace_shifted = (m_trace - m_min) / (m_max - m_min) * 2 - 1
    m_trace_ternary = (m_trace_shifted > dwell_threshold).astype(np.int32) - (
        m_trace_shifted < -dwell_threshold
    ).astype(np.int32)

    def compute_shifted_states(trace):
        shifted_states = np.zeros_like(trace)
        shifted_states = shifted_states.at[0].set(np.sign(trace[0]))

        def update_state(i, shifted_states):
            new_state = trace[i] + (trace[i] == 0) * (shifted_states[i - 1] - trace[i])
            return shifted_states.at[i].set(new_state)

        shifted_states = np.array(
            lax.fori_loop(1, len(trace), update_state, shifted_states)
        )
        return shifted_states

    m_shifted_states = compute_shifted_states(m_trace_ternary)
    switching_id = np.where(np.diff(m_shifted_states))[0]

    # compute the dwell times
    id_start = switching_id[:-1] + 1
    id_end = switching_id[1:]
    dwell_state = onp.sign(m_shifted_states[id_start])
    dwell_times = t_trace[id_end + 1] - t_trace[id_start]
    t_dwell_up = dwell_times[dwell_state > 0]
    t_dwell_down = dwell_times[dwell_state < 0]

    if activity_only:
        t_dwell_mean = [t_dwell_up.mean(), t_dwell_down.mean()]
        if return_thresholds:
            return t_dwell_mean, [m_min, m_max]
        else:
            return t_dwell_mean[0] / sum(t_dwell_mean)

    ## determine the switching events
    def scan_previous_switch(carry, i):
        new_carry = (
            carry[0] + (m_trace_shifted[i] > switching_threshold) * (i - carry[0]),
            carry[1] + (m_trace_shifted[i] < -switching_threshold) * (i - carry[1]),
        )
        return new_carry, new_carry

    _, previous_switch = lax.scan(scan_previous_switch, (0, 0), np.arange(len(t_trace)))
    previous_switch = np.stack(previous_switch, axis=1)

    def scan_next_switch(carry, i):
        new_carry = (
            carry[0] - (m_trace_shifted[i] > switching_threshold) * (carry[0] - i),
            carry[1] - (m_trace_shifted[i] < -switching_threshold) * (carry[1] - i),
        )
        return new_carry, new_carry

    _, next_switch = lax.scan(
        scan_next_switch,
        (len(t_trace) - 1, len(t_trace) - 1),
        np.arange(len(t_trace))[::-1],
    )
    next_switch = np.stack(next_switch, axis=1)[::-1]

    switching_dir = m_shifted_states[id_start]
    switching_up_ids = onp.where(switching_dir == 1)[0][2:-2]
    switching_up_successful_mask = (
        t_trace[switching_id[switching_up_ids + 1]]
        > t_trace[next_switch[switching_id[switching_up_ids], 0]]
    ) & (
        t_trace[switching_id[switching_up_ids - 1]]
        < t_trace[previous_switch[switching_id[switching_up_ids], 1] + 1]
    )

    switching_up_ids = switching_id[switching_up_ids[switching_up_successful_mask]]
    t_switching_up = (
        t_trace[next_switch[switching_up_ids, 0]]
        - t_trace[previous_switch[switching_up_ids, 1] + 1]
    ).flatten() / switching_threshold_factor

    switching_down_ids = onp.where(switching_dir == -1)[0][2:-2]
    switching_down_successful_mask = (
        t_trace[switching_id[switching_down_ids + 1]]
        > t_trace[next_switch[switching_id[switching_down_ids], 1]]
    ) & (
        t_trace[switching_id[switching_down_ids - 1]]
        < t_trace[previous_switch[switching_id[switching_down_ids], 0] + 1]
    )

    switching_down_ids = switching_id[
        switching_down_ids[switching_down_successful_mask]
    ]
    t_switching_down = (
        t_trace[next_switch[switching_down_ids, 1]]
        - t_trace[previous_switch[switching_down_ids, 0] + 1]
    ).flatten() / switching_threshold_factor

    return (
        [t_dwell_up, t_dwell_down],
        [t_switching_up, t_switching_down],
        [m_counts, bin_edges],
        [m_min, m_max],
    )


def calc_kstar_binary(
    k2,
    k3,
    kn2,
    kn3,
    J,
    epsilon,
    N=flags_default.N,
    max_steps=int(1e5),
    activity_tolerance=flags_default.activity_threshold,
    k1_tolerance=1e-3,
    params=None,
):
    k1_th = onp.array(
        [
            activity_estimate_binary_search(
                _, k2, k3, kn2, kn3, J, epsilon, N, int(max_steps / 4)
            )
            for _ in [0.2, 0.9]
        ]
    )
    a_th = 0.5
    k1_l, k1_r = k1_th[0], k1_th[1]
    a_l = calc_dwell_and_switching_times(
        k1_l, k2, k3, k1_l * epsilon, kn2, kn3, J, N, max_steps, True, **params
    )
    a_r = calc_dwell_and_switching_times(
        k1_r, k2, k3, k1_r * epsilon, kn2, kn3, J, N, max_steps, True, **params
    )

    nitr = 0
    while (
        onp.abs(a_l - a_th) > activity_tolerance
        or onp.abs(a_r - a_th) > activity_tolerance
    ):
        nitr += 1
        print(f"iteration {nitr}:")
        print("k1_l = {:.6f}, k1_r = {:.6f}".format(k1_l, k1_r))
        k1_m = (k1_l + k1_r) / 2
        a_m = calc_dwell_and_switching_times(
            k1_m, k2, k3, k1_m * epsilon, kn2, kn3, J, N, max_steps, True, **params
        )
        print("k1_m = {:.6f}, a_m = {:.2f}".format(k1_m, a_m))
        if a_m < a_th:
            k1_l = k1_m
            a_l = a_m
        else:
            k1_r = k1_m
            a_r = a_m
        if onp.abs(k1_l - k1_r) < k1_tolerance:
            break

    return (k1_l + k1_r) / 2
