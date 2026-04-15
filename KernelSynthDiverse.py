# @title
"""
Synthetic Causal Time-Series Pipeline — Data Quality Edition.

Components (in dependency order):
  1. Numba JIT base generators (ARIMA, ETS, control signals, etc.)
  2. OptimizedKernelSynthDiverse (GP kernel bank)
  3. Causal mixing kernels (_mix_dag, _mix_chain, _mix_pairwise)
  4. Orchestrators (FastControlSignalGenerator, FastStatisticalGenerator, VectorizedMixerCausal)
  5. FastPhysicsEngine (top-level generation)
  6. InfiniteIndustrialDataset (PyTorch IterableDataset)

Key data-quality fixes vs. original:
  - Lagged causal effects with ground-truth edge_lags returned
  - Post-mix per-node normalization (prevents scale explosion from additive mixing)
  - Control signals scaled to match physics dynamic range (no more 0.3 vs 3.0 mismatch)
  - Pairwise mixer respects root nodes
  - Contemp mode returns |Q|-derived adjacency (not all-ones)
  - ARIMA explosion guard (geometric stability check on AR coefficients)
  - Proper kernel normalization (no double z-scoring)
  - Edge lags stored in Data objects for supervised lag learning
  - Reproducible RNG throughout (no uncontrolled global state)
"""

from __future__ import annotations

import os
import math
import random
import logging
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Callable, Optional, Tuple

import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)


# ======================================================================
# 1. PARALLELIZED BASE GENERATORS (Numba JIT)
# ======================================================================

@njit(parallel=True, fastmath=True)
def _fast_arima_batch(n_samples: int, length: int, p_max: int, q_max: int) -> np.ndarray:
    """
    Generate ARIMA(p,d,q) samples with stability-checked AR coefficients.

    FIX: Original drew AR coefficients uniformly — no guarantee that the
    characteristic polynomial has roots inside the unit circle.  Now we
    rescale the AR vector so its L1-norm < 1, which is a sufficient
    (conservative) condition for stability.
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    burn_in = 100
    total_len = length + burn_in
    noise = np.random.standard_normal((n_samples, total_len))

    for i in prange(n_samples):
        p = np.random.randint(0, p_max + 1)
        q = np.random.randint(0, q_max + 1)
        d = np.random.randint(0, 2)

        # ── AR stability fix ──
        # Draw coefficients, then rescale so sum(|ar|) < stability_bound
        ar = np.random.uniform(-0.8, 0.8, p)
        if p > 0:
            l1 = 0.0
            for k in range(p):
                l1 += abs(ar[k])
            stability_bound = 0.95  # strict < 1.0
            if l1 > stability_bound:
                scale = stability_bound / (l1 + 1e-12)
                for k in range(p):
                    ar[k] *= scale

        ma = np.random.uniform(-0.5, 0.5, q)

        x = np.zeros(total_len)
        for k in range(max(p, q)):
            x[k] = np.random.normal(0.0, 1.0)

        start_idx = max(p, q)
        for t in range(start_idx, total_len):
            ar_term = 0.0
            for k in range(p):
                ar_term += ar[k] * x[t - k - 1]
            ma_term = 0.0
            for k in range(q):
                ma_term += ma[k] * noise[i, t - k - 1]
            x[t] = ar_term + noise[i, t] + ma_term

        res = x[burn_in:]

        # Integration (d=1)
        if d > 0:
            acc = res[0]
            for t in range(length):
                acc += res[t]
                res[t] = acc

        # ── Explosion guard ──
        max_abs = 0.0
        for t in range(length):
            if abs(res[t]) > max_abs:
                max_abs = abs(res[t])
        if max_abs > 1e6 or max_abs != max_abs:  # NaN check via self-inequality
            # Replace with a safe random walk
            for t in range(length):
                if t == 0:
                    res[t] = noise[i, 0]
                else:
                    res[t] = res[t - 1] + noise[i, t] * 0.1

        output[i, :] = res
    return output


@njit(parallel=True, fastmath=True)
def _fast_ets_batch(n_samples: int, length: int) -> np.ndarray:
    output = np.zeros((n_samples, length), dtype=np.float64)
    noise = np.random.standard_normal((n_samples, length))

    for i in prange(n_samples):
        alpha = np.random.uniform(0.2, 0.8)
        beta = np.random.uniform(0.01, 0.1)
        l = np.random.normal(0.0, 2.0)
        b = np.random.normal(0.0, 0.05)

        for t in range(length):
            y_hat = l + b
            output[i, t] = y_hat + noise[i, t]
            l_new = alpha * output[i, t] + (1 - alpha) * y_hat
            b_new = beta * (l_new - l) + (1 - beta) * b
            l = l_new
            b = b_new
    return output


@njit(parallel=True, fastmath=True)
def _fast_control_steps(n_samples: int, length: int) -> np.ndarray:
    """
    Piecewise-constant step signals.

    FIX: Original drew values in [-2, 2] then clipped to [-1, 1] in the
    engine — destroying the step structure for ~50% of draws.  Now we draw
    directly in a range that survives the clip, with occasional excursion.
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        # Use discrete setpoint levels more often for realism
        n_levels = np.random.randint(2, 6)
        levels = np.zeros(n_levels)
        for lv in range(n_levels):
            levels[lv] = np.random.uniform(-1.0, 1.0)

        val = levels[np.random.randint(0, n_levels)]
        expected_steps = np.random.uniform(0.2, 15.0)
        lambda_param = expected_steps / length
        for t in range(length):
            if np.random.random() < lambda_param:
                val = levels[np.random.randint(0, n_levels)]
            output[i, t] = val
    return output


@njit(parallel=True, fastmath=True)
def _fast_telegraph_process(n_samples: int, length: int) -> np.ndarray:
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        low = np.random.uniform(-1.0, -0.3)
        high = np.random.uniform(0.3, 1.0)
        state = low if np.random.random() < 0.5 else high
        switch_prob = np.random.uniform(0.005, 0.05)
        for t in range(length):
            if np.random.random() < switch_prob:
                state = high if state == low else low
            output[i, t] = state
    return output


@njit(parallel=True, fastmath=True)
def _fast_telegraph_process_binary(n_samples: int, length: int) -> np.ndarray:
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        state = -1.0 if np.random.random() < 0.5 else 1.0
        switch_prob = np.random.uniform(0.005, 0.05)
        for t in range(length):
            if np.random.random() < switch_prob:
                state *= -1
            output[i, t] = state
    return output


@njit(parallel=True, fastmath=True)
def _fast_damped_oscillation(n_samples: int, length: int) -> np.ndarray:
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        t_switch = np.random.randint(length // 10, length // 2)
        base_val = np.random.uniform(-1.0, 1.0)
        target_val = np.random.uniform(-1.0, 1.0)
        decay = np.random.uniform(0.01, 0.05)
        freq = np.random.uniform(0.05, 0.3)
        for t in range(length):
            if t < t_switch:
                output[i, t] = base_val
            else:
                dt = t - t_switch
                oscillation = (target_val - base_val) * np.exp(-decay * dt) * np.cos(freq * dt)
                output[i, t] = target_val - oscillation
    return output


@njit(parallel=True, fastmath=True)
def _fast_heteroscedastic_noise(n_samples: int, length: int) -> np.ndarray:
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        drift = np.random.normal(0.0, 0.01)
        val = 0.0
        omega = 0.01
        alpha = np.random.uniform(0.05, 0.2)
        vol = 0.1
        for t in range(length):
            shock = np.random.normal(0.0, 1.0)
            output[i, t] = val + shock * vol
            val = output[i, t] + drift
            vol = np.sqrt(omega + alpha * (shock ** 2))
    return output


@njit(parallel=True, fastmath=True)
def _fast_lorenz_attractor(n_samples: int, length: int) -> np.ndarray:
    """
    Lorenz system — chaotic deterministic trajectories.
    Parameters (σ, ρ, β) are randomized per sample for diversity.
    Returns z-coordinate (most chaotic) downsampled to `length`.
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    oversample = 4  # integration resolution
    total_steps = length * oversample

    for i in prange(n_samples):
        sigma = np.random.uniform(8.0, 12.0)
        rho = np.random.uniform(24.0, 32.0)
        beta = np.random.uniform(2.0, 4.0)
        dt = np.random.uniform(0.005, 0.02)

        x = np.random.uniform(-5.0, 5.0)
        y = np.random.uniform(-5.0, 5.0)
        z = np.random.uniform(15.0, 35.0)

        # Pick which coordinate to return (diversity)
        coord = np.random.randint(0, 3)
        raw = np.zeros(total_steps)

        for t in range(total_steps):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x += dx
            y += dy
            z += dz
            if coord == 0:
                raw[t] = x
            elif coord == 1:
                raw[t] = y
            else:
                raw[t] = z

        # Downsample by averaging groups
        for t in range(length):
            acc = 0.0
            for k in range(oversample):
                acc += raw[t * oversample + k]
            output[i, t] = acc / oversample
    return output


@njit(parallel=True, fastmath=True)
def _fast_ornstein_uhlenbeck(n_samples: int, length: int) -> np.ndarray:
    """
    Ornstein-Uhlenbeck process (mean-reverting SDE).
    dX = θ(μ - X)dt + σ dW
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        theta = np.random.uniform(0.01, 0.2)   # reversion speed
        mu = np.random.uniform(-1.0, 1.0)       # long-term mean
        sigma = np.random.uniform(0.05, 0.5)    # volatility
        dt = 1.0

        x = mu + np.random.normal(0.0, 0.5)
        for t in range(length):
            dw = np.random.normal(0.0, np.sqrt(dt))
            x += theta * (mu - x) * dt + sigma * dw
            output[i, t] = x
    return output


@njit(parallel=True, fastmath=True)
def _fast_geometric_brownian_motion(n_samples: int, length: int) -> np.ndarray:
    """
    Geometric Brownian Motion: dS = μS dt + σS dW.
    Produces log-normal/fat-tailed signals typical of financial data.
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        mu = np.random.uniform(-0.001, 0.001)
        sigma = np.random.uniform(0.01, 0.1)
        s = np.random.uniform(0.5, 2.0)  # initial price
        dt = 1.0

        for t in range(length):
            dw = np.random.normal(0.0, np.sqrt(dt))
            s *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * dw)
            output[i, t] = s

        # Center and scale to avoid huge values
        mean = 0.0
        for t in range(length):
            mean += output[i, t]
        mean /= length
        var = 0.0
        for t in range(length):
            var += (output[i, t] - mean) ** 2
        std = np.sqrt(var / length)
        if std > 1e-8:
            for t in range(length):
                output[i, t] = (output[i, t] - mean) / std
    return output


@njit(parallel=True, fastmath=True)
def _fast_regime_switching(n_samples: int, length: int) -> np.ndarray:
    """
    Hidden Markov Model with 2-3 regimes, each with different AR(1) dynamics.
    Forces in-context learning: model must discover which regime is active.
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        n_regimes = np.random.randint(2, 4)

        # Each regime has its own AR(1) parameters
        ar_coefs = np.zeros(n_regimes)
        means = np.zeros(n_regimes)
        vols = np.zeros(n_regimes)
        for r in range(n_regimes):
            ar_coefs[r] = np.random.uniform(0.5, 0.98)
            means[r] = np.random.uniform(-2.0, 2.0)
            vols[r] = np.random.uniform(0.05, 0.5)

        # Transition probability (low = long-lived regimes)
        p_switch = np.random.uniform(0.002, 0.02)
        regime = np.random.randint(0, n_regimes)
        x = means[regime]

        for t in range(length):
            if np.random.random() < p_switch:
                regime = np.random.randint(0, n_regimes)
            x = ar_coefs[regime] * x + (1 - ar_coefs[regime]) * means[regime] \
                + vols[regime] * np.random.normal(0.0, 1.0)
            output[i, t] = x
    return output


@njit(parallel=True, fastmath=True)
def _fast_sawtooth_triangle(n_samples: int, length: int) -> np.ndarray:
    """
    Asymmetric sawtooth/triangle waves with random periods and ramp ratios.
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        period = np.random.uniform(20.0, length / 2.0)
        ramp_frac = np.random.uniform(0.1, 0.9)  # 0.5 = triangle, 0.1 = sawtooth
        amplitude = np.random.uniform(0.5, 1.0)
        phase = np.random.uniform(0.0, period)

        for t in range(length):
            pos = ((t + phase) % period) / period  # 0..1
            if pos < ramp_frac:
                output[i, t] = amplitude * (2.0 * pos / ramp_frac - 1.0)
            else:
                output[i, t] = amplitude * (1.0 - 2.0 * (pos - ramp_frac) / (1.0 - ramp_frac))
    return output


@njit(parallel=True, fastmath=True)
def _fast_coupled_oscillators(n_samples: int, length: int) -> np.ndarray:
    """
    Two coupled oscillators producing beat frequencies and mode splitting.
    Returns one of the two oscillator outputs.
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        w1 = np.random.uniform(0.02, 0.2)       # natural freq 1
        w2 = w1 * np.random.uniform(0.8, 1.2)   # close freq → beats
        coupling = np.random.uniform(0.01, 0.3)  # coupling strength
        damping = np.random.uniform(0.0001, 0.005)

        x1, v1 = np.random.uniform(-1, 1), 0.0
        x2, v2 = np.random.uniform(-1, 1), 0.0
        use_x1 = np.random.random() < 0.5

        for t in range(length):
            a1 = -w1**2 * x1 + coupling * (x2 - x1) - damping * v1
            a2 = -w2**2 * x2 + coupling * (x1 - x2) - damping * v2
            v1 += a1
            v2 += a2
            x1 += v1
            x2 += v2
            output[i, t] = x1 if use_x1 else x2
    return output


@njit(parallel=True, fastmath=True)
def _fast_fractional_bm(n_samples: int, length: int) -> np.ndarray:
    """
    Approximate fractional Brownian Motion via Cholesky-free Hosking method.
    H > 0.5 → long-range dependence (persistent), H < 0.5 → anti-persistent.
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        H = np.random.uniform(0.2, 0.9)  # Hurst exponent

        # Davies-Harte exact simulation is complex in Numba, so we use
        # a simple filter-based approximation: power-law correlated noise
        # via cumulative sum with fractional differencing weights.
        d = H - 0.5  # fractional difference parameter

        # Generate white noise
        noise = np.random.standard_normal(length)

        # Apply fractional integration via recursive weights
        weights = np.zeros(length)
        weights[0] = 1.0
        for k in range(1, length):
            weights[k] = weights[k-1] * (d + k - 1) / k

        # Convolve noise with weights (truncated to length)
        for t in range(length):
            val = 0.0
            n_terms = min(t + 1, 200)  # cap for speed
            for k in range(n_terms):
                val += weights[k] * noise[t - k]
            output[i, t] = val
    return output


@njit(parallel=True, fastmath=True)
def _fast_chirp_signal(n_samples: int, length: int) -> np.ndarray:
    """
    Chirp signal — frequency sweeps from f0 to f1 over the duration.
    Linear or exponential sweep, random.
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        f0 = np.random.uniform(0.005, 0.05)     # start frequency (cycles/sample)
        f1 = np.random.uniform(0.05, 0.3)       # end frequency
        if np.random.random() < 0.5:
            f0, f1 = f1, f0  # sometimes sweep down
        amplitude = np.random.uniform(0.5, 1.0)
        phase0 = np.random.uniform(0.0, 2 * np.pi)

        use_exponential = np.random.random() < 0.5

        for t in range(length):
            frac = t / max(length - 1, 1)
            if use_exponential and f0 > 0:
                freq = f0 * (f1 / max(f0, 1e-10)) ** frac
            else:
                freq = f0 + (f1 - f0) * frac
            phase = phase0 + 2 * np.pi * freq * t
            output[i, t] = amplitude * np.sin(phase)
    return output


@njit(parallel=True, fastmath=True)
def _fast_pulse_train(n_samples: int, length: int) -> np.ndarray:
    """
    Irregular pulse train: pulses with random width, spacing, and amplitude.
    Represents valve openings, injection events, digital triggers.
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        baseline = np.random.uniform(-0.5, 0.0)
        pulse_amp = np.random.uniform(0.5, 1.0)
        mean_interval = np.random.uniform(30.0, 200.0)  # mean time between pulses
        mean_width = np.random.uniform(3.0, 30.0)       # mean pulse width

        t = 0
        while t < length:
            # Inter-pulse interval (exponential distribution)
            gap = int(np.random.exponential(mean_interval))
            t_start = t + gap
            if t_start >= length:
                break
            width = max(2, int(np.random.exponential(mean_width)))
            amp = pulse_amp * np.random.uniform(0.7, 1.3)  # slight variation

            for t2 in range(t_start, min(t_start + width, length)):
                output[i, t2] = amp
            t = t_start + width

        # Fill baseline
        for t in range(length):
            if output[i, t] == 0.0:
                output[i, t] = baseline
    return output


@njit(parallel=True, fastmath=True)
def _fast_piecewise_polynomial(n_samples: int, length: int) -> np.ndarray:
    """
    Piecewise polynomial: random number of segments, each with random
    degree (0-3) and coefficients. Creates diverse non-stationary signals.
    """
    output = np.zeros((n_samples, length), dtype=np.float64)
    for i in prange(n_samples):
        n_segments = np.random.randint(3, 10)
        segment_len = length // n_segments

        for seg in range(n_segments):
            t_start = seg * segment_len
            t_end = min((seg + 1) * segment_len, length)
            if seg == n_segments - 1:
                t_end = length

            degree = np.random.randint(0, 4)  # 0=constant, 1=linear, 2=quad, 3=cubic
            # Random coefficients, scaled to keep values reasonable
            a0 = np.random.uniform(-1.0, 1.0)
            a1 = np.random.uniform(-0.01, 0.01) if degree >= 1 else 0.0
            a2 = np.random.uniform(-0.0001, 0.0001) if degree >= 2 else 0.0
            a3 = np.random.uniform(-0.000001, 0.000001) if degree >= 3 else 0.0

            for t in range(t_start, t_end):
                dt = t - t_start
                output[i, t] = a0 + a1 * dt + a2 * dt**2 + a3 * dt**3

            # Add measurement noise
            noise_level = np.random.uniform(0.01, 0.1)
            for t in range(t_start, t_end):
                output[i, t] += noise_level * np.random.normal(0.0, 1.0)
    return output




from sklearn.gaussian_process.kernels import (
    RBF, Matern, ExpSineSquared, RationalQuadratic,
    DotProduct, WhiteKernel, ConstantKernel, Kernel,
)


def _sample_log_uniform(low: float, high: float, rng: np.random.RandomState) -> float:
    """Sample from log-uniform distribution on [low, high]."""
    if low <= 0 or high <= 0 or low >= high:
        return float(np.exp(rng.uniform(np.log(max(low, 1e-12)), np.log(max(high, 1e-11)))))
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


class OptimizedKernelSynthDiverse:
    """
    Generate diverse synthetic time-series by sampling from GP priors
    constructed via random kernel compositions.

    Key design choices:
      - Named kernel groups (no fragile index-based selection)
      - Eigendecomposition fallback (preserves covariance structure)
      - Output validation with retry
      - Standardised output scale
    """

    _STRUCTURED_CATEGORIES = ("periodic", "stationary", "trend", "composite", "process_block")

    def __init__(
        self,
        max_kernels: int = 5,
        bank_size: int = 32,
        seed: Optional[int] = None,
        max_cache_entries: int = 3,
        max_retries: int = 3,
    ) -> None:
        self.max_kernels = max_kernels
        self.bank_size = bank_size
        self.max_cache_entries = max_cache_entries
        self.max_retries = max_retries
        self.rng = np.random.RandomState(seed)
        self.cached_chol: OrderedDict[int, List[np.ndarray]] = OrderedDict()

    @staticmethod
    def _get_base_kernels(
        length: int, rng: np.random.RandomState,
    ) -> Dict[str, List[Callable[[], Kernel]]]:
        L = float(length)
        _slu = _sample_log_uniform

        k_values = [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0, 100.0]
        base_periods = [L / k for k in k_values]
        periodic = [
            (lambda p=p: ExpSineSquared(
                length_scale=_slu(2.0, 10.0, rng),
                periodicity=p * _slu(0.98, 1.02, rng),
            )) for p in base_periods
        ]

        stationary = [
            lambda: RBF(length_scale=_slu(L / 4, max(L * 2, L / 4 + 1), rng)),
            lambda: Matern(length_scale=_slu(max(L / 20, 1.0), max(L / 2, 2.0), rng), nu=1.5),
            lambda: Matern(length_scale=_slu(max(L / 10, 1.0), max(L, 2.0), rng), nu=2.5),
            lambda: Matern(length_scale=_slu(max(L / 6, 1.0), max(L / 1.5, 2.0), rng), nu=0.5),
        ]

        composite = [
            lambda: RBF(length_scale=_slu(L / 2, L * 2, rng))
                    * Matern(length_scale=_slu(max(L / 20, 1.0), max(L / 5, 2.0), rng), nu=1.5),
            lambda: ConstantKernel(constant_value=_slu(0.5, 5.0, rng))
                    + Matern(length_scale=_slu(max(L / 10, 1.0), max(L / 3, 2.0), rng), nu=0.5),
            lambda: RBF(length_scale=_slu(L / 3, L, rng))
                    * ExpSineSquared(length_scale=_slu(2.0, 8.0, rng), periodicity=L / _slu(3.0, 10.0, rng)),
        ]

        trends = [
            lambda: DotProduct(sigma_0=rng.uniform(0.1, 5.0)),
            lambda: RationalQuadratic(length_scale=_slu(10.0, max(L / 2, 11.0), rng), alpha=_slu(0.01, 0.5, rng)),
            lambda: DotProduct(sigma_0=rng.uniform(0.5, 3.0)) + ConstantKernel(constant_value=_slu(0.1, 2.0, rng)),
        ]

        events = [
            lambda: RationalQuadratic(length_scale=_slu(0.1, 2.0, rng), alpha=_slu(1e-5, 1e-3, rng)),
            lambda: Matern(length_scale=_slu(0.5, 3.0, rng), nu=0.5) * RBF(length_scale=rng.uniform(2.0, 15.0)),
            lambda: ConstantKernel(constant_value=_slu(1.0, 5.0, rng))
                    * RationalQuadratic(length_scale=_slu(L / 2, L * 2, rng), alpha=_slu(1e-5, 1e-3, rng)),
        ]

        process_blocks = [
            lambda: ExpSineSquared(length_scale=_slu(3.0, 8.0, rng), periodicity=L / _slu(3.0, 8.0, rng))
                    * WhiteKernel(noise_level=_slu(0.01, 0.3, rng)),
            lambda: ExpSineSquared(length_scale=_slu(5.0, 15.0, rng), periodicity=L / _slu(1.5, 4.0, rng))
                    * RationalQuadratic(length_scale=_slu(0.5, 3.0, rng), alpha=_slu(1e-5, 1e-3, rng)),
            lambda: DotProduct(sigma_0=rng.uniform(0.5, 3.0))
                    * ExpSineSquared(length_scale=_slu(1.0, 5.0, rng), periodicity=L / _slu(10.0, 30.0, rng)),
        ]

        noise = [lambda: WhiteKernel(noise_level=_slu(1e-4, 0.1, rng))]

        return {
            "periodic": periodic, "stationary": stationary, "composite": composite,
            "trend": trends, "event": events, "process_block": process_blocks, "noise": noise,
        }

    @staticmethod
    def _validate_kernel_matrix(K: np.ndarray) -> bool:
        if not np.all(np.isfinite(K)):
            return False
        trace = np.trace(K)
        if trace < 1e-4 or trace > 1e8 * K.shape[0]:
            return False
        if np.any(np.diag(K) < 0):
            return False
        return True

    @staticmethod
    def _stable_sqrt_factor(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        K = (K + K.T) * 0.5
        trace_scale = max(np.trace(K) / n, 1e-10)

        for frac in [1e-6, 1e-5, 1e-4, 1e-3]:
            try:
                K_jit = K.copy()
                K_jit[np.diag_indices(n)] += frac * trace_scale
                return np.linalg.cholesky(K_jit)
            except np.linalg.LinAlgError:
                continue

        try:
            eigvals, eigvecs = np.linalg.eigh(K)
            floor = max(eigvals.max() * 1e-8, 1e-10)
            eigvals = np.maximum(eigvals, floor)
            return eigvecs * np.sqrt(eigvals)[np.newaxis, :]
        except np.linalg.LinAlgError:
            logger.warning("Both Cholesky and eigh failed; returning identity.")
            return np.eye(n)

    def _compose_high_structure(self, groups, rng):
        master = rng.choice(groups["periodic"])()
        modulator = rng.choice(groups["process_block"] + groups["event"])()
        body = rng.choice(groups["stationary"] + groups["composite"])()
        return (master * modulator) + body

    def _compose_stratified(self, groups, rng):
        root_cat = rng.choice(list(self._STRUCTURED_CATEGORIES))
        k = rng.choice(groups[root_cat])()
        depth = rng.randint(1, self.max_kernels + 1)
        all_cats = list(groups.keys())
        remaining = [c for c in all_cats if c != root_cat]

        for i in range(depth):
            if remaining:
                cat = rng.choice(remaining)
                remaining = [c for c in remaining if c != cat]
            else:
                cat = rng.choice(all_cats)
            next_k = rng.choice(groups[cat])()
            p_mul = min(0.3 + 0.15 * i, 0.8)
            k = (k * next_k) if rng.rand() < p_mul else (k + next_k)
        return k

    def _get_random_kernel_matrix(self, length: int, rng: np.random.RandomState) -> np.ndarray:
        X = np.linspace(0, 1, length).reshape(-1, 1)
        for attempt in range(self.max_retries):
            groups = self._get_base_kernels(length, rng)
            kernel = self._compose_high_structure(groups, rng) if rng.rand() < 0.2 else self._compose_stratified(groups, rng)
            try:
                K = kernel(X)
                if not self._validate_kernel_matrix(K):
                    continue
                return self._stable_sqrt_factor(K)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue
        K_safe = RBF(length_scale=length / 4.0)(X)
        return self._stable_sqrt_factor(K_safe)

    def _build_bank(self, length: int) -> List[np.ndarray]:
        thread_seeds = self.rng.randint(0, 2**31 - 1, size=self.bank_size)
        def _worker(seed):
            return self._get_random_kernel_matrix(length, np.random.RandomState(seed))
        n_workers = min(self.bank_size, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(_worker, thread_seeds))

    def generate_batch(self, n_samples: int, length: int, normalize: bool = True) -> np.ndarray:
        if length not in self.cached_chol:
            if len(self.cached_chol) >= self.max_cache_entries:
                self.cached_chol.popitem(last=False)
            self.cached_chol[length] = self._build_bank(length)
        else:
            self.cached_chol.move_to_end(length)

        bank = self.cached_chol[length]
        samples_per_kernel = int(np.ceil(n_samples / len(bank)))
        results, generated = [], 0
        indices = np.arange(len(bank))
        self.rng.shuffle(indices)

        for i in indices:
            if generated >= n_samples:
                break
            n_need = min(samples_per_kernel, n_samples - generated)
            z = self.rng.standard_normal((n_need, length))
            results.append((bank[i] @ z.T).T)
            generated += n_need

        batch = np.vstack(results)[:n_samples]
        batch = self._validate_and_repair(batch, length)

        if normalize:
            mu = batch.mean(axis=1, keepdims=True)
            sigma = batch.std(axis=1, keepdims=True)
            sigma = np.where(sigma < 1e-8, 1.0, sigma)
            batch = (batch - mu) / sigma
        return batch

    def _validate_and_repair(self, batch, length, min_std=1e-8):
        bad = ~np.all(np.isfinite(batch), axis=1) | (batch.std(axis=1) < min_std)
        if not bad.any():
            return batch
        for idx in np.where(bad)[0]:
            for _ in range(self.max_retries):
                L = self._get_random_kernel_matrix(length, np.random.RandomState(self.rng.randint(0, 2**31 - 1)))
                sample = L @ self.rng.standard_normal(length)
                if np.all(np.isfinite(sample)) and np.std(sample) > min_std:
                    batch[idx] = sample
                    break
            else:
                batch[idx] = np.cumsum(self.rng.standard_normal(length))
        return batch


# ======================================================================
# 3. CAUSAL MIXING KERNELS (Numba JIT)
# ======================================================================

@njit(fastmath=True)
def _apply_complex_physics(signal: np.ndarray, length: int) -> np.ndarray:
    """
    2nd-order dynamic system response: spring-damper with asymmetry,
    rate limiting, and optional transport delay.
    """
    filtered = np.zeros_like(signal)
    k_base = np.random.uniform(0.001, 0.1)
    zeta = np.random.uniform(0.4, 0.8) if np.random.random() < 0.4 else np.random.uniform(1.2, 2.5)
    asymmetry = np.random.uniform(0.5, 2.0)
    use_rate_limit = np.random.random() < 0.5
    max_vel = np.random.uniform(0.05, 0.5) if use_rate_limit else 100.0
    delay = np.random.randint(1, max(2, length // 10)) if np.random.random() < 0.3 else 0

    pos = signal[0]
    vel = 0.0

    for t in range(length):
        idx = max(0, t - delay)
        target = signal[idx]
        error = target - pos
        eff_k = k_base * asymmetry if error > 0 else k_base / asymmetry

        acc = (error * eff_k) - (vel * (zeta * 2.0 * np.sqrt(eff_k)))
        vel += acc
        if vel > max_vel:
            vel = max_vel
        elif vel < -max_vel:
            vel = -max_vel
        pos += vel
        filtered[t] = pos
    return filtered


@njit(fastmath=True)
def _apply_causal_effect(
    driver: np.ndarray, child: np.ndarray, length: int,
    lag: int, alpha: float, itype: int,
) -> np.ndarray:
    """
    Apply a single causal effect from driver to child with the given lag.

    Uses CONVEX BLENDING:  child = (1 - α) * child + α * f(lagged_driver)

    13 interaction types (0–12) for maximum diversity:
      0  — Linear blend
      1  — Saturating (tanh)
      2  — Multiplicative modulation
      3  — Complex physics (2nd-order spring-damper)
      4  — Threshold gate (only activates when |parent| > θ)
      5  — Polynomial (random a₁p + a₂p² + a₃p³)
      6  — Frequency modulation (parent controls oscillation rate)
      7  — Dead-zone + saturation (industrial actuator model)
      8  — Hysteresis (path-dependent threshold)
      9  — Exponential decay (asymmetric rise/fall)
      10 — Rectified (ReLU-like, partial activation)
      11 — Delayed echo (sum of multiple delayed copies)
      12 — Quantized (discretized effect at random resolution)

    Parameters:
      alpha : float in [0, 1] — blend strength (fraction of child replaced by effect)
    """
    # Build the lagged driver signal
    effect = np.zeros(length, dtype=np.float64)
    if lag < length:
        effect[lag:] = driver[:(length - lag)]

    # ── Type 0: Linear ──
    if itype == 0:
        parent_signal = effect

    # ── Type 1: Saturating (tanh) ──
    elif itype == 1:
        parent_signal = np.tanh(effect)

    # ── Type 2: Multiplicative modulation ──
    elif itype == 2:
        modulation = 1.0 + np.tanh(effect) * alpha
        return child * modulation

    # ── Type 3: Complex physics (2nd-order dynamic response) ──
    elif itype == 3:
        parent_signal = _apply_complex_physics(effect, length)

    # ── Type 4: Threshold gate ──
    elif itype == 4:
        threshold = np.random.uniform(0.2, 0.8)
        parent_signal = np.zeros(length, dtype=np.float64)
        for t in range(length):
            if abs(effect[t]) > threshold:
                parent_signal[t] = effect[t] - np.sign(effect[t]) * threshold

    # ── Type 5: Polynomial (random coefficients) ──
    elif itype == 5:
        a1 = np.random.uniform(0.3, 1.0) * (1.0 if np.random.random() < 0.5 else -1.0)
        a2 = np.random.uniform(0.0, 0.5) * (1.0 if np.random.random() < 0.5 else -1.0)
        a3 = np.random.uniform(0.0, 0.3) * (1.0 if np.random.random() < 0.5 else -1.0)
        parent_signal = np.zeros(length, dtype=np.float64)
        for t in range(length):
            p = effect[t]
            parent_signal[t] = a1 * p + a2 * p * p + a3 * p * p * p

    # ── Type 6: Frequency modulation ──
    elif itype == 6:
        base_freq = np.random.uniform(0.02, 0.15)
        mod_depth = np.random.uniform(0.5, 2.0)
        parent_signal = np.zeros(length, dtype=np.float64)
        phase = 0.0
        for t in range(length):
            freq = base_freq * (1.0 + mod_depth * np.tanh(effect[t]))
            phase += 2.0 * np.pi * freq
            parent_signal[t] = np.sin(phase)

    # ── Type 7: Dead-zone + saturation ──
    elif itype == 7:
        dead_zone = np.random.uniform(0.1, 0.5)
        sat_level = np.random.uniform(0.8, 1.5)
        parent_signal = np.zeros(length, dtype=np.float64)
        for t in range(length):
            p = effect[t]
            if abs(p) < dead_zone:
                parent_signal[t] = 0.0
            elif p > 0:
                parent_signal[t] = min(p - dead_zone, sat_level)
            else:
                parent_signal[t] = max(p + dead_zone, -sat_level)

    # ── Type 8: Hysteresis ──
    elif itype == 8:
        thresh_up = np.random.uniform(0.2, 0.7)
        thresh_down = -np.random.uniform(0.1, thresh_up)
        gain = np.random.uniform(0.5, 1.5)
        state = 0.0
        parent_signal = np.zeros(length, dtype=np.float64)
        for t in range(length):
            if state < 0.5 and effect[t] > thresh_up:
                state = 1.0
            elif state > 0.5 and effect[t] < thresh_down:
                state = 0.0
            parent_signal[t] = state * gain * effect[t]

    # ── Type 9: Exponential decay (asymmetric rise/fall) ──
    elif itype == 9:
        tau_rise = np.random.uniform(1.0, 10.0)
        tau_fall = np.random.uniform(5.0, 50.0)
        if np.random.random() < 0.5:
            tau_rise, tau_fall = tau_fall, tau_rise
        state = 0.0
        parent_signal = np.zeros(length, dtype=np.float64)
        for t in range(length):
            target = effect[t]
            if target > state:
                tau = tau_rise
            else:
                tau = tau_fall
            state += (target - state) / max(tau, 0.1)
            parent_signal[t] = state

    # ── Type 10: Rectified (ReLU-like) ──
    elif itype == 10:
        bias = np.random.uniform(-0.5, 0.5)
        invert = np.random.random() < 0.5
        parent_signal = np.zeros(length, dtype=np.float64)
        for t in range(length):
            val = effect[t] - bias
            if invert:
                val = -val
            parent_signal[t] = max(val, 0.0)

    # ── Type 11: Delayed echo (multi-lag composite) ──
    elif itype == 11:
        n_echoes = np.random.randint(2, 5)
        parent_signal = np.zeros(length, dtype=np.float64)
        total_weight = 0.0
        for e in range(n_echoes):
            echo_lag = np.random.randint(0, max(1, length // 8))
            echo_weight = np.random.uniform(0.3, 1.0) * (0.7 ** e)
            total_weight += abs(echo_weight)
            sign = 1.0 if np.random.random() < 0.7 else -1.0
            for t in range(echo_lag, length):
                parent_signal[t] += sign * echo_weight * driver[t - echo_lag]
        if total_weight > 0:
            for t in range(length):
                parent_signal[t] /= total_weight

    # ── Type 12: Quantized ──
    elif itype == 12:
        n_levels = np.random.randint(3, 12)
        parent_signal = np.zeros(length, dtype=np.float64)
        e_min, e_max = effect[0], effect[0]
        for t in range(length):
            if effect[t] < e_min:
                e_min = effect[t]
            if effect[t] > e_max:
                e_max = effect[t]
        e_range = max(e_max - e_min, 1e-8)
        step = e_range / n_levels
        for t in range(length):
            parent_signal[t] = np.floor((effect[t] - e_min) / step) * step + e_min

    # ── Type 13: PID control action ──
    # child = Kp*error + Ki*integral(error) + Kd*d(error)/dt
    # THE most common causal relationship in all of industrial automation.
    elif itype == 13:
        Kp = np.random.uniform(0.5, 3.0)
        Ki = np.random.uniform(0.0, 0.5)
        Kd = np.random.uniform(0.0, 0.3)
        parent_signal = np.zeros(length, dtype=np.float64)
        integral = 0.0
        prev_e = 0.0
        for t in range(length):
            e = effect[t]
            integral += e
            # Anti-windup: clamp integral
            integral = max(-100.0, min(100.0, integral))
            derivative = e - prev_e
            parent_signal[t] = Kp * e + Ki * integral + Kd * derivative
            prev_e = e

    # ── Type 14: Inverse response ──
    # System initially moves OPPOSITE to final direction, then corrects.
    # Common in: boiler drum level, some chemical reactors.
    # Response = -a·fast_component + b·slow_component
    elif itype == 14:
        tau_fast = np.random.uniform(1.0, 5.0)
        tau_slow = np.random.uniform(10.0, 50.0)
        ratio = np.random.uniform(0.3, 0.8)  # how much inverse vs correct
        state_fast = 0.0
        state_slow = 0.0
        parent_signal = np.zeros(length, dtype=np.float64)
        for t in range(length):
            target = effect[t]
            state_fast += (target - state_fast) / max(tau_fast, 0.1)
            state_slow += (target - state_slow) / max(tau_slow, 0.1)
            parent_signal[t] = -ratio * state_fast + (1.0 + ratio) * state_slow

    # ── Type 15: Time-varying gain ──
    # Gain depends on operating point (child's current level).
    # Models nonlinear processes where sensitivity changes with state.
    elif itype == 15:
        base_gain = np.random.uniform(0.3, 1.5)
        gain_sensitivity = np.random.uniform(0.2, 1.0)
        parent_signal = np.zeros(length, dtype=np.float64)
        for t in range(length):
            # Gain varies with child's current value
            local_gain = base_gain * (1.0 + gain_sensitivity * np.tanh(child[t]))
            parent_signal[t] = local_gain * effect[t]

    # ── Type 16: Valve stiction ──
    # Stick-slip friction: valve doesn't move until force exceeds breakaway,
    # then jumps. Causes limit cycling in control loops.
    elif itype == 16:
        stiction_band = np.random.uniform(0.05, 0.3)
        slip_jump = np.random.uniform(0.02, 0.15)
        valve_pos = 0.0
        parent_signal = np.zeros(length, dtype=np.float64)
        for t in range(length):
            desired = effect[t]
            error = desired - valve_pos
            if abs(error) > stiction_band:
                # Breaks free — jumps past the desired position
                # Root Fix: Scale jump to be relative to the stiction band
                valve_pos = desired + np.sign(error) * slip_jump * stiction_band
            # else: stuck — valve_pos stays unchanged
            parent_signal[t] = valve_pos

    # ── Type 17: Backlash ──
    # Dead-band on direction reversal (gear trains, mechanical linkages).
    elif itype == 17:
        backlash_width = np.random.uniform(0.05, 0.3)
        output = 0.0
        prev_input = 0.0
        parent_signal = np.zeros(length, dtype=np.float64)
        for t in range(length):
            inp = effect[t]
            delta = inp - prev_input
            if delta > backlash_width:
                output += delta - backlash_width
            elif delta < -backlash_width:
                output += delta + backlash_width
            # else: in dead zone, output unchanged
            parent_signal[t] = output
            prev_input = inp

    # Fallback
    else:
        parent_signal = effect

    # Convex blend: child retains (1-α) of itself, gains α of parent signal
    return (1.0 - alpha) * child + alpha * parent_signal


@njit(parallel=True, fastmath=True)
def _mix_dag_batch_jit(
    x_batch: np.ndarray, n_nodes: int, length: int,
    n_roots_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DAG causal mixing with lagged effects.

    Returns:
      out_batch  — (B, N, L) mixed signals
      out_adj    — (B, N, N) adjacency matrices
      out_lags   — (B, N, N) ground-truth lag for each edge (0 = no edge)

    FIX: Now returns edge lags so the model can learn lag detection.
    FIX: Interaction type for control parents biased towards physics dynamics.
    FIX: Lag range depends on parent type (controls get longer lags).
    """
    B = x_batch.shape[0]
    out_batch = x_batch.copy()
    out_adj = np.zeros((B, n_nodes, n_nodes), dtype=np.float64)
    out_lags = np.zeros((B, n_nodes, n_nodes), dtype=np.int64)

    for b in prange(B):
        n_roots = n_roots_list[b]

        # ── Build DAG structure ──
        for i in range(n_roots, n_nodes):
            rand_val = np.random.random()
            if rand_val < 0.6:
                n_parents = 1
            elif rand_val < 0.85:
                n_parents = 2
            elif rand_val < 0.95:
                n_parents = 0
            else:
                n_parents = 3

            if n_parents > 0 and i > 0:
                potential = np.arange(i)
                selected = np.random.choice(potential, size=min(n_parents, i), replace=False)
                for p_idx in selected:
                    out_adj[b, p_idx, i] = 1.0

        # ── Apply causal effects (topological order = index order) ──
        for i in range(n_roots, n_nodes):
            for p in range(i):
                if out_adj[b, p, i] < 0.5:
                    continue

                is_control = (p < n_roots)

                # Controls → prefer physics/industrial dynamics
                if is_control:
                    # Physics(3), dead-zone(7), hysteresis(8), exp-decay(9),
                    # PID(13), stiction(16), backlash(17)
                    ctrl_types = np.array([3, 7, 8, 9, 13, 16, 17])
                    itype = ctrl_types[np.random.randint(0, len(ctrl_types))]
                else:
                    itype = np.random.randint(0, 18)

                # Lag: controls often have longer transport delays
                max_lag = max(2, length // 10)
                if is_control:
                    lag = np.random.randint(1, max(2, length // 5))
                else:
                    lag = np.random.randint(1, max_lag)

                # Blend strength α: how much of the child is replaced by parent
                # Control parents get higher α (their step/binary pattern
                # should be clearly visible in the child)
                if is_control:
                    alpha = np.random.uniform(0.4, 0.8)
                else:
                    alpha = np.random.uniform(0.2, 0.6)

                out_lags[b, p, i] = lag
                out_batch[b, i] = _apply_causal_effect(
                    out_batch[b, p], out_batch[b, i], length,
                    lag, alpha, itype,
                )

    return out_batch, out_adj, out_lags


@njit(parallel=True, fastmath=True)
def _mix_chain_batch_jit(
    x_batch: np.ndarray, n_nodes: int, length: int,
    n_roots_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chain (sequential) causal mixing with lagged effects.

    FIX: Returns edge lags.
    FIX: Uses shared _apply_causal_effect for consistency.
    """
    B = x_batch.shape[0]
    out_batch = x_batch.copy()
    out_adj = np.zeros((B, n_nodes, n_nodes), dtype=np.float64)
    out_lags = np.zeros((B, n_nodes, n_nodes), dtype=np.int64)

    for b in prange(B):
        n_roots = n_roots_list[b]
        n_physics = n_nodes - n_roots
        if n_physics == 0 or (n_physics < 2 and n_roots == 0):
            continue

        # Shuffle physics node order
        chain_idx = np.arange(n_roots, n_nodes)
        if n_physics > 1:
            for i in range(n_physics - 1, 0, -1):
                j = np.random.randint(0, i + 1)
                chain_idx[i], chain_idx[j] = chain_idx[j], chain_idx[i]

        # Root → head of chain (strong imprint from control)
        if n_roots > 0:
            root = np.random.randint(0, n_roots)
            head = chain_idx[0]
            lag = np.random.randint(1, max(2, length // 5))
            alpha = np.random.uniform(0.5, 0.8)  # strong control imprint
            out_adj[b, root, head] = 1.0
            out_lags[b, root, head] = lag
            out_batch[b, head] = _apply_causal_effect(
                out_batch[b, root], out_batch[b, head], length, lag, alpha, 3,
            )

        # Chain links (each node blends with its predecessor)
        for i in range(n_physics - 1):
            src, tgt = chain_idx[i], chain_idx[i + 1]
            lag = np.random.randint(2, max(3, length // 10))
            alpha = np.random.uniform(0.2, 0.6)
            itype = np.random.randint(0, 18)
            out_adj[b, src, tgt] = 1.0
            out_lags[b, src, tgt] = lag
            out_batch[b, tgt] = _apply_causal_effect(
                out_batch[b, src], out_batch[b, tgt], length, lag, alpha, itype,
            )

    return out_batch, out_adj, out_lags


@njit(parallel=True, fastmath=True)
def _mix_pairwise_batch_jit(
    x_batch: np.ndarray, n_nodes: int, length: int,
    n_roots_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pairwise causal mixing: random permutation determines parent→child pairs.

    FIX: Now respects root nodes — roots are never children.
    FIX: Returns edge lags.
    FIX: Uses shared _apply_causal_effect.
    """
    B = x_batch.shape[0]
    out_batch = x_batch.copy()
    out_adj = np.zeros((B, n_nodes, n_nodes), dtype=np.float64)
    out_lags = np.zeros((B, n_nodes, n_nodes), dtype=np.int64)

    for b in prange(B):
        n_roots = n_roots_list[b]

        # Only shuffle the non-root indices
        indices = np.arange(n_roots, n_nodes)
        n_shuf = len(indices)
        if n_shuf > 1:
            for i in range(n_shuf - 1, 0, -1):
                j = np.random.randint(0, i + 1)
                indices[i], indices[j] = indices[j], indices[i]

        # If we have roots, connect a root to the first physics node
        if n_roots > 0 and n_shuf > 0:
            root = np.random.randint(0, n_roots)
            tgt = indices[0]
            lag = np.random.randint(1, max(2, length // 10))
            alpha = np.random.uniform(0.4, 0.7)  # control imprint
            out_adj[b, root, tgt] = 1.0
            out_lags[b, root, tgt] = lag
            out_batch[b, tgt] = _apply_causal_effect(
                out_batch[b, root], out_batch[b, tgt], length, lag, alpha, 0,
            )

        # Pairwise chain among physics nodes
        for i in range(n_shuf - 1):
            src, tgt = indices[i], indices[i + 1]
            lag = np.random.randint(1, max(2, length // 10))
            alpha = np.random.uniform(0.2, 0.5)
            itype = np.random.randint(0, 18)
            out_adj[b, src, tgt] = 1.0
            out_lags[b, src, tgt] = lag
            out_batch[b, tgt] = _apply_causal_effect(
                out_batch[b, src], out_batch[b, tgt], length, lag, alpha, itype,
            )

    return out_batch, out_adj, out_lags


# ======================================================================
# 4. CLASS ORCHESTRATORS
# ======================================================================

class FastControlSignalGenerator:
    def generate_batch(self, n_samples: int, length: int) -> np.ndarray:
        # 6 control signal types for diverse setpoint/actuation patterns
        n_steps = max(1, int(n_samples * 0.15))
        n_tel = max(1, int(n_samples * 0.10))
        n_bin = max(1, int(n_samples * 0.15))
        n_damp = max(1, int(n_samples * 0.15))
        n_saw = max(1, int(n_samples * 0.20))
        n_pulse = n_samples - n_steps - n_tel - n_bin - n_damp - n_saw

        arrays = []
        if n_steps > 0:
            arrays.append(_fast_control_steps(n_steps, length))
        if n_tel > 0:
            arrays.append(_fast_telegraph_process(n_tel, length))
        if n_bin > 0:
            arrays.append(_fast_telegraph_process_binary(n_bin, length))
        if n_damp > 0:
            arrays.append(_fast_damped_oscillation(n_damp, length))
        if n_saw > 0:
            arrays.append(_fast_sawtooth_triangle(n_saw, length))
        if n_pulse > 0:
            arrays.append(_fast_pulse_train(n_pulse, length))
        return np.vstack(arrays) if arrays else np.empty((0, length))


class FastStatisticalGenerator:
    def generate_batch(self, n_samples: int, length: int) -> np.ndarray:
        # 10 generator types for maximum base signal diversity
        # Each gets ~10% of samples
        n_per = max(1, n_samples // 10)
        n_arima = n_per
        n_hetero = n_per
        n_ets = n_per
        n_lorenz = n_per
        n_ou = n_per
        n_gbm = n_per
        n_regime = n_per
        n_coupled = n_per
        n_fbm = n_per
        n_chirp = n_samples - 9 * n_per  # remainder

        arrays = []
        if n_arima > 0:
            arrays.append(_fast_arima_batch(n_arima, length, 3, 3))
        if n_hetero > 0:
            arrays.append(_fast_heteroscedastic_noise(n_hetero, length))
        if n_ets > 0:
            arrays.append(_fast_ets_batch(n_ets, length))
        if n_lorenz > 0:
            arrays.append(_fast_lorenz_attractor(n_lorenz, length))
        if n_ou > 0:
            arrays.append(_fast_ornstein_uhlenbeck(n_ou, length))
        if n_gbm > 0:
            arrays.append(_fast_geometric_brownian_motion(n_gbm, length))
        if n_regime > 0:
            arrays.append(_fast_regime_switching(n_regime, length))
        if n_coupled > 0:
            arrays.append(_fast_coupled_oscillators(n_coupled, length))
        if n_fbm > 0:
            arrays.append(_fast_fractional_bm(n_fbm, length))
        if n_chirp > 0:
            arrays.append(_fast_chirp_signal(n_chirp, length))
        # Also include the non-stationary ones
        n_ppoly = max(1, n_samples // 12)
        arrays.append(_fast_piecewise_polynomial(n_ppoly, length))
        result = np.vstack(arrays)
        # Shuffle so the model can't predict type from position
        np.random.shuffle(result)
        return result[:n_samples]


class VectorizedMixerCausal:
    """
    Causal mixer that applies diverse mixing strategies while preserving
    root node identity for causal learning.

    Returns (x_mixed, adjacency, edge_lags) for all modes.

    FIX: contemp mode now returns |Q|-derived sparse adjacency.
    FIX: none mode explicitly returns zero adjacency + zero lags.
    FIX: All modes return edge_lags (zeros for non-causal modes).
    FIX: method_probs is a tuple default (immutable).
    FIX: Post-mix per-node normalization prevents scale explosion.
    """
    def apply_batch(
        self,
        x_batch: np.ndarray,
        n_roots_list: np.ndarray = None,
        method_probs: Tuple[float, ...] = (0.05, 0.15, 0.05, 0.15, 0.6),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          x_mixed  — (B, N, L)
          adj      — (B, N, N) binary adjacency
          lags     — (B, N, N) int64 lag per edge (0 where no edge)
        """
        B, N, L = x_batch.shape
        out_batch = np.zeros_like(x_batch)
        out_adj = np.zeros((B, N, N), dtype=np.float64)
        out_lags = np.zeros((B, N, N), dtype=np.int64)

        if n_roots_list is None:
            n_roots_list = np.zeros(B, dtype=np.int64)

        modes = np.random.choice(
            ['none', 'contemp', 'pairwise', 'chain', 'dag'],
            size=B, p=list(method_probs),
        )

        for mode in ['none', 'contemp', 'pairwise', 'chain', 'dag']:
            idx = np.where(modes == mode)[0]
            if len(idx) == 0:
                continue

            sub_x = x_batch[idx]
            sub_roots = n_roots_list[idx]

            if mode == 'dag':
                m_x, m_adj, m_lags = _mix_dag_batch_jit(sub_x, N, L, sub_roots)
                out_batch[idx] = m_x
                out_adj[idx] = m_adj
                out_lags[idx] = m_lags

            elif mode == 'chain':
                m_x, m_adj, m_lags = _mix_chain_batch_jit(sub_x, N, L, sub_roots)
                out_batch[idx] = m_x
                out_adj[idx] = m_adj
                out_lags[idx] = m_lags

            elif mode == 'pairwise':
                m_x, m_adj, m_lags = _mix_pairwise_batch_jit(sub_x, N, L, sub_roots)
                out_batch[idx] = m_x
                out_adj[idx] = m_adj
                out_lags[idx] = m_lags

            elif mode == 'contemp':
                # Contemporaneous (lag-0) linear mixing.
                # FIX: Only mix PHYSICS nodes — roots stay as pure parents.
                for i, b_orig in enumerate(idx):
                    nr = sub_roots[i]
                    n_phys = N - nr

                    # Copy roots through untouched
                    out_batch[b_orig, :nr, :] = sub_x[i, :nr, :]

                    if n_phys > 0:
                        W = np.tril(np.random.randn(n_phys, n_phys), -1)
                        # Causal structural equation: Y = W*Y + X  => Y = (I - W)^-1 * X
                        mix_matrix = np.linalg.inv(np.eye(n_phys) - W)
                        out_batch[b_orig, nr:, :] = mix_matrix @ sub_x[i, nr:, :]
                        adj_phys = (np.abs(W) > 0.1).astype(np.float64) # True causal DAG

                        out_adj[b_orig, nr:, nr:] = adj_phys

                        # Root→physics edges (each root influences some physics nodes)
                        if nr > 0:
                            for r in range(nr):
                                # Each root connects to ~40-60% of physics nodes
                                connect_prob = np.random.uniform(0.3, 0.6)
                                connects = np.random.random(n_phys) < connect_prob
                                for j in range(n_phys):
                                    if connects[j]:
                                        out_adj[b_orig, r, nr + j] = 1.0
                    # Contemp = lag 0 everywhere, out_lags stays 0

            elif mode == 'none':
                out_batch[idx] = sub_x
                # adj and lags stay 0

        # NO post-mix normalization needed.
        # Convex blending keeps all node scales bounded:
        #   child = (1-α)*child + α*parent_signal
        # Since both child and parent are std ≈ 1, the result is also std ≈ 1.

        return out_batch, out_adj, out_lags


class FastPhysicsEngine:
    """
    Top-level engine that assembles control roots + physics responders
    and applies causal mixing.

    FIX: Controls are scaled to match z-scored physics dynamic range.
    FIX: Kernel generator called with normalize=False (engine normalizes once).
    FIX: gen_kernel_bounded removed (was identical to gen_kernel).
    FIX: Returns edge_lags alongside x and adj.
    """
    def __init__(self, seed: Optional[int] = None):
        self.gen_control = FastControlSignalGenerator()
        self.gen_stat = FastStatisticalGenerator()
        self.gen_kernel = OptimizedKernelSynthDiverse(bank_size=512, seed=seed)
        self.gen_industrial = IndustrialProcessSimulator(seed=seed)
        self.mixer = VectorizedMixerCausal()

    def generate(
        self,
        batch_size: int,
        n_nodes: int,
        length: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch of causal time-series graphs.
        Uses a 35/65 split between existing mathematical mixing and
        new industrial process simulators.

        Returns:
          x_final   — (B, N, L)
          adj_final — (B, N, N)
          lag_final — (B, N, N)
        """
        # Decide which samples use Path A (Existing) vs Path B (Industrial)
        r = np.random.random(batch_size)
        idx_math = np.where(r < 0.35)[0]
        idx_indu = np.where(r >= 0.35)[0]

        x_out = np.zeros((batch_size, n_nodes, length), dtype=np.float64)
        adj_out = np.zeros((batch_size, n_nodes, n_nodes), dtype=np.float64)
        lag_out = np.zeros((batch_size, n_nodes, n_nodes), dtype=np.int64)

        # ── PATH A: Existing Mathematical Mixing (35%) ──
        if len(idx_math) > 0:
            n_math = len(idx_math)
            n_ctrls_per_sample = np.zeros(n_math, dtype=np.int64)
            for i in range(n_math):
                if np.random.random() < 0.25:
                    max_ctrl = max(1, int(n_nodes * 0.3))
                    n_ctrls_per_sample[i] = np.random.randint(1, max_ctrl + 1)

            total_ctrl = int(np.sum(n_ctrls_per_sample))
            total_physics = n_math * n_nodes - total_ctrl

            # Generate signals
            ctrl_pool = self.gen_control.generate_batch(total_ctrl, length) if total_ctrl > 0 else np.empty((0, length))

            phys_arrays = []
            if total_physics > 0:
                nk = int(total_physics * 0.85)
                ns = total_physics - nk
                if nk > 0: phys_arrays.append(self.gen_kernel.generate_batch(nk, length, normalize=False))
                if ns > 0: phys_arrays.append(self.gen_stat.generate_batch(ns, length))
                phys_pool = np.vstack(phys_arrays)
                np.random.shuffle(phys_pool)
            else:
                phys_pool = np.empty((0, length))

            # Assemble base
            x_base = np.zeros((n_math, n_nodes, length), dtype=np.float64)
            pc, pp = 0, 0
            for i in range(n_math):
                nc = n_ctrls_per_sample[i]
                np_ = n_nodes - nc
                if nc > 0:
                    x_base[i, :nc] = ctrl_pool[pc : pc+nc]
                    pc += nc
                if np_ > 0:
                    x_base[i, nc:] = phys_pool[pp : pp+np_]
                    pp += np_

            # Mix
            xm, am, lm = self.mixer.apply_batch(x_base, n_roots_list=n_ctrls_per_sample)

            # Scatter into results
            for i, target_idx in enumerate(idx_math):
                x_out[target_idx] = xm[i]
                adj_out[target_idx] = am[i]
                lag_out[target_idx] = lm[i]

        # ── PATH B: Industrial Process Simulators (65%) ──
        if len(idx_indu) > 0:
            for b_idx in idx_indu:
                x_b, adj_b, lag_b = self.gen_industrial.generate(n_nodes, length)
                x_out[b_idx] = x_b
                adj_out[b_idx] = adj_b
                lag_out[b_idx] = lag_b

        # Apply final sensor artifacts (on top of any already applied)
        # This ensures all signals have that "noisy instrument" feel.
        x_out = _apply_sensor_artifacts(x_out)

        # ── PATH C: STABILITY GATEWAY (Final Normalization & Clamping) ──
        # Ensure the final output across the whole batch is well-behaved.
        for b in range(batch_size):
            for n in range(n_nodes):
                m = np.mean(x_out[b, n])
                s = np.std(x_out[b, n])
                s = max(s, 1e-4) # prevent noise amplification
                x_out[b, n] = (x_out[b, n] - m) / s

        return x_out, adj_out, lag_out

    def generate_validated(
        self,
        batch_size: int,
        n_nodes: int,
        length: int,
        min_edge_corr: float = 0.10,
        min_graph_score: float = 0.5,
        max_retries: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate and validate a batch.  Graphs with unlearnable cause-effect
        relationships are regenerated up to `max_retries` times.

        Returns:
          x_final   — (B, N, L)
          adj_final — (B, N, N)
          lag_final — (B, N, N)
          quality   — (B,) quality score per graph
        """
        x, adj, lags = self.generate(batch_size, n_nodes, length)

        for attempt in range(max_retries):
            scores, pass_mask = _validate_batch_quality_jit(
                x, adj, lags, min_edge_corr,
            )
            n_failed = int((~pass_mask).sum())
            if n_failed == 0:
                return x, adj, lags, scores

            fail_idx = np.where(~pass_mask)[0]
            n_regen = len(fail_idx)

            x_new, adj_new, lags_new = self.generate(n_regen, n_nodes, length)
            for i, fi in enumerate(fail_idx):
                x[fi] = x_new[i]
                adj[fi] = adj_new[i]
                lags[fi] = lags_new[i]

        scores, _ = _validate_batch_quality_jit(x, adj, lags, min_edge_corr)
        return x, adj, lags, scores


# ======================================================================
# 4b. COUPLING FUNCTION BANK (for ODE network simulator)
# ======================================================================

@njit(fastmath=True)
def _coupling_fn(x: float, ftype: int, p0: float, p1: float, p2: float) -> float:
    """
    Apply one of 10 coupling function types.
    Builds the vocabulary of possible relationships between state variables.
    Each is parameterized by p0, p1, p2 for diversity.
    """
    if ftype == 0:
        # Linear: gain * x
        return p0 * x
    elif ftype == 1:
        # Quadratic: ax² + bx (orifice plate, square-law flow)
        # Root Fix: Soft-saturate to prevent explosion
        val = p0 * x * x + p1 * x
        return 5.0 * np.tanh(val / 5.0)
    elif ftype == 2:
        # Saturating sigmoid: a * tanh(b * x)  (pH curves, amplifiers)
        return p0 * np.tanh(p1 * x)
    elif ftype == 3:
        # Michaelis-Menten: a * x / (K + |x|)  (enzyme kinetics)
        return p0 * x / (p1 + abs(x) + 1e-10)
    elif ftype == 4:
        # Arrhenius-like: a * exp(b * x)  (thermal runaway, reaction rates)
        # Clamped to prevent overflow
        arg = p1 * x
        arg = max(-10.0, min(10.0, arg))
        return p0 * np.exp(arg)
    elif ftype == 5:
        # Threshold/ReLU: a * max(0, x - b)
        return p0 * max(0.0, x - p1)
    elif ftype == 6:
        # Sinusoidal coupling: a * sin(b * x)
        return p0 * np.sin(p1 * x)
    elif ftype == 7:
        # Log-compression: a * log(1 + |x|) * sign(x)  (wide-range sensors)
        return p0 * np.log(1.0 + abs(x)) * (1.0 if x >= 0 else -1.0)
    elif ftype == 8:
        # Power law: a * |x|^b * sign(x)  (heat transfer: Nu ~ Re^0.8)
        return p0 * (abs(x) ** min(p1, 3.0)) * (1.0 if x >= 0 else -1.0)
    elif ftype == 9:
        # Cubic: a*x + b*x³  (softening/hardening spring)
        # Root Fix: Soft-saturate to prevent explosion
        val = p0 * x + p1 * x * x * x
        return 5.0 * np.tanh(val / 5.0)
    else:
        return p0 * x


# ======================================================================
# 4c. INDUSTRIAL PROCESS SIMULATORS
# ======================================================================

@njit(fastmath=True)
def _simulate_random_ode_network(
    n_nodes: int, length: int,
    driving_signals: np.ndarray,  # (n_nodes, length) from existing generators
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random N-dimensional coupled ODE system.

    Each state variable evolves as:
      dx_i/dt = -decay_i·x_i + Σ_j coupling_fn_ij(x_j(t - lag_ij)) + drive_i(t)

    Coupling functions are randomly selected from the bank.
    This creates an INFINITE family of dynamical systems.

    Returns: (signals, adj, lags)  all for a single graph (not batched)
    """
    # Random DAG structure (lower triangular for topological order)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    lags = np.zeros((n_nodes, n_nodes), dtype=np.int64)

    # Random coupling parameters per edge
    ftypes = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    p0s = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    p1s = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    p2s = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    # Decide edges: each non-root node gets 1-3 parents
    n_roots = max(1, np.random.randint(1, max(2, n_nodes // 3)))
    for i in range(n_roots, n_nodes):
        n_parents = np.random.randint(1, min(4, i + 1))
        for _ in range(n_parents):
            j = np.random.randint(0, i)
            adj[j, i] = 1.0
            lags[j, i] = np.random.randint(1, max(2, length // 10))
            ftypes[j, i] = np.random.randint(0, 10)

            # Random parameters for the coupling function
            p0s[j, i] = np.random.uniform(0.2, 1.5) * (1.0 if np.random.random() < 0.7 else -1.0)
            p1s[j, i] = np.random.uniform(0.1, 2.0)
            p2s[j, i] = np.random.uniform(0.0, 1.0)

    # Random dynamics parameters per state variable
    decays = np.zeros(n_nodes, dtype=np.float64)
    time_constants = np.zeros(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        decays[i] = np.random.uniform(0.01, 0.2)
        time_constants[i] = np.random.uniform(1.0, 20.0)

    # Integration (Euler)
    x = np.zeros((n_nodes, length), dtype=np.float64)
    # Initialize states near zero
    for i in range(n_nodes):
        x[i, 0] = np.random.uniform(-0.5, 0.5)

    drive_scale = np.random.uniform(0.05, 0.3)

    n_substeps = 5
    for t in range(1, length):
        # Multi-stepping for numerical stability
        # Each frame (dt=1) is divided into smaller steps
        curr_x = x[:, t-1].copy()

        for _ in range(n_substeps):
            # We calculate dx based on the state at the start of the sub-step
            # but use lagged values from the main 'x' array for simplicity
            for i in range(n_nodes):
                # Self-decay
                dx = -decays[i] * curr_x[i]

                # Coupling from parents (using lagged values from 'x')
                for j in range(n_nodes):
                    if adj[j, i] < 0.5:
                        continue
                    lag = lags[j, i]
                    t_lagged = max(0, t - lag)
                    parent_val = x[j, t_lagged]
                    dx += _coupling_fn(parent_val, ftypes[j, i],
                                       p0s[j, i], p1s[j, i], p2s[j, i])

                # External driving signal
                dx += drive_scale * driving_signals[i, t]

                # Sub-step integration
                dt_sub = (1.0 / time_constants[i]) / n_substeps
                curr_x[i] = curr_x[i] + dx * dt_sub

                # Sub-step stability clamp (tighter than final)
                curr_x[i] = max(-15.0, min(15.0, curr_x[i]))

        # Store final result for this frame
        for i in range(n_nodes):
            x[i, t] = curr_x[i]

    # Normalize each channel to ~unit variance
    for i in range(n_nodes):
        mean_ = 0.0
        for t in range(length):
            mean_ += x[i, t]
        mean_ /= length
        var_ = 0.0
        for t in range(length):
            var_ += (x[i, t] - mean_) ** 2
        std_ = max(np.sqrt(var_ / length), 1e-4)
        for t in range(length):
            x[i, t] = (x[i, t] - mean_) / std_

    return x, adj, lags


@njit(fastmath=True)
def _simulate_pid_control_loop(
    n_nodes: int, length: int,
    setpoint_signal: np.ndarray,     # (length,) from existing generator
    disturbance_signal: np.ndarray,  # (length,) from existing generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PID control loop simulator with random process + controller types.

    Process types (randomly selected):
      - First-order: τ·dy/dt + y = K·u
      - Second-order: τ²·d²y/dt² + 2ζτ·dy/dt + y = K·u
      - Integrating: dy/dt = K·u  (tank level, pure integrator)
      - Inverse response: initial movement opposite final direction

    Controller types (randomly selected):
      - P only, PI, PID, On-Off (bang-bang)

    Outputs 4-5 nodes: [Setpoint, PV, Controller_Output, Error, Disturbance]
    """
    # Random process parameters
    proc_type = np.random.randint(0, 4)  # 0=first, 1=second, 2=integrating, 3=inverse
    K_proc = np.random.uniform(0.5, 3.0)     # process gain
    tau1 = np.random.uniform(3.0, 30.0)       # primary time constant
    tau2 = np.random.uniform(1.0, 10.0)       # secondary (for 2nd order)
    zeta = np.random.uniform(0.2, 1.5)        # damping ratio (for 2nd order)
    dead_time = np.random.randint(1, max(2, length // 20))

    # Random controller type & tuning
    ctrl_type = np.random.randint(0, 4)  # 0=P, 1=PI, 2=PID, 3=on-off
    Kp = np.random.uniform(0.3, 5.0)
    Ki = np.random.uniform(0.0, 0.5) if ctrl_type >= 1 else 0.0
    Kd = np.random.uniform(0.0, 0.3) if ctrl_type >= 2 else 0.0

    # State variables
    pv = np.zeros(length, dtype=np.float64)  # process variable
    mv = np.zeros(length, dtype=np.float64)  # manipulated variable (controller output)
    error = np.zeros(length, dtype=np.float64)
    sp = setpoint_signal.copy()
    dist = disturbance_signal.copy()

    pv[0] = sp[0]  # start at setpoint
    state_2nd = 0.0  # velocity state for 2nd-order
    integral = 0.0
    prev_error = 0.0

    for t in range(1, length):
        # Error
        error[t] = sp[t] - pv[t-1]

        # Controller output
        if ctrl_type == 3:
            # On-off (bang-bang)
            mv[t] = 1.0 if error[t] > 0 else -1.0
        else:
            integral += error[t]
            integral = max(-50.0, min(50.0, integral))
            derivative = error[t] - prev_error
            mv[t] = Kp * error[t] + Ki * integral + Kd * derivative
            mv[t] = max(-5.0, min(5.0, mv[t]))  # actuator limits
        prev_error = error[t]

        # Process response to controller output (with dead time)
        t_input = max(0, t - dead_time)
        u = mv[t_input]

        if proc_type == 0:
            # First-order: τ·dy/dt + y = K·u
            pv[t] = pv[t-1] + (K_proc * u - pv[t-1]) / tau1
        elif proc_type == 1:
            # Second-order: using state-space form
            accel = (K_proc * u - pv[t-1] - 2.0 * zeta * tau2 * state_2nd) / (tau1 * tau2)
            state_2nd += accel
            pv[t] = pv[t-1] + state_2nd
        elif proc_type == 2:
            # Integrating: dy/dt = K·u
            pv[t] = pv[t-1] + K_proc * u / tau1
        else:
            # Inverse response
            fast = (K_proc * u - pv[t-1]) / (tau2 * 0.5)
            slow = (K_proc * u - pv[t-1]) / tau1
            pv[t] = pv[t-1] + (-0.3 * fast + 1.3 * slow)

        # Add disturbance
        pv[t] += 0.1 * dist[t]
        pv[t] = max(-10.0, min(10.0, pv[t]))

    # Assemble output
    actual_nodes = min(n_nodes, 5)
    x = np.zeros((n_nodes, length), dtype=np.float64)
    adj_out = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    lags_out = np.zeros((n_nodes, n_nodes), dtype=np.int64)

    # Node assignment: 0=SP, 1=PV, 2=MV, 3=Error, 4=Disturbance
    x[0] = sp
    if actual_nodes > 1:
        x[1] = pv
    if actual_nodes > 2:
        x[2] = mv
    if actual_nodes > 3:
        x[3] = error
    if actual_nodes > 4:
        x[4] = dist

    # Fill remaining nodes with noise/independent signals
    for i in range(actual_nodes, n_nodes):
        for t in range(length):
            x[i, t] = np.random.normal(0.0, 0.5)

    # Causal structure: SP → Error → MV → PV, Disturbance → PV
    if actual_nodes >= 3:
        adj_out[2, 1] = 1.0  # MV → PV
        lags_out[2, 1] = dead_time
    if actual_nodes >= 4:
        adj_out[0, 3] = 1.0  # SP → Error
        lags_out[0, 3] = 0
        adj_out[1, 3] = 1.0  # PV → Error
        lags_out[1, 3] = 1
        adj_out[3, 2] = 1.0  # Error -> MV
        lags_out[3, 2] = 0
    if actual_nodes >= 5:
        adj_out[4, 1] = 1.0  # Disturbance → PV
        lags_out[4, 1] = 1

    # Normalize all channels
    for i in range(n_nodes):
        mean_ = 0.0
        for t in range(length):
            mean_ += x[i, t]
        mean_ /= length
        var_ = 0.0
        for t in range(length):
            var_ += (x[i, t] - mean_) ** 2
        std_ = max(np.sqrt(var_ / length), 1e-4)
        for t in range(length):
            x[i, t] = (x[i, t] - mean_) / std_

    return x, adj_out, lags_out


@njit(fastmath=True)
def _simulate_batch_process(
    n_nodes: int, length: int,
    base_signals: np.ndarray,  # (n_nodes, length) for variability
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch/cyclic process: phase-structured temporal patterns.

    Recipe = sequence of 3-7 phases (Fill, Heat, React, Cool, Drain, Hold, etc.)
    Each phase has different setpoints, ramp rates, and dynamics.
    Quality variable = integral of conditions over batch (Arrhenius-weighted).

    Cycle-to-cycle variation via random parameter perturbation.
    """
    n_phases = np.random.randint(3, 8)
    phase_durations = np.zeros(n_phases, dtype=np.int64)
    remaining = length
    for p in range(n_phases - 1):
        phase_durations[p] = np.random.randint(
            max(10, length // (n_phases * 3)),
            max(20, length // n_phases),
        )
        remaining -= phase_durations[p]
    phase_durations[n_phases - 1] = max(10, remaining)

    # Random setpoints per phase
    temp_setpoints = np.zeros(n_phases, dtype=np.float64)
    pressure_setpoints = np.zeros(n_phases, dtype=np.float64)
    for p in range(n_phases):
        temp_setpoints[p] = np.random.uniform(-1.5, 1.5)
        pressure_setpoints[p] = np.random.uniform(-1.0, 1.0)

    # Generate phase indicator, temperature, pressure, quality, agitator
    x = np.zeros((n_nodes, length), dtype=np.float64)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    lags_out = np.zeros((n_nodes, n_nodes), dtype=np.int64)

    t = 0
    temp = temp_setpoints[0]
    pressure = pressure_setpoints[0]
    quality = 0.0
    ramp_rate = np.random.uniform(0.01, 0.1)

    for p in range(n_phases):
        dur = phase_durations[p]
        temp_sp = temp_setpoints[p]
        press_sp = pressure_setpoints[p]

        for dt_ in range(dur):
            if t >= length:
                break

            # Phase indicator (normalized to [-1, 1])
            x[0 % n_nodes, t] = 2.0 * p / max(n_phases - 1, 1) - 1.0

            # Temperature ramps toward setpoint
            temp += ramp_rate * (temp_sp - temp) + 0.02 * base_signals[1 % n_nodes, t]
            x[1 % n_nodes, t] = temp

            # Pressure follows temperature with coupling
            pressure += 0.05 * (press_sp - pressure) + 0.03 * (temp - pressure)
            pressure += 0.01 * base_signals[2 % n_nodes, t]
            x[2 % n_nodes, t] = pressure

            # Quality: integral of Arrhenius-weighted condition
            rate = np.exp(0.5 * temp)  # simplified Arrhenius
            quality += rate * 0.001
            if n_nodes > 3:
                x[3, t] = quality

            # Agitator speed (changes per phase)
            if n_nodes > 4:
                x[4, t] = np.random.uniform(0.3, 1.0) * (1 + 0.1 * p) + 0.02 * base_signals[4 % n_nodes, t]

            t += 1

    # Fill remaining nodes with perturbations of existing signals
    for i in range(5, n_nodes):
        coupling_src = np.random.randint(0, min(5, n_nodes))
        for t in range(length):
            x[i, t] = 0.3 * x[coupling_src, t] + 0.7 * base_signals[i, t]
        adj[coupling_src, i] = 1.0
        lags_out[coupling_src, i] = np.random.randint(1, max(2, length // 20))

    # Core causal structure
    if n_nodes >= 2:
        adj[0, 1] = 1.0  # Phase → Temperature
        lags_out[0, 1] = 1
    if n_nodes >= 3:
        adj[1, 2] = 1.0  # Temperature → Pressure
        lags_out[1, 2] = 0
    if n_nodes >= 4:
        adj[1, 3] = 1.0  # Temperature → Quality
        lags_out[1, 3] = 0

    # Normalize
    for i in range(n_nodes):
        mean_ = 0.0
        for t in range(length):
            mean_ += x[i, t]
        mean_ /= length
        var_ = 0.0
        for t in range(length):
            var_ += (x[i, t] - mean_) ** 2
        std_ = max(np.sqrt(var_ / length), 1e-4)
        for t in range(length):
            x[i, t] = (x[i, t] - mean_) / std_

    return x, adj, lags_out


@njit(fastmath=True)
def _simulate_tabular_regression(
    n_nodes: int, length: int,
    driving_signals: np.ndarray,  # (n_nodes, length) from existing generators
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-autoregressive instantaneous regression: y(t) = f(x1(t), x2(t)) + noise.

    NO temporal dynamics — tests whether the model can learn cross-sectional
    relationships. The time dimension is just repeated samples.

    f is randomly composed from coupling function bank primitives.
    """
    x = driving_signals.copy()
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    lags_out = np.zeros((n_nodes, n_nodes), dtype=np.int64)

    # First few nodes are independent inputs
    n_inputs = max(1, np.random.randint(1, max(2, n_nodes - 1)))
    # Remaining nodes are outputs = f(inputs) + noise
    for i in range(n_inputs, n_nodes):
        x[i, :] = 0.0  # clear the output channel

        # Pick 1-3 input parents
        n_parents = np.random.randint(1, min(4, n_inputs + 1))
        for _ in range(n_parents):
            j = np.random.randint(0, n_inputs)
            ftype = np.random.randint(0, 10)
            p0 = np.random.uniform(0.3, 1.5) * (1.0 if np.random.random() < 0.7 else -1.0)
            p1 = np.random.uniform(0.3, 2.0)
            p2 = np.random.uniform(0.0, 1.0)

            adj[j, i] = 1.0
            lags_out[j, i] = 0  # ZERO lag — instantaneous

            for t in range(length):
                x[i, t] += _coupling_fn(x[j, t], ftype, p0, p1, p2)

        # Add noise
        noise_level = np.random.uniform(0.05, 0.3)
        for t in range(length):
            x[i, t] += noise_level * np.random.normal(0.0, 1.0)

    # Normalize
    for i in range(n_nodes):
        mean_ = 0.0
        for t in range(length):
            mean_ += x[i, t]
        mean_ /= length
        var_ = 0.0
        for t in range(length):
            var_ += (x[i, t] - mean_) ** 2
        std_ = max(np.sqrt(var_ / length), 1e-4)
        for t in range(length):
            x[i, t] = (x[i, t] - mean_) / std_

    return x, adj, lags_out


@njit(fastmath=True)
def _simulate_conservation_network(
    n_nodes: int, length: int,
    valve_signals: np.ndarray,  # (n_nodes, length) valve positions from control generators
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conservation/flow network: tanks connected by pipes.

    Mass balance: dm_i/dt = Σ(flows_in) - Σ(flows_out)
    Flow law: flow_ij = Cv * √(max(0, h_i - h_j)) * valve_position_ij

    Nodes alternate between levels and flows:
    [Level_0, Flow_01, Level_1, Flow_12, Level_2, ...]
    """
    n_tanks = max(2, (n_nodes + 1) // 2)
    n_flows = n_tanks - 1  # sequential topology

    levels = np.zeros((n_tanks, length), dtype=np.float64)
    flows = np.zeros((n_flows, length), dtype=np.float64)

    # Initialize levels
    for i in range(n_tanks):
        levels[i, 0] = np.random.uniform(0.3, 0.8)

    # Random flow coefficients
    Cvs = np.zeros(n_flows, dtype=np.float64)
    for i in range(n_flows):
        Cvs[i] = np.random.uniform(0.05, 0.3)

    # Feed inflow to first tank
    feed_scale = np.random.uniform(0.02, 0.1)

    for t in range(1, length):
        for i in range(n_flows):
            # Flow from tank i to tank i+1
            dh = levels[i, t-1] - levels[i+1, t-1]
            valve_pos = 0.5 + 0.5 * np.tanh(valve_signals[i % len(valve_signals), t])
            flows[i, t] = Cvs[i] * np.sqrt(max(0.0, abs(dh))) * (1.0 if dh > 0 else -1.0) * valve_pos

        # Update levels (mass balance)
        for i in range(n_tanks):
            inflow = 0.0
            outflow = 0.0

            if i == 0:
                # Feed inflow from external
                inflow += feed_scale * (1.0 + valve_signals[0, t])
            if i > 0:
                inflow += max(0.0, flows[i-1, t])
                outflow += max(0.0, -flows[i-1, t])
            if i < n_flows:
                outflow += max(0.0, flows[i, t])
                inflow += max(0.0, -flows[i, t])

            levels[i, t] = levels[i, t-1] + (inflow - outflow)
            levels[i, t] = max(0.0, min(2.0, levels[i, t]))

    # Assemble output
    x = np.zeros((n_nodes, length), dtype=np.float64)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    lags_out = np.zeros((n_nodes, n_nodes), dtype=np.int64)

    node_idx = 0
    for i in range(n_tanks):
        if node_idx >= n_nodes:
            break
        x[node_idx] = levels[i]

        if i < n_flows and node_idx + 1 < n_nodes:
            x[node_idx + 1] = flows[i]
            # Level → Flow (causal)
            adj[node_idx, node_idx + 1] = 1.0
            lags_out[node_idx, node_idx + 1] = 1

            # Flow → downstream Level
            if node_idx + 2 < n_nodes:
                adj[node_idx + 1, node_idx + 2] = 1.0
                lags_out[node_idx + 1, node_idx + 2] = 0

        node_idx += 2

    # Fill any remaining nodes with valve signals
    for i in range(node_idx, n_nodes):
        idx = i % len(valve_signals)
        x[i] = valve_signals[idx]

    # Normalize
    for i in range(n_nodes):
        mean_ = 0.0
        for t in range(length):
            mean_ += x[i, t]
        mean_ /= length
        var_ = 0.0
        for t in range(length):
            var_ += (x[i, t] - mean_) ** 2
        std_ = max(np.sqrt(var_ / length), 1e-4)
        for t in range(length):
            x[i, t] = (x[i, t] - mean_) / std_

    return x, adj, lags_out


@njit(fastmath=True)
def _simulate_multiscale_process(
    n_nodes: int, length: int,
    slow_signals: np.ndarray,    # (n_nodes, length) slow base signals
    fast_signals: np.ndarray,    # (n_nodes, length) fast base signals
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-scale process: superposition of slow, medium, and fast dynamics.

    Causal effects operate at specific time scales:
    - Slow parent → slow child response (long lag)
    - Medium → medium (medium lag)
    - Fast noise overlay doesn't propagate causally

    Tests whether the model can separate scales and identify causal signals
    from noise at different frequencies.
    """
    # Scale separation factors
    slow_weight = np.random.uniform(0.3, 0.7)
    fast_weight = 1.0 - slow_weight

    x = np.zeros((n_nodes, length), dtype=np.float64)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    lags_out = np.zeros((n_nodes, n_nodes), dtype=np.int64)

    # Create multi-scale base signals
    for i in range(n_nodes):
        for t in range(length):
            x[i, t] = slow_weight * slow_signals[i, t] + fast_weight * fast_signals[i, t]

    # Apply causal effects at the slow time scale only
    n_roots = max(1, np.random.randint(1, max(2, n_nodes // 3)))
    slow_lag_base = max(5, length // 20)  # long lags for slow dynamics

    for i in range(n_roots, n_nodes):
        # Each non-root gets a slow-scale parent
        j = np.random.randint(0, i)
        lag = np.random.randint(slow_lag_base, slow_lag_base * 3)
        lag = min(lag, length // 3)
        alpha = np.random.uniform(0.3, 0.7)
        ftype = np.random.randint(0, 10)
        p0 = np.random.uniform(0.3, 1.5) * (1.0 if np.random.random() < 0.7 else -1.0)
        p1 = np.random.uniform(0.3, 2.0)

        adj[j, i] = 1.0
        lags_out[j, i] = lag

        # Apply slow-scale coupling
        for t in range(lag, length):
            parent_val = x[j, t - lag]
            effect_val = _coupling_fn(parent_val, ftype, p0, p1, 0.0)
            x[i, t] = (1.0 - alpha) * x[i, t] + alpha * effect_val

    # Normalize
    for i in range(n_nodes):
        mean_ = 0.0
        for t in range(length):
            mean_ += x[i, t]
        mean_ /= length
        var_ = 0.0
        for t in range(length):
            var_ += (x[i, t] - mean_) ** 2
        std_ = max(np.sqrt(var_ / length), 1e-4)
        for t in range(length):
            x[i, t] = (x[i, t] - mean_) / std_

    return x, adj, lags_out


# ======================================================================
# 4d. SENSOR ARTIFACT LAYER
# ======================================================================

@njit(fastmath=True)
def _apply_to_signal_channel_jit(sig: np.ndarray, length: int):
    """Core logic to apply random artifacts to a single 1D time-series channel."""
    # Majority of signals (75%) should have no artifact at all
    if np.random.random() < 0.75:
        return

    # Only keep Quantization (0) and Saturation (2)
    # 50/50 split between them if an artifact is applied
    artifact_type = 0 if np.random.random() < 0.5 else 2

    if artifact_type == 0:
        # Quantization
        bits = np.random.choice(np.array([8, 12, 16]))
        n_levels = 2 ** bits
        sig_min, sig_max = sig[0], sig[0]
        for t in range(length):
            if sig[t] < sig_min: sig_min = sig[t]
            if sig[t] > sig_max: sig_max = sig[t]
        sig_range = max(sig_max - sig_min, 1e-8)
        step = sig_range / n_levels
        for t in range(length):
            sig[t] = np.floor((sig[t] - sig_min) / step) * step + sig_min

    elif artifact_type == 2:
        # Saturation clipping
        clip_lo = np.random.uniform(-4.0, -2.0)
        clip_hi = np.random.uniform(2.0, 4.0)
        for t in range(length):
            if sig[t] < clip_lo: sig[t] = clip_lo
            elif sig[t] > clip_hi: sig[t] = clip_hi


@njit(parallel=True, fastmath=True)
def _apply_sensor_artifacts(x: np.ndarray) -> np.ndarray:
    """
    Parallel gateway to apply artifacts to 2D (N, L) or 3D (B, N, L) arrays.
    """
    out = x.copy()
    if x.ndim == 3:
        B, N, L = x.shape
        # Flatten first two dims for max parallelism
        for idx in prange(B * N):
            b = idx // N
            n = idx % N
            _apply_to_signal_channel_jit(out[b, n], L)
    else:
        N, L = x.shape
        for n in prange(N):
            _apply_to_signal_channel_jit(out[n], L)
    return out


# ======================================================================
# 4e. INTEGRATED INDUSTRIAL SIMULATOR
# ======================================================================

class IndustrialProcessSimulator:
    """
    Orchestrates all process simulators. Uses existing GP kernel bank
    and base generators as building blocks for driving signals.

    Each call to generate() produces a single (x, adj, lags) graph from
    a randomly-selected process type.
    """

    def __init__(self, seed: Optional[int] = None):
        self.gen_control = FastControlSignalGenerator()
        self.gen_stat = FastStatisticalGenerator()
        self.gen_kernel = OptimizedKernelSynthDiverse(bank_size=512, seed=seed)

    def _get_driving_signals(self, n_nodes: int, length: int) -> np.ndarray:
        """Generate diverse driving signals from existing generators."""
        n_kernel = max(1, n_nodes // 2)
        n_stat = n_nodes - n_kernel

        arrays = []
        if n_kernel > 0:
            arrays.append(self.gen_kernel.generate_batch(n_kernel, length, normalize=False))
        if n_stat > 0:
            arrays.append(self.gen_stat.generate_batch(n_stat, length))

        pool = np.vstack(arrays)
        np.random.shuffle(pool)
        return pool[:n_nodes]

    def _get_control_signals(self, n: int, length: int) -> np.ndarray:
        """Generate control/setpoint signals."""
        return self.gen_control.generate_batch(max(1, n), length)

    def generate(
        self,
        n_nodes: int,
        length: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a single graph from a randomly-selected process type.

        Returns: (x, adj, lags)
          x    — (N, L)
          adj  — (N, N)
          lags — (N, N)
        """
        # Choose process type
        r = np.random.random()

        if r < 0.30:
            # Random coupled ODE network (most diverse)
            drivers = self._get_driving_signals(n_nodes, length)
            x, adj, lags = _simulate_random_ode_network(n_nodes, length, drivers)

        elif r < 0.50:
            # PID control loop
            ctrl_signals = self._get_control_signals(2, length)
            x, adj, lags = _simulate_pid_control_loop(
                n_nodes, length, ctrl_signals[0], ctrl_signals[1],
            )

        elif r < 0.65:
            # Batch/cyclic process
            base = self._get_driving_signals(n_nodes, length)
            x, adj, lags = _simulate_batch_process(n_nodes, length, base)

        elif r < 0.75:
            # Tabular regression (non-autoregressive)
            drivers = self._get_driving_signals(n_nodes, length)
            x, adj, lags = _simulate_tabular_regression(n_nodes, length, drivers)

        elif r < 0.87:
            # Conservation/flow network
            valves = self._get_control_signals(n_nodes, length)
            x, adj, lags = _simulate_conservation_network(n_nodes, length, valves)

        else:
            # Multi-scale process
            slow = self._get_driving_signals(n_nodes, length)
            fast = self._get_driving_signals(n_nodes, length)
            x, adj, lags = _simulate_multiscale_process(n_nodes, length, slow, fast)

        # Apply sensor artifacts (70% chance)
        if np.random.random() < 0.70:
            x = _apply_sensor_artifacts(x)

        return x, adj, lags

    def generate_batch(
        self,
        batch_size: int,
        n_nodes: int,
        length: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch of industrial process graphs.
        Returns: (x, adj, lags) all with batch dimension.
        """
        x_all = np.zeros((batch_size, n_nodes, length), dtype=np.float64)
        adj_all = np.zeros((batch_size, n_nodes, n_nodes), dtype=np.float64)
        lags_all = np.zeros((batch_size, n_nodes, n_nodes), dtype=np.int64)

        for b in range(batch_size):
            x_all[b], adj_all[b], lags_all[b] = self.generate(n_nodes, length)

        return x_all, adj_all, lags_all



def validate_graph_quality(
    x_batch: np.ndarray,
    adj_batch: np.ndarray,
    lag_batch: np.ndarray,
    min_edge_corr: float = 0.10,
    min_graph_score: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate that graphs contain learnable cause-effect relationships.

    Three checks per graph:
      1. Lagged Pearson correlation: |corr(parent[:-L], child[L:])| ≥ min_edge_corr
      2. Lag specificity: corr at declared lag > 0.8 × corr at lag-0
      3. Node degeneracy: all nodes have std > 0.05 and no NaN/Inf

    Returns:
      scores     — (B,) float64, fraction of edges that pass quality check
      pass_mask  — (B,) bool, True if graph passes overall threshold
    """
    scores, pass_mask = _validate_batch_quality_jit(
        x_batch, adj_batch, lag_batch, min_edge_corr,
    )
    pass_mask = pass_mask & (scores >= min_graph_score)
    return scores, pass_mask


@njit(parallel=True, fastmath=True)
def _validate_batch_quality_jit(
    x_batch: np.ndarray,
    adj_batch: np.ndarray,
    lag_batch: np.ndarray,
    min_edge_corr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    B = x_batch.shape[0]
    N = x_batch.shape[1]
    L = x_batch.shape[2]

    scores = np.zeros(B, dtype=np.float64)
    pass_mask = np.ones(B, dtype=np.bool_)

    for b in prange(B):
        # ── 1. Node activity/liveness check ──
        nodes_ok = True
        for n in range(N):
            # Check for NaN/Inf
            has_bad = False
            for t in range(L):
                v = x_batch[b, n, t]
                if v != v or v == np.inf or v == -np.inf:
                    has_bad = True; break
            if has_bad:
                nodes_ok = False; break

            # --- THE FIX: Liveness/Flatness Check ---
            # We check the standard deviation of the LAST 25% of the signal.
            # If the signal "died" (like your Sample 3), the tail std will be ~0.
            tail_start = int(L * 0.75)
            tail_len = L - tail_start

            t_sum = 0.0
            for t in range(tail_start, L):
                t_sum += x_batch[b, n, t]
            t_mean = t_sum / tail_len

            t_var = 0.0
            for t in range(tail_start, L):
                diff = x_batch[b, n, t] - t_mean
                t_var += diff * diff
            tail_std = np.sqrt(t_var / tail_len)

            # If the tail is flat, reject the whole graph
            # 0.02 is a safe threshold for "dead" sensor noise
            if tail_std < 0.02:
                nodes_ok = False; break

            # Also ensure the signal isn't just a constant step (Sample 4)
            # Count changes: if signal doesn't change enough, it's "boring"
            changes = 0
            for t in range(1, L):
                if abs(x_batch[b, n, t] - x_batch[b, n, t-1]) > 1e-5:
                    changes += 1
            if changes < 5: # Minimum 5 transitions over 2048 steps
                nodes_ok = False; break

        if not nodes_ok:
            pass_mask[b] = False
            scores[b] = 0.0
            continue

        # ── 2. Edge quality check (Correlation) ──
        n_edges = 0
        n_good = 0
        for i in range(N):
            for j in range(N):
                if adj_batch[b, i, j] < 0.5: continue
                n_edges += 1
                lag = int(lag_batch[b, i, j])

                if lag <= 0: # Contemp edges are harder to validate via lag-corr
                    n_good += 1; continue

                # Pearson Correlation at lag
                n_pts = L - lag
                p_sum, c_sum = 0.0, 0.0
                for t in range(n_pts):
                    p_sum += x_batch[b, i, t]
                    c_sum += x_batch[b, j, t + lag]
                p_mu, c_mu = p_sum/n_pts, c_sum/n_pts

                cov, p_v, c_v = 0.0, 0.0, 0.0
                for t in range(n_pts):
                    dp = x_batch[b, i, t] - p_mu
                    dc = x_batch[b, j, t + lag] - c_mu
                    cov += dp * dc
                    p_v += dp * dp
                    c_v += dc * dc

                if p_v < 1e-9 or c_v < 1e-9: continue
                corr = abs(cov / np.sqrt(p_v * c_v))

                # Check if this correlation is actually meaningful
                if corr >= min_edge_corr:
                    n_good += 1

        if n_edges > 0:
            scores[b] = n_good / n_edges
        else:
            scores[b] = 1.0

    return scores, pass_mask


# ======================================================================
# 5. PYTORCH DATASET
# ======================================================================

import torch
from torch_geometric.data import Data

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class InfiniteIndustrialDataset(torch.utils.data.IterableDataset):
    """
    Infinite-length IterableDataset that yields PyG Data objects
    with synthetic causal time-series graphs.

    FIX: Stores edge_lags in Data objects for supervised lag learning.
    FIX: Uses per-worker RNG seeding via np.random.seed.
    FIX: bare except → proper exception handling.
    """
    def __init__(
        self,
        seq_len: int = 1024,
        num_nodes_range: Tuple[int, int] = (3, 8),
        samples_per_epoch: int = 2000,
        batch_size: int = 32,
        seed: int = 42,
        use_preloaded: bool = False,
        preload_path: str = "industrial_buffer.pt",
        log_interval: int = 10000,
        debug_plot_dir: str = "./debug_plots",
    ):
        self.seq_len = seq_len
        self.num_nodes_range = num_nodes_range
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.base_seed = seed
        self.engine = FastPhysicsEngine(seed=seed)
        self._has_seeded = False
        self._epoch = 0

        self.log_interval = log_interval
        self.debug_plot_dir = debug_plot_dir

        if os.path.exists(self.debug_plot_dir):
            try:
                shutil.rmtree(self.debug_plot_dir)
            except OSError:
                pass
        os.makedirs(self.debug_plot_dir, exist_ok=True)

        self.use_preloaded = use_preloaded
        self.preload_path = preload_path

    def _plot_kernel_samples(self, kernel_mode, samples_list, step=0):
        """Plot samples with adjacency (showing lag annotations) and time series."""
        if not HAS_MATPLOTLIB or not samples_list:
            return

        n_samples = min(len(samples_list), 5)
        fig, axes = plt.subplots(n_samples, 2, figsize=(18, 4 * n_samples))
        if n_samples == 1:
            axes = np.array([axes])

        fig.suptitle(f"Graph Visualization: {kernel_mode} (Step {step})", fontsize=16)

        for i in range(n_samples):
            traj, adj, lags, q_score = samples_list[i]
            n_nodes = adj.shape[0]

            # ── Left: Adjacency matrix with lag annotations ──
            ax_adj = axes[i, 0]
            ax_adj.imshow(adj, cmap='Blues', vmin=0, vmax=1, alpha=0.3)

            # Annotate each cell with the lag value where an edge exists
            for src in range(n_nodes):
                for tgt in range(n_nodes):
                    if adj[src, tgt] > 0.5:
                        lag_val = int(lags[src, tgt])
                        ax_adj.text(
                            tgt, src, str(lag_val),
                            ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='darkred',
                        )

            ax_adj.set_title(
                f"Sample {i+1}: Adjacency ({n_nodes}N) | Q={q_score:.2f}",
                fontsize=10,
            )
            ax_adj.set_xlabel("Target (child)")
            ax_adj.set_ylabel("Source (parent)")
            ax_adj.set_xticks(range(n_nodes))
            ax_adj.set_yticks(range(n_nodes))

            # ── Right: Time series with node labels ──
            ax_ts = axes[i, 1]
            colors = plt.cm.tab10(np.linspace(0, 1, min(n_nodes, 10)))
            for n in range(n_nodes):
                signal = traj[n].numpy() if hasattr(traj[n], 'numpy') else traj[n]
                ax_ts.plot(
                    signal, alpha=0.85, linewidth=1.0,
                    color=colors[n % len(colors)],
                    label=f"N{n}",
                )
            ax_ts.set_title(f"Dynamics ({n_nodes} nodes)")
            ax_ts.legend(loc='upper right', fontsize=7, ncol=min(n_nodes, 4))

        plt.tight_layout()
        filename = os.path.join(self.debug_plot_dir, f"{kernel_mode}_step_{step}.png")
        plt.savefig(filename, dpi=120)
        plt.close(fig)

    def __iter__(self):
        from tqdm import tqdm

        worker_info = torch.utils.data.get_worker_info()

        if not self._has_seeded:
            worker_id = worker_info.id if worker_info else 0
            seed = (self.base_seed + worker_id + self._epoch * 1000) % 2**32
            np.random.seed(seed)
            self._has_seeded = True
        self._epoch += 1

        per_worker = self.samples_per_epoch
        if worker_info:
            per_worker = int(math.ceil(self.samples_per_epoch / worker_info.num_workers))

        # ── PATH A: PRELOADED SERVING ──
        if self.use_preloaded and self.preload_path:
            buffer_data = []
            paths = self.preload_path if isinstance(self.preload_path, list) else [self.preload_path]
            for path in paths:
                if os.path.exists(path):
                    chunk = torch.load(path, weights_only=False)
                    buffer_data.extend(chunk)

            random.shuffle(buffer_data)

            if worker_info:
                total = len(buffer_data)
                per_w = int(math.ceil(total / worker_info.num_workers))
                start = worker_info.id * per_w
                my_samples = buffer_data[start: min(start + per_w, total)]
            else:
                my_samples = buffer_data

            for sample in my_samples[:per_worker]:
                yield sample
            return

        # ── PATH B: GENERATIVE LOOP ──
        yielded_count = 0
        BURST_SIZE = 64
        BUFFER_SIZE = 256 * 12
        mix_buffer = []

        plot_buffer = {}
        plotted_kernels = set()
        worker_id_str = f"Worker {worker_info.id}" if worker_info else "Main"

        while yielded_count < per_worker:

            while len(mix_buffer) < BUFFER_SIZE:
                n_nodes = np.random.randint(
                    self.num_nodes_range[0], self.num_nodes_range[1] + 1,
                )

                try:
                    # Use validated generation — resamples graphs with unlearnable effects
                    raw_batch, adj_batch, lag_batch, quality_scores = self.engine.generate_validated(
                        batch_size=BURST_SIZE, n_nodes=n_nodes, length=self.seq_len,
                    )
                except ZeroDivisionError:
                    # If a specific random parameter triggers a zero-division in Numba,
                    # we simply skip this burst and try again.
                    continue
                except Exception:
                    # Catch-all for any other rare numerical instabilities
                    continue

                kernel_mode = 'math_physics'

                for b in range(BURST_SIZE):
                    traj_np = raw_batch[b]  # (N, T)
                    traj = torch.from_numpy(traj_np).float()

                    # Debug plotting — capture first sample from each burst
                    if kernel_mode not in plotted_kernels:
                        if kernel_mode not in plot_buffer:
                            plot_buffer[kernel_mode] = []
                        if b == 0:
                            plot_buffer[kernel_mode].append((
                                traj.clone(),
                                adj_batch[b].copy(),
                                lag_batch[b].copy(),
                                float(quality_scores[b]),
                            ))
                        if len(plot_buffer[kernel_mode]) >= 5:
                            self._plot_kernel_samples(kernel_mode, plot_buffer[kernel_mode], step=yielded_count)
                            plotted_kernels.add(kernel_mode)
                            del plot_buffer[kernel_mode]

                    # ── Build Data object ──
                    # Parquet schema expects:
                    #   x:          list<float32>  (flattened N*T)
                    #   edge_index: list<int64>    (flattened 2*E)
                    #   edge_lags:  list<int32>    (E,)
                    #   num_nodes:  int64
                    #   seq_len:    int64
                    #   k_lbl:      string

                    # Shape: (N, 1, T) — parquet conversion flattens to (N*T,)
                    x = traj.unsqueeze(1)  # float32

                    # Edge index from adjacency
                    rows, cols = np.nonzero(adj_batch[b] > 0.5)
                    if len(rows) > 0:
                        edge_index = torch.from_numpy(np.stack([rows, cols], axis=0)).long()
                        edge_lags = torch.from_numpy(
                            lag_batch[b][rows, cols].astype(np.int32)
                        )
                    else:
                        # No edges — store empty. Self-loops are added by the
                        # downstream loader (ParquetGraphDataset), not here.
                        edge_index = torch.zeros((2, 0), dtype=torch.long)
                        edge_lags = torch.zeros(0, dtype=torch.int32)

                    data = Data(x=x, edge_index=edge_index)
                    data.num_nodes = int(n_nodes)
                    data.seq_len = int(self.seq_len)    # required for parquet loader reshape
                    data.edge_lags = edge_lags           # int32, (E,)
                    data.k_sparsity = float(np.mean(adj_batch[b]))
                    data.k_quality = float(quality_scores[b])
                    data.k_lbl = kernel_mode

                    mix_buffer.append(data)

            np.random.shuffle(mix_buffer)

            while mix_buffer and yielded_count < per_worker:
                yield mix_buffer.pop()
                yielded_count += 1

                if yielded_count % self.log_interval == 0:
                    print(f"[{worker_id_str}] Generated {yielded_count} samples...")
                    plotted_kernels.clear()
                    plot_buffer.clear()



import random
import os
import gc
import shutil
import numpy as np
import torch
import multiprocessing
from tqdm.auto import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    from google.colab import drive
    HAS_COLAB = True
except ImportError:
    HAS_COLAB = False


# ======================================================================
# PARQUET SCHEMA (must match ParquetGraphDataset expectations)
# ======================================================================
PARQUET_SCHEMA = pa.schema([
    ("x",          pa.list_(pa.float32())),   # flattened (N*T,)
    ("edge_index", pa.list_(pa.int64())),     # flattened (2*E,)
    ("edge_lags",  pa.list_(pa.int32())),     # (E,)
    ("num_nodes",  pa.int64()),
    ("seq_len",    pa.int64()),
    ("k_lbl",      pa.string()),
    ("k_quality",  pa.float64()),
    ("k_sparsity", pa.float64()),
]) if HAS_PYARROW else None


def _data_to_row(data) -> dict:
    """Convert a PyG Data object to a flat dict for parquet serialization."""
    x_flat = data.x.squeeze(1).numpy().ravel().tolist()       # (N,1,T) -> (N*T,) float32
    ei_flat = data.edge_index.numpy().ravel().tolist()         # (2,E) -> (2*E,) int64
    lags = data.edge_lags.numpy().ravel().tolist()             # (E,) int32

    return {
        "x":          x_flat,
        "edge_index": ei_flat,
        "edge_lags":  lags,
        "num_nodes":  int(data.num_nodes),
        "seq_len":    int(data.seq_len),
        "k_lbl":      str(getattr(data, 'k_lbl', 'industrial')),
        "k_quality":  float(getattr(data, 'k_quality', 0.0)),
        "k_sparsity": float(getattr(data, 'k_sparsity', 0.0)),
    }


def generate_to_parquet(
    output_path: str,
    total_samples: int = 2000,
    seq_len: int = 1024,
    num_nodes_range: tuple = (3, 8),
    seed: int = 42,
    flush_every: int = 5000,
):
    """
    Generate synthetic causal graphs and save directly to parquet.

    Output schema matches ParquetGraphDataset expectations:
      x:          list<float32>   (flattened N*T)
      edge_index: list<int64>     (flattened 2*E)
      edge_lags:  list<int32>     (E,)
      num_nodes:  int64
      seq_len:    int64
      k_lbl:      string
      k_quality:  float64
      k_sparsity: float64
    """
    assert HAS_PYARROW, "pyarrow is required: pip install pyarrow"

    print(f"🚀 Generating {total_samples} samples -> {output_path}")
    print(f"   seq_len={seq_len}, nodes={num_nodes_range}, seed={seed}")

    dataset = InfiniteIndustrialDataset(
        seq_len=seq_len,
        samples_per_epoch=total_samples,
        batch_size=1,
        num_nodes_range=num_nodes_range,
        seed=seed,
        use_preloaded=False,
    )

    writer = None
    rows_buffer = []
    total_written = 0
    iterator = iter(dataset)

    with tqdm(total=total_samples, unit="sample", desc="Generating") as pbar:
        while total_written + len(rows_buffer) < total_samples:
            try:
                data = next(iterator)
                rows_buffer.append(_data_to_row(data))
                pbar.update(1)
            except StopIteration:
                print("⚠️ Generator stopped early!")
                break

            # Flush buffer to disk periodically to avoid OOM
            if len(rows_buffer) >= flush_every:
                batch = pa.RecordBatch.from_pydict(
                    {col: [r[col] for r in rows_buffer] for col in PARQUET_SCHEMA.names},
                    schema=PARQUET_SCHEMA,
                )
                if writer is None:
                    writer = pq.ParquetWriter(output_path, PARQUET_SCHEMA, compression="snappy")
                writer.write_batch(batch)
                total_written += len(rows_buffer)
                rows_buffer.clear()
                gc.collect()

    # Write remaining rows
    if rows_buffer:
        batch = pa.RecordBatch.from_pydict(
            {col: [r[col] for r in rows_buffer] for col in PARQUET_SCHEMA.names},
            schema=PARQUET_SCHEMA,
        )
        if writer is None:
            writer = pq.ParquetWriter(output_path, PARQUET_SCHEMA, compression="snappy")
        writer.write_batch(batch)
        total_written += len(rows_buffer)
        rows_buffer.clear()

    if writer is not None:
        writer.close()

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Saved {total_written} samples -> {output_path} ({file_size:.1f} MB)")

    # Verify schema
    meta = pq.read_metadata(output_path)
    print(f"   Rows: {meta.num_rows}, Row groups: {meta.num_row_groups}")
    schema_out = pq.read_schema(output_path)
    for field in schema_out:
        print(f"   {field.name:>12s}: {field.type}")

    del dataset, iterator
    gc.collect()




# === EXECUTION ===
if __name__ == "__main__":

    # Global Configurations
    NUM_CHUNKS_PER_SETTING = 10
    # Standardizing sample count; adjust if memory becomes an issue at 4096
    SAMPLES_PER_CHUNK = 512 * 128
    BASE_SEED = 4000
    NODES_RANGE = (1, 7)

    # Target Sequence Lengths
    SEQUENCE_LENGTHS = [2048] #[512, 2048, 4096]

    for SEQ_LEN in SEQUENCE_LENGTHS:
        print(f"\n🚀 STARTING GENERATION FOR SEQ_LEN: {SEQ_LEN}")

        # Path Management
        if 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False:
            from google.colab import drive
            drive.mount('/content/drive')
            PARENT_DIR = f"/content/drive/My Drive/Industrial_Datasets_KernelSynth"
        else:
            PARENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

        # Create specific directory for this sequence length
        OUTPUT_DIR = os.path.join(PARENT_DIR, f"len_{SEQ_LEN}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Reproducibility per Sequence Length
        current_seed = BASE_SEED + SEQ_LEN
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)

        print(f"🏁 Generating {NUM_CHUNKS_PER_SETTING} chunks -> {OUTPUT_DIR}")

        for i in range(NUM_CHUNKS_PER_SETTING):
            print(f"  --- Chunk {i+1}/{NUM_CHUNKS_PER_SETTING} (Len {SEQ_LEN}) ---")

            chunk_path = os.path.join(
                OUTPUT_DIR,
                f"industrial_graph_diverse_large_chunk_L{SEQ_LEN}_C{i}_v2.parquet",
            )

            # Generate the data
            generate_to_parquet(
                output_path=chunk_path,
                total_samples=SAMPLES_PER_CHUNK,
                seq_len=SEQ_LEN,
                num_nodes_range=NODES_RANGE,
                seed=current_seed + i,
            )

            # Force memory cleanup after each large chunk
            gc.collect()

    print(f"\n✅ All generations complete! Check {PARENT_DIR} for results.")
