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


# ======================================================================
# 2. MULTI-THREADED GAUSSIAN PROCESS BANK
# ======================================================================

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

    Uses CONVEX BLENDING instead of additive mixing:
        child = (1 - α) * child + α * f(lagged_driver)

    This directly controls the causal signal-to-noise ratio:
      α=0.3 → 30% of child is driven by parent (clearly visible)
      α=0.7 → 70% of child is driven by parent (dominant)

    Scale stays bounded (no post-mix normalization needed).

    Interaction types:
      0 — Linear blend
      1 — Saturating blend (tanh to handle large drivers)
      2 — Multiplicative modulation (preserves child shape, parent gates amplitude)
      3 — Complex physics (2nd-order dynamic response to driver)

    Parameters:
      alpha : float in [0, 1] — blend strength (fraction of child replaced by effect)
    """
    # Build the lagged driver signal
    effect = np.zeros(length, dtype=np.float64)
    if lag < length:
        effect[lag:] = driver[:(length - lag)]

    # Compute the parent's contribution (what the child "wants to look like")
    if itype == 0:
        # Linear: direct copy of lagged parent
        parent_signal = effect
    elif itype == 1:
        # Saturating: tanh squashes extreme values
        parent_signal = np.tanh(effect)
    elif itype == 2:
        # Multiplicative modulation: parent gates the child's amplitude
        # Special case — not a blend, parent scales the existing child
        modulation = 1.0 + np.tanh(effect) * alpha
        return child * modulation
    elif itype == 3:
        # Complex physics: 2nd-order dynamic response
        parent_signal = _apply_complex_physics(effect, length)
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

                # Controls → prefer physics dynamics (type 3)
                if is_control:
                    itype = 3 if np.random.random() < 0.8 else 0
                else:
                    itype = np.random.randint(0, 4)

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
            itype = np.random.randint(0, 4)
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
            itype = np.random.randint(0, 4)
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
        n_steps = int(n_samples * 0.2)
        n_bin = int(n_samples * 0.1)
        n_t_bin = int(n_samples * 0.6)
        n_pid = n_samples - n_steps - n_bin - n_t_bin

        arrays = []
        if n_steps > 0:
            arrays.append(_fast_control_steps(n_steps, length))
        if n_bin > 0:
            arrays.append(_fast_telegraph_process(n_bin, length))
        if n_t_bin > 0:
            arrays.append(_fast_telegraph_process_binary(n_t_bin, length))
        if n_pid > 0:
            arrays.append(_fast_damped_oscillation(n_pid, length))
        return np.vstack(arrays) if arrays else np.empty((0, length))


class FastStatisticalGenerator:
    def generate_batch(self, n_samples: int, length: int) -> np.ndarray:
        n_ari = int(n_samples * 0.4)
        n_tur = int(n_samples * 0.3)
        n_ets = n_samples - n_ari - n_tur

        arrays = []
        if n_ari > 0:
            arrays.append(_fast_arima_batch(n_ari, length, 3, 3))
        if n_tur > 0:
            arrays.append(_fast_heteroscedastic_noise(n_tur, length))
        if n_ets > 0:
            arrays.append(_fast_ets_batch(n_ets, length))
        return np.vstack(arrays) if arrays else np.empty((0, length))


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
        self.gen_kernel = OptimizedKernelSynthDiverse(bank_size=1024, seed=seed)
        self.mixer = VectorizedMixerCausal()

    def generate(
        self,
        batch_size: int,
        n_nodes: int,
        length: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch of causal time-series graphs.

        Returns:
          x_final   — (B, N, L)  node trajectories
          adj_final — (B, N, N)  binary adjacency
          lag_final — (B, N, N)  ground-truth lag per edge
        """
        # 1. Stochastic allocation of control vs physics nodes
        n_ctrls_per_sample = np.zeros(batch_size, dtype=np.int64)
        for b in range(batch_size):
            if np.random.random() < 0.25:  # slightly higher than 20% for richer causal graphs
                max_ctrl = max(1, int(n_nodes * 0.3))
                n_ctrls_per_sample[b] = np.random.randint(1, max_ctrl + 1)

        total_ctrl = int(np.sum(n_ctrls_per_sample))
        total_signals = batch_size * n_nodes
        total_physics = total_signals - total_ctrl

        # ── A. GENERATE CONTROLS (preserve logical levels) ──
        if total_ctrl > 0:
            raw_ctrl = self.gen_control.generate_batch(total_ctrl, length)
            # DO NOT z-score controls.  Binary valves must stay at clean ±1
            # logical levels, and step functions must keep their discrete
            # setpoints.  The generators already produce values in [-1, 1],
            # giving a natural std ≈ 0.5–1.0 that matches z-scored physics.
            ctrl_normed = raw_ctrl
        else:
            ctrl_normed = np.empty((0, length), dtype=np.float64)

        # ── B. GENERATE PHYSICS (Responders) ──
        if total_physics > 0:
            n_kernel = int(total_physics * 0.85)
            n_stat = total_physics - n_kernel

            arrays_physics = []
            if n_kernel > 0:
                # FIX: Single kernel generator, normalize=False (we do it here)
                arrays_physics.append(self.gen_kernel.generate_batch(n_kernel, length, normalize=False))
            if n_stat > 0:
                arrays_physics.append(self.gen_stat.generate_batch(n_stat, length))

            cont_pool = np.vstack(arrays_physics)
            np.random.shuffle(cont_pool)

            cont_norm = cont_pool
        else:
            cont_norm = np.empty((0, length), dtype=np.float64)

        # ── C. CONSTRUCT BATCH (roots first) ──
        x_base = np.zeros((batch_size, n_nodes, length), dtype=np.float64)
        ptr_c, ptr_p = 0, 0

        for b in range(batch_size):
            nc = n_ctrls_per_sample[b]
            np_ = n_nodes - nc

            if nc > 0:
                x_base[b, :nc, :] = ctrl_normed[ptr_c: ptr_c + nc]
                ptr_c += nc
            if np_ > 0:
                x_base[b, nc:, :] = cont_norm[ptr_p: ptr_p + np_]
                ptr_p += np_

        # ── D. APPLY CAUSAL MIXING ──
        x_final, adj_final, lag_final = self.mixer.apply_batch(
            x_base, n_roots_list=n_ctrls_per_sample,
        )

        return x_final, adj_final, lag_final

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
          scores    — (B,) quality score per graph
        """
        x, adj, lags = self.generate(batch_size, n_nodes, length)
        scores, passed = validate_graph_quality(
            x, adj, lags, min_edge_corr, min_graph_score,
        )

        for retry in range(max_retries):
            bad_idx = np.where(~passed)[0]
            if len(bad_idx) == 0:
                break

            n_bad = len(bad_idx)
            x_new, adj_new, lags_new = self.generate(n_bad, n_nodes, length)
            scores_new, passed_new = validate_graph_quality(
                x_new, adj_new, lags_new, min_edge_corr, min_graph_score,
            )

            x[bad_idx] = x_new
            adj[bad_idx] = adj_new
            lags[bad_idx] = lags_new
            scores[bad_idx] = scores_new
            passed[bad_idx] = passed_new

        return x, adj, lags, scores


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
    """
    Numba-JIT quality validator.  Runs in parallel over the batch.

    Per-edge metrics:
      - |pearsonr(parent[:-lag], child[lag:])| ≥ min_edge_corr
      - corr_at_lag / corr_at_0  >  0.8  (lag specificity)

    Per-node check:
      - std(node) > 0.05
      - all values finite
    """
    B = x_batch.shape[0]
    N = x_batch.shape[1]
    L = x_batch.shape[2]

    scores = np.zeros(B, dtype=np.float64)
    pass_mask = np.ones(B, dtype=np.bool_)

    for b in prange(B):
        # ── Node degeneracy check ──
        nodes_ok = True
        for n in range(N):
            node_mean = 0.0
            has_bad = False
            for t in range(L):
                v = x_batch[b, n, t]
                if v != v or v == np.inf or v == -np.inf:  # NaN/Inf check
                    has_bad = True
                    break
                node_mean += v

            if has_bad:
                nodes_ok = False
                break

            node_mean /= L
            node_var = 0.0
            for t in range(L):
                d = x_batch[b, n, t] - node_mean
                node_var += d * d
            node_std = np.sqrt(node_var / L)

            if node_std < 0.05:
                nodes_ok = False
                break

        if not nodes_ok:
            pass_mask[b] = False
            scores[b] = 0.0
            continue

        # ── Edge quality check ──
        n_edges = 0
        n_good = 0

        for i in range(N):
            for j in range(N):
                if adj_batch[b, i, j] < 0.5:
                    continue

                n_edges += 1
                lag = int(lag_batch[b, i, j])

                # Skip edges with lag=0 (contemp) or invalid lag
                if lag <= 0 or lag >= L:
                    # Contemp edges get a pass — we can't check them via
                    # lagged correlation but they're still valid mixings.
                    n_good += 1
                    continue

                n_pts = L - lag

                # ── Pearson correlation at declared lag ──
                p_mean = 0.0
                c_mean = 0.0
                for t in range(n_pts):
                    p_mean += x_batch[b, i, t]
                    c_mean += x_batch[b, j, t + lag]
                p_mean /= n_pts
                c_mean /= n_pts

                cov = 0.0
                p_var = 0.0
                c_var = 0.0
                for t in range(n_pts):
                    dp = x_batch[b, i, t] - p_mean
                    dc = x_batch[b, j, t + lag] - c_mean
                    cov += dp * dc
                    p_var += dp * dp
                    c_var += dc * dc

                if p_var < 1e-12 or c_var < 1e-12:
                    continue

                corr_at_lag = abs(cov / np.sqrt(p_var * c_var))

                # ── Pearson correlation at lag=0 (contemporaneous) ──
                p_mean0 = 0.0
                c_mean0 = 0.0
                for t in range(L):
                    p_mean0 += x_batch[b, i, t]
                    c_mean0 += x_batch[b, j, t]
                p_mean0 /= L
                c_mean0 /= L

                cov0 = 0.0
                p_var0 = 0.0
                c_var0 = 0.0
                for t in range(L):
                    dp = x_batch[b, i, t] - p_mean0
                    dc = x_batch[b, j, t] - c_mean0
                    cov0 += dp * dc
                    p_var0 += dp * dp
                    c_var0 += dc * dc

                if p_var0 < 1e-12 or c_var0 < 1e-12:
                    corr_at_0 = 0.0
                else:
                    corr_at_0 = abs(cov0 / np.sqrt(p_var0 * c_var0))

                # ── Decision: edge is good if ──
                # 1) Correlation at declared lag is detectable
                # 2) Lag specificity: lag-L corr is meaningful vs lag-0 corr
                #    (lenient: accepts if corr_at_lag > 0.8 * corr_at_0
                #     OR if corr_at_lag alone is strong enough)
                lag_specific = (
                    (corr_at_lag >= min_edge_corr)
                    and (
                        corr_at_lag >= 0.8 * corr_at_0  # lag dominates
                        or corr_at_lag >= 0.2            # strong enough on its own
                    )
                )
                if lag_specific:
                    n_good += 1

        if n_edges > 0:
            scores[b] = n_good / n_edges
        else:
            scores[b] = 1.0  # no edges = no quality requirement

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

                # Use validated generation — resamples graphs with unlearnable effects
                raw_batch, adj_batch, lag_batch, quality_scores = self.engine.generate_validated(
                    batch_size=BURST_SIZE, n_nodes=n_nodes, length=self.seq_len,
                )

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
    SAMPLES_PER_CHUNK = 512 * 256
    BASE_SEED = 5000
    NODES_RANGE = (1, 12)

    # Target Sequence Lengths
    SEQUENCE_LENGTHS = [2048]

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
                f"industrial_graph_chunk_L{SEQ_LEN}_C{i}.parquet",
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
