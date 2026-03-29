"""
Entropy and throughput metrics for the output energy distribution.

Two key metrics:
1. Output entropy: Shannon entropy of the mode distribution at the bottom row.
   Higher = more spread out output = more "thermodynamic work" done by channels.
2. Fragmentation: count of active (cell, mode) bins at output, measuring how
   many distinct output channels carry energy.
"""
import numpy as np
from .grid import ACTIVE, REPLICATING


def output_entropy(output):
    """Shannon entropy (bits) of the output mode distribution.

    Sums energy per mode across all output cells, normalizes to a
    probability distribution, and computes -sum(p * log2(p)).
    """
    mode_totals = output.sum(axis=0)  # (K,) energy per mode
    total = mode_totals.sum()
    if total < 1e-12:
        return 0.0
    p = mode_totals / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def fragmentation(output, threshold=0.01):
    """Count of (cell, mode) pairs at output with energy above threshold."""
    return int((output > threshold).sum())


def effective_modes(output):
    """Effective number of output modes: exp(Shannon entropy in nats)."""
    mode_totals = output.sum(axis=0)
    total = mode_totals.sum()
    if total < 1e-12:
        return 1.0
    p = mode_totals / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def kl_from_uniform(output):
    """KL divergence D(output || uniform).

    Measures how far the output distribution is from maximum entropy
    (uniform / blackbody-like). Lower = closer to uniform.
    """
    mode_totals = output.sum(axis=0)
    total = mode_totals.sum()
    if total < 1e-12:
        return 0.0
    K = len(mode_totals)
    p = mode_totals / total
    q = 1.0 / K
    p_pos = p[p > 0]
    return float(np.sum(p_pos * np.log2(p_pos / q)))


def shannon_entropy_nats(p, eps=1e-12):
    """Shannon entropy in nats (natural log) of a distribution."""
    p = np.asarray(p, dtype=float)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(p, eps=1e-12):
    """Shannon entropy normalized to [0, 1] by dividing by log(K)."""
    k = len(p)
    if k <= 1:
        return 0.0
    return shannon_entropy_nats(p, eps) / np.log(k)


def disequilibrium(p):
    """Euclidean distance squared from uniform distribution."""
    p = np.asarray(p, dtype=float)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    k = len(p)
    uniform = np.full(k, 1.0 / k)
    return float(np.sum((p - uniform) ** 2))


def statistical_complexity(p):
    """Statistical complexity: normalized_entropy * disequilibrium.

    Peaks at intermediate structure — neither fully ordered nor fully random.
    """
    return normalized_entropy(p) * disequilibrium(p)


def complexity_timeseries(output_modes_over_time):
    """Compute statistical complexity for each row of a (T, K) array."""
    return np.array([statistical_complexity(row) for row in output_modes_over_time])


def collect_step_metrics(output, cell_throughput, grid, balance, step):
    """Gather all metrics for one timestep into a flat dict."""
    return {
        "step": step,
        "output_entropy": output_entropy(output),
        "fragmentation": fragmentation(output),
        "effective_modes": effective_modes(output),
        "kl_from_uniform": kl_from_uniform(output),
        "energy_in": balance["energy_in"],
        "energy_out": balance["energy_out"],
        "energy_lost": balance["energy_lost"],
        "throughput_mean": float(cell_throughput.mean()),
        "active_area": grid.active_area(),
        "mean_age": grid.mean_channel_age(),
        "n_replicating": int((grid.states == REPLICATING).sum()),
    }
