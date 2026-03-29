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
