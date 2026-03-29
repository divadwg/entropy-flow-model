#!/usr/bin/env python3
"""
Statistical Complexity Analysis

Computes and plots statistical complexity (normalized_entropy * disequilibrium)
across the three base regimes and the evolving system. Reuses existing simulation
infrastructure without modifying it.

Usage:
    python run_complexity.py [config.yaml]
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.grid import Grid
from src.dynamics import flow_energy, update_states, flow_energy_with_traits, update_states_evolving
from src.experiment import build_params, REGIME_NAMES
from src.metrics import statistical_complexity, complexity_timeseries


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_and_collect_modes(config, regime, seed, n_steps=None):
    """Run a base-model simulation and return per-step output mode distributions.

    Returns: output_modes_over_time, shape (T, K)
    """
    rng = np.random.default_rng(seed)
    gc = config["grid"]
    grid = Grid(gc["height"], gc["width"], gc["n_modes"], rng)
    grid.initialize(
        active_frac=config["states"]["active_fraction"],
        passive_frac=config["states"]["passive_fraction"],
    )
    params = build_params(config)
    e_in_per_cell = config["energy"]["E_in"] / gc["width"]
    if n_steps is None:
        n_steps = config["simulation"]["n_steps"]

    modes = np.zeros((n_steps, gc["n_modes"]), dtype=np.float64)

    for step in range(n_steps):
        output, throughput, balance = flow_energy(grid, e_in_per_cell, params)
        modes[step] = output.sum(axis=0)  # sum across cells → (K,)
        update_states(grid, throughput, params, regime)

    return modes


def run_evolving_and_collect_modes(config, seed, n_steps=None):
    """Run an evolving simulation and return per-step output mode distributions.

    Returns: output_modes_over_time, shape (T, K)
    """
    rng = np.random.default_rng(seed)
    gc = config["grid"]
    evo = config.get("evolution", {})
    grid = Grid(gc["height"], gc["width"], gc["n_modes"], rng)
    grid.initialize(
        active_frac=config["states"]["active_fraction"],
        passive_frac=config["states"]["passive_fraction"],
    )
    grid.enable_traits(randomize=False)

    params = build_params(config)
    params["transform_cost"] = evo.get("transform_cost", 0.50)
    params["trait_mutation_std"] = evo.get("trait_mutation_std", 0.03)
    e_in_per_cell = config["energy"]["E_in"] / gc["width"]
    if n_steps is None:
        n_steps = evo.get("n_steps", 3000)

    modes = np.zeros((n_steps, gc["n_modes"]), dtype=np.float64)

    for step in range(n_steps):
        output, throughput, balance = flow_energy_with_traits(grid, e_in_per_cell, params)
        modes[step] = output.sum(axis=0)
        update_states_evolving(grid, throughput, params)

    return modes


def plot_complexity_over_time(results_dict, save_path):
    """Plot statistical complexity time series for each regime."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = {
        "Memoryless": "#e74c3c",
        "Persistent": "#3498db",
        "Persistent+Branching": "#2ecc71",
        "Evolving": "#9b59b6",
    }

    for label, modes_list in results_dict.items():
        color = colors.get(label, None)

        # Compute per-seed complexity
        all_c = np.array([complexity_timeseries(m) for m in modes_list])
        mean_c = all_c.mean(axis=0)
        std_c = all_c.std(axis=0)
        steps = np.arange(len(mean_c))

        # Smooth for readability
        window = max(1, len(mean_c) // 100)
        if window > 1:
            kernel = np.ones(window) / window
            mean_smooth = np.convolve(mean_c, kernel, mode="same")
            std_smooth = np.convolve(std_c, kernel, mode="same")
        else:
            mean_smooth = mean_c
            std_smooth = std_c

        ax1.plot(steps, mean_smooth, label=label, color=color, linewidth=1.5)
        ax1.fill_between(steps, mean_smooth - std_smooth, mean_smooth + std_smooth,
                         alpha=0.15, color=color)

        # Cumulative complexity
        cum_c = np.cumsum(all_c, axis=1)
        cum_mean = cum_c.mean(axis=0)
        cum_std = cum_c.std(axis=0)
        ax2.plot(steps, cum_mean, label=label, color=color, linewidth=1.5)
        ax2.fill_between(steps, cum_mean - cum_std, cum_mean + cum_std,
                         alpha=0.15, color=color)

    ax1.set_ylabel("Statistical Complexity")
    ax1.set_title("Statistical Complexity Over Time")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Cumulative Complexity")
    ax2.set_title("Cumulative Statistical Complexity")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_complexity_vs_alpha(config, save_path, n_seeds=3, n_steps=500):
    """Sweep alpha and plot mean complexity vs transform_strength."""
    import copy

    alphas = np.linspace(0.05, 0.8, 12)
    mean_complexities = []
    std_complexities = []

    for alpha in alphas:
        cfg = copy.deepcopy(config)
        cfg["dynamics"]["transform_strength"] = float(alpha)
        seed_means = []
        for seed in range(n_seeds):
            modes = run_and_collect_modes(cfg, regime=3, seed=seed, n_steps=n_steps)
            c_ts = complexity_timeseries(modes)
            # Use last 50% for steady-state
            seed_means.append(c_ts[len(c_ts)//2:].mean())
        mean_complexities.append(np.mean(seed_means))
        std_complexities.append(np.std(seed_means))

    mean_complexities = np.array(mean_complexities)
    std_complexities = np.array(std_complexities)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(alphas, mean_complexities, yerr=std_complexities,
                fmt="o-", color="#2c3e50", capsize=4, linewidth=1.5)
    ax.set_xlabel("Transform Strength (alpha)")
    ax.set_ylabel("Mean Statistical Complexity (steady-state)")
    ax.set_title("Statistical Complexity vs Transform Strength")
    ax.grid(True, alpha=0.3)

    # Mark peak
    peak_idx = np.argmax(mean_complexities)
    ax.annotate(f"peak at alpha={alphas[peak_idx]:.2f}",
                xy=(alphas[peak_idx], mean_complexities[peak_idx]),
                xytext=(alphas[peak_idx] + 0.1, mean_complexities[peak_idx] * 1.1),
                arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                fontsize=10, color="#e74c3c")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")

    return alphas, mean_complexities


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)

    n_seeds = 5
    n_steps_base = config["simulation"]["n_steps"]  # 500
    n_steps_evo = min(config.get("evolution", {}).get("n_steps", 3000), 3000)

    print("=" * 64)
    print("STATISTICAL COMPLEXITY ANALYSIS")
    print("=" * 64)

    # ── Collect output mode distributions ──────────────────────
    results = {}

    for regime, name in REGIME_NAMES.items():
        label = {"memoryless": "Memoryless", "persistent": "Persistent",
                 "persistent+branching": "Persistent+Branching"}[name]
        print(f"  Running {label} ({n_seeds} seeds, {n_steps_base} steps)...")
        modes_list = []
        for seed in range(n_seeds):
            modes = run_and_collect_modes(config, regime, seed, n_steps=n_steps_base)
            modes_list.append(modes)
        results[label] = modes_list

    print(f"  Running Evolving ({n_seeds} seeds, {n_steps_evo} steps)...")
    evo_modes_list = []
    for seed in range(n_seeds):
        modes = run_evolving_and_collect_modes(config, seed, n_steps=n_steps_evo)
        evo_modes_list.append(modes)
    results["Evolving"] = evo_modes_list

    # ── Plot complexity over time ──────────────────────────────
    print("\nGenerating plots...")
    plot_complexity_over_time(results, os.path.join(fig_dir, "complexity_over_time.png"))

    # ── Plot complexity vs alpha ───────────────────────────────
    print("  Running alpha sweep for complexity vs alpha...")
    alphas, mean_c = plot_complexity_vs_alpha(
        config, os.path.join(fig_dir, "complexity_vs_alpha.png"),
        n_seeds=3, n_steps=500,
    )

    # ── Sanity check: print summary ───────────────────────────
    print("\n" + "=" * 64)
    print("COMPLEXITY SUMMARY")
    print("=" * 64)

    for label, modes_list in results.items():
        all_c = [complexity_timeseries(m) for m in modes_list]
        # Use last 50% for steady-state mean
        ss_means = [c[len(c)//2:].mean() for c in all_c]
        overall_means = [c.mean() for c in all_c]
        print(f"  {label:25s}  mean={np.mean(overall_means):.6f}  "
              f"steady-state={np.mean(ss_means):.6f}  "
              f"std={np.std(ss_means):.6f}")

    print(f"\nAlpha sweep peak: alpha={alphas[np.argmax(mean_c)]:.2f}, "
          f"complexity={mean_c.max():.6f}")

    print(f"\nOutputs: {fig_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
