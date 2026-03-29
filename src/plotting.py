"""
Visualization for the entropy flow model.

All plots compare the three regimes with consistent colors and show
mean +/- std across seeds.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {1: "#e74c3c", 2: "#3498db", 3: "#2ecc71"}
NAMES = {1: "Memoryless", 2: "Persistent", 3: "Persistent + Branching"}


def _regime_timeseries(df, metric, ax, ylabel, title):
    """Plot a metric over time for all 3 regimes with mean/std bands."""
    for regime in [1, 2, 3]:
        sub = df[df["regime"] == regime]
        pivot = sub.pivot_table(index="step", columns="seed", values=metric)
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1)
        ax.plot(mean.index, mean.values, color=COLORS[regime],
                label=NAMES[regime], linewidth=1.5)
        ax.fill_between(mean.index, mean - std, mean + std,
                        color=COLORS[regime], alpha=0.12)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_cumulative_entropy(df, path):
    """Cumulative output entropy over time — the main hypothesis test."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for regime in [1, 2, 3]:
        sub = df[df["regime"] == regime]
        pivot = sub.pivot_table(index="step", columns="seed", values="output_entropy")
        cum = pivot.cumsum()
        mean = cum.mean(axis=1)
        std = cum.std(axis=1)
        ax.plot(mean.index, mean.values, color=COLORS[regime],
                label=NAMES[regime], linewidth=2)
        ax.fill_between(mean.index, mean - std, mean + std,
                        color=COLORS[regime], alpha=0.12)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative output entropy (bits)")
    ax.set_title("Cumulative Output Entropy Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_output_entropy(df, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    _regime_timeseries(df, "output_entropy", ax,
                       "Output entropy (bits)",
                       "Output Distribution Entropy Over Time")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_fragmentation(df, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    _regime_timeseries(df, "fragmentation", ax,
                       "Fragmentation (active output bins)",
                       "Output Fragmentation Over Time")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_structure(df, path):
    """Active area and mean channel age side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _regime_timeseries(df, "active_area", ax1,
                       "Active cells", "Active / Transforming Area")
    _regime_timeseries(df, "mean_age", ax2,
                       "Mean channel age (steps)", "Channel Persistence")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_energy_balance(df, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    _regime_timeseries(df, "energy_out", ax,
                       "Energy exported (bottom row)",
                       "Energy Throughput Over Time")
    if len(df) > 0:
        ax.axhline(df["energy_in"].iloc[0], color="gray", linestyle="--",
                   alpha=0.5, label="Energy in")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_kl_divergence(df, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    _regime_timeseries(df, "kl_from_uniform", ax,
                       "KL(output || uniform) bits",
                       "Distance from Uniform (Blackbody) Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_effective_modes(df, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    _regime_timeseries(df, "effective_modes", ax,
                       "Effective output modes",
                       "Effective Number of Output Modes Over Time")
    if len(df) > 0:
        K = 16  # max modes
        ax.axhline(K, color="gray", linestyle="--", alpha=0.5, label=f"Max ({K})")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_sweep(sweep_df, param_name, path):
    """Cumulative entropy and final entropy vs sweep parameter for all regimes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    final_step = sweep_df["step"].max()

    # Cumulative entropy
    cum = (sweep_df.groupby(["regime", "sweep_value", "seed"])["output_entropy"]
           .sum().reset_index().rename(columns={"output_entropy": "cum_entropy"}))
    for regime in [1, 2, 3]:
        sub = cum[cum["regime"] == regime]
        means = sub.groupby("sweep_value")["cum_entropy"].mean()
        stds = sub.groupby("sweep_value")["cum_entropy"].std()
        ax1.errorbar(means.index, means.values, yerr=stds.values,
                     color=COLORS[regime], label=NAMES[regime],
                     linewidth=2, marker="o", markersize=5, capsize=4)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel("Cumulative output entropy (bits)")
    ax1.set_title(f"Cumulative Entropy vs {param_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Final output entropy (mean of last 20%)
    late = sweep_df[sweep_df["step"] > final_step * 0.8]
    final = (late.groupby(["regime", "sweep_value", "seed"])["output_entropy"]
             .mean().reset_index())
    for regime in [1, 2, 3]:
        sub = final[final["regime"] == regime]
        means = sub.groupby("sweep_value")["output_entropy"].mean()
        stds = sub.groupby("sweep_value")["output_entropy"].std()
        ax2.errorbar(means.index, means.values, yerr=stds.values,
                     color=COLORS[regime], label=NAMES[regime],
                     linewidth=2, marker="o", markersize=5, capsize=4)
    ax2.set_xlabel(param_name)
    ax2.set_ylabel("Output entropy (bits)")
    ax2.set_title(f"Steady-State Entropy vs {param_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_phase_diagram(sweep_results, path):
    """Bar charts showing regime 3 advantage over regime 1 for each sweep."""
    params = list(sweep_results.keys())
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, param in enumerate(params):
        df = sweep_results[param]
        cum = (df.groupby(["regime", "sweep_value", "seed"])["output_entropy"]
               .sum().reset_index())
        r1 = cum[cum["regime"] == 1].groupby("sweep_value")["output_entropy"].mean()
        r3 = cum[cum["regime"] == 3].groupby("sweep_value")["output_entropy"].mean()
        advantage = r3 - r1

        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in advantage.values]
        axes[idx].bar(range(len(advantage)), advantage.values, color=colors)
        axes[idx].set_xticks(range(len(advantage)))
        axes[idx].set_xticklabels([f"{v:.3g}" for v in advantage.index],
                                  rotation=45, ha="right")
        axes[idx].set_xlabel(param)
        axes[idx].set_ylabel("Cum. entropy advantage (R3 - R1)")
        axes[idx].set_title(param)
        axes[idx].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle("Branching Advantage Over Memoryless by Parameter", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
