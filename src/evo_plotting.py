"""
Visualization for the evolution experiment.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .grid import TRAIT_NAMES, TRAIT_DEFAULTS, N_TRAITS

COLORS = {"fixed": "#3498db", "evolving": "#2ecc71"}


def plot_fixed_vs_evolving(df_fixed, df_evolving, path):
    """Cumulative entropy production and per-step metrics: fixed vs evolving."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Cumulative entropy production (primary metric)
    ax1 = axes[0, 0]
    for label, df, color in [("fixed", df_fixed, COLORS["fixed"]),
                              ("evolving", df_evolving, COLORS["evolving"])]:
        pivot = df.pivot_table(index="step", columns="seed", values="entropy_production")
        cum = pivot.cumsum()
        mean = cum.mean(axis=1)
        std = cum.std(axis=1)
        ax1.plot(mean.index, mean.values, color=color, label=label.capitalize(), linewidth=2)
        ax1.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.12)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Cumulative entropy production")
    ax1.set_title("Cumulative Entropy Production (S × E_out/E_in)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top-right: Per-step entropy production
    ax2 = axes[0, 1]
    for label, df, color in [("fixed", df_fixed, COLORS["fixed"]),
                              ("evolving", df_evolving, COLORS["evolving"])]:
        pivot = df.pivot_table(index="step", columns="seed", values="entropy_production")
        inst_mean = pivot.mean(axis=1)
        inst_std = pivot.std(axis=1)
        ax2.plot(inst_mean.index, inst_mean.values, color=color,
                 label=label.capitalize(), linewidth=1.5)
        ax2.fill_between(inst_mean.index, inst_mean - inst_std, inst_mean + inst_std,
                         color=color, alpha=0.12)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Entropy production")
    ax2.set_title("Per-Step Entropy Production")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Cumulative output entropy
    ax3 = axes[1, 0]
    for label, df, color in [("fixed", df_fixed, COLORS["fixed"]),
                              ("evolving", df_evolving, COLORS["evolving"])]:
        pivot = df.pivot_table(index="step", columns="seed", values="output_entropy")
        cum = pivot.cumsum()
        mean = cum.mean(axis=1)
        std = cum.std(axis=1)
        ax3.plot(mean.index, mean.values, color=color, label=label.capitalize(), linewidth=2)
        ax3.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.12)
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Cumulative output entropy (bits)")
    ax3.set_title("Cumulative Output Entropy")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Energy throughput over time
    ax4 = axes[1, 1]
    for label, df, color in [("fixed", df_fixed, COLORS["fixed"]),
                              ("evolving", df_evolving, COLORS["evolving"])]:
        pivot = df.pivot_table(index="step", columns="seed", values="energy_out")
        inst_mean = pivot.mean(axis=1)
        inst_std = pivot.std(axis=1)
        ax4.plot(inst_mean.index, inst_mean.values, color=color,
                 label=label.capitalize(), linewidth=1.5)
        ax4.fill_between(inst_mean.index, inst_mean - inst_std, inst_mean + inst_std,
                         color=color, alpha=0.12)
    ax4.set_xlabel("Timestep")
    ax4.set_ylabel("Energy throughput")
    ax4.set_title("Energy Throughput Over Time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Fixed vs Evolving Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_trait_evolution(df_evolving, path):
    """Mean +/- std of each heritable trait over time across seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    trait_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for i, tname in enumerate(TRAIT_NAMES):
        ax = axes[i]
        col_mean = f"trait_{tname}_mean"
        col_std = f"trait_{tname}_std"

        if col_mean not in df_evolving.columns:
            continue

        pivot_mean = df_evolving.pivot_table(index="step", columns="seed", values=col_mean)
        grand_mean = pivot_mean.mean(axis=1)
        grand_std = pivot_mean.std(axis=1)

        ax.plot(grand_mean.index, grand_mean.values,
                color=trait_colors[i], linewidth=2, label="Population mean")
        ax.fill_between(grand_mean.index,
                        grand_mean - grand_std, grand_mean + grand_std,
                        color=trait_colors[i], alpha=0.15, label="Across-seed std")

        # Within-population std (mean across seeds)
        if col_std in df_evolving.columns:
            pivot_std = df_evolving.pivot_table(index="step", columns="seed", values=col_std)
            within_mean = pivot_std.mean(axis=1)
            ax.plot(within_mean.index, within_mean.values,
                    color=trait_colors[i], linewidth=1, linestyle="--",
                    alpha=0.7, label="Within-pop std")

        ax.axhline(TRAIT_DEFAULTS[i], color="gray", linestyle=":", alpha=0.5,
                   label=f"Default ({TRAIT_DEFAULTS[i]})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(tname)
        ax.set_title(f"Trait: {tname}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Trait Evolution Over Time", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_trait_histograms(all_snapshots, path):
    """Trait value histograms at different timepoints.

    all_snapshots: {seed: {step: (N, N_TRAITS) array}}
    """
    # Collect all snapshot steps
    all_steps = set()
    for seed_snaps in all_snapshots.values():
        all_steps.update(seed_snaps.keys())
    steps = sorted(all_steps)

    if len(steps) == 0:
        return

    n_steps_plot = min(len(steps), 5)
    step_indices = np.linspace(0, len(steps) - 1, n_steps_plot, dtype=int)
    plot_steps = [steps[i] for i in step_indices]

    fig, axes = plt.subplots(N_TRAITS, n_steps_plot, figsize=(3 * n_steps_plot, 3 * N_TRAITS))
    if N_TRAITS == 1:
        axes = axes[np.newaxis, :]
    if n_steps_plot == 1:
        axes = axes[:, np.newaxis]

    trait_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for col, step in enumerate(plot_steps):
        # Pool traits across seeds for this step
        all_vals = []
        for seed_snaps in all_snapshots.values():
            if step in seed_snaps and len(seed_snaps[step]) > 0:
                all_vals.append(seed_snaps[step])

        if not all_vals:
            continue
        pooled = np.vstack(all_vals)

        for row in range(N_TRAITS):
            ax = axes[row, col]
            vals = pooled[:, row]
            ax.hist(vals, bins=20, color=trait_colors[row], alpha=0.7, edgecolor="white")
            ax.axvline(TRAIT_DEFAULTS[row], color="gray", linestyle="--", alpha=0.7)
            if col == 0:
                ax.set_ylabel(TRAIT_NAMES[row])
            if row == 0:
                ax.set_title(f"t={step}")
            if row == N_TRAITS - 1:
                ax.set_xlabel("Trait value")

    plt.suptitle("Trait Distributions Over Time", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_lineage_dominance(all_lineage, path):
    """Track top lineages over time.

    all_lineage: {seed: [step_dict, ...]} where step_dict = {lineage_id: count}
    """
    # Use first seed for detailed lineage tracking
    seed = min(all_lineage.keys())
    history = all_lineage[seed]
    n_steps = len(history)

    if n_steps == 0:
        return

    # Find the top-10 lineages by total cell-steps
    all_ids = {}
    for t, counts in enumerate(history):
        for lid, cnt in counts.items():
            all_ids[lid] = all_ids.get(lid, 0) + cnt

    if not all_ids:
        return

    top_ids = sorted(all_ids, key=all_ids.get, reverse=True)[:10]
    cmap = plt.cm.tab10

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Stacked area: top lineages
    time = np.arange(n_steps)
    lineage_counts = np.zeros((n_steps, len(top_ids)))
    for t, counts in enumerate(history):
        for j, lid in enumerate(top_ids):
            lineage_counts[t, j] = counts.get(lid, 0)

    ax1.stackplot(time, lineage_counts.T, labels=[f"L{lid}" for lid in top_ids],
                  colors=[cmap(i) for i in range(len(top_ids))], alpha=0.7)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Cell count")
    ax1.set_title(f"Lineage Dominance Over Time (seed {seed})")
    ax1.legend(fontsize=7, loc="upper left", ncol=2)
    ax1.grid(True, alpha=0.3)

    # Number of distinct lineages over time (all seeds)
    for s, hist in all_lineage.items():
        n_lin = [len(counts) for counts in hist]
        ax2.plot(range(len(n_lin)), n_lin, alpha=0.4, linewidth=0.8)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Distinct lineages")
    ax2.set_title("Lineage Diversity Over Time (all seeds)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mutation_sweep(sweep_df, path):
    """Cumulative entropy vs mutation rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cum = (sweep_df.groupby(["sweep_value", "seed"])["output_entropy"]
           .sum().reset_index().rename(columns={"output_entropy": "cum_entropy"}))
    means = cum.groupby("sweep_value")["cum_entropy"].mean()
    stds = cum.groupby("sweep_value")["cum_entropy"].std()

    ax1.errorbar(means.index, means.values, yerr=stds.values,
                 color="#2ecc71", linewidth=2, marker="o", markersize=6, capsize=5)
    ax1.set_xlabel("Trait mutation std")
    ax1.set_ylabel("Cumulative entropy (bits)")
    ax1.set_title("Cumulative Entropy vs Mutation Rate")
    ax1.grid(True, alpha=0.3)

    # Final transform_strength
    final_step = sweep_df["step"].max()
    late = sweep_df[sweep_df["step"] > final_step * 0.8]
    col = "trait_transform_strength_mean"
    if col in late.columns:
        alpha_means = late.groupby("sweep_value")[col].mean()
        alpha_stds = late.groupby("sweep_value")[col].std()
        ax2.errorbar(alpha_means.index, alpha_means.values, yerr=alpha_stds.values,
                     color="#e74c3c", linewidth=2, marker="o", markersize=6, capsize=5)
        ax2.axhline(0.3, color="gray", linestyle="--", alpha=0.5, label="Default (0.3)")
        ax2.set_ylabel("Evolved transform_strength")
        ax2.legend()
    ax2.set_xlabel("Trait mutation std")
    ax2.set_title("Evolved Transform Strength vs Mutation Rate")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_ablation(ablation_df, path):
    """Bar chart comparing trait ablation variants."""
    cum = (ablation_df.groupby(["variant", "seed"])["output_entropy"]
           .sum().reset_index().rename(columns={"output_entropy": "cum_entropy"}))
    means = cum.groupby("variant")["cum_entropy"].mean()
    stds = cum.groupby("variant")["cum_entropy"].std()

    variants = sorted(means.index)
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(variants))
    bars = ax.bar(x, [means[v] for v in variants],
                  yerr=[stds[v] for v in variants],
                  color=colors[:len(variants)], alpha=0.8, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylabel("Cumulative entropy (bits)")
    ax.set_title("Trait Ablation: Which Traits Matter?")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_energy_vs_entropy_tradeoff(df_fixed, df_evolving, path):
    """Show the efficiency/entropy tradeoff: fixed vs evolved."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, df, color, marker in [("Fixed", df_fixed, COLORS["fixed"], "s"),
                                      ("Evolving", df_evolving, COLORS["evolving"], "o")]:
        final_step = df["step"].max()
        late = df[df["step"] > final_step * 0.8]
        per_seed = late.groupby("seed").agg(
            entropy=("output_entropy", "mean"),
            energy=("energy_out", "mean"),
        )
        ax.scatter(per_seed["energy"], per_seed["entropy"],
                   color=color, marker=marker, s=60, alpha=0.7, label=label)

    ax.set_xlabel("Energy throughput")
    ax.set_ylabel("Output entropy (bits)")
    ax.set_title("Efficiency vs Entropy Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
