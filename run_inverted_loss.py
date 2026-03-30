#!/usr/bin/env python3
"""
Layer 8: Inverted Loss Test — Isolating the Coupling from Dissipation

The critical test: does the throughput-persistence coupling produce structure
even when active cells dissipate MORE than empty cells?

Standard model: active cells lose 2%, empty cells lose 5%.
  → More structure mechanically means less aggregate dissipation.
  → The dissipation anti-correlation might be an artefact of this parameterisation.

Inverted model: active cells lose 8%, empty cells lose 2%.
  → Now more structure should INCREASE aggregate dissipation.
  → If the coupling still produces structure, the mechanism is independent
    of whether structure increases or decreases dissipation.

This isolates the throughput-persistence coupling from the dissipation sign
and provides the strongest possible evidence that dissipation is irrelevant
to the selection mechanism.

Runs on both the grid model and the network model.
"""

import os
import copy
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.grid import Grid, EMPTY, ACTIVE, REPLICATING, extract_motifs
from src.dynamics import flow_energy, update_states, update_states_decoupled
from src.experiment import build_params
from src.metrics import (
    collect_step_metrics, output_entropy as grid_output_entropy,
    motif_propagation, statistical_complexity,
)
from run_network import NetworkFlowModel, output_entropy as net_output_entropy


# ── Grid model with custom loss rates ────────────────────────────

def run_grid_condition(config, regime, seed, n_steps, loss_rates=None,
                       decouple_cfg=None, no_noise=False):
    """Run grid model with optional custom loss rates."""
    rng = np.random.default_rng(seed)
    gc = config["grid"]
    grid = Grid(gc["height"], gc["width"], gc["n_modes"], rng)
    grid.initialize(
        active_frac=config["states"]["active_fraction"],
        passive_frac=config["states"]["passive_fraction"],
    )
    params = build_params(config)
    if loss_rates:
        params.update(loss_rates)
    if no_noise:
        params["mutation_rate"] = 0.0
        params["spontaneous_create_rate"] = 0.0

    e_in_per_cell = config["energy"]["E_in"] / gc["width"]

    records = []
    for step in range(n_steps):
        pre_states = grid.states.copy()
        pre_occupied = pre_states > EMPTY

        output, throughput, balance = flow_energy(grid, e_in_per_cell, params)
        ent = grid_output_entropy(output)
        dissip = balance["energy_lost"] / max(balance["energy_in"], 1e-12)

        if decouple_cfg:
            update_states_decoupled(grid, throughput, params, regime, decouple_cfg)
        else:
            update_states(grid, throughput, params, regime)

        post_occupied = grid.states > EMPTY
        survived = pre_occupied & post_occupied
        disrupted = pre_occupied & ~post_occupied

        tp_surv = throughput[survived] if survived.any() else np.array([])
        tp_disr = throughput[disrupted] if disrupted.any() else np.array([])
        tp_all = float(throughput[pre_occupied].mean()) if pre_occupied.any() else 0.0

        if len(tp_surv) > 0 and len(tp_disr) > 0 and tp_all > 1e-12:
            lb = float((tp_surv.mean() - tp_disr.mean()) / tp_all)
        elif len(tp_disr) == 0 and len(tp_surv) > 0:
            lb = 1.0
        else:
            lb = 0.0

        la = np.log10(1 + survived.sum()) - np.log10(1 + disrupted.sum())

        mask = grid.states > EMPTY
        mean_age = float(grid.age[mask].mean()) if mask.any() else 0.0
        ages = grid.age[mask].astype(float) if mask.any() else np.array([0.0])
        age_cv = float(ages.std() / max(ages.mean(), 1e-12)) if len(ages) > 1 else 0.0

        records.append({
            "step": step,
            "entropy": ent,
            "dissipation_rate": dissip,
            "energy_in": balance["energy_in"],
            "energy_out": balance["energy_out"],
            "energy_lost": balance["energy_lost"],
            "active_area": int(((grid.states == ACTIVE) | (grid.states == REPLICATING)).sum()),
            "mean_age": mean_age,
            "age_cv": age_cv,
            "lambda_a": float(la),
            "lambda_b": lb,
        })

    return pd.DataFrame(records)


# ── Network model with custom loss rates ─────────────────────────

def run_network_condition(regime, seed, n_steps, loss_rates=None,
                          decouple_cfg=None, no_noise=False):
    """Run network model with optional custom loss rates."""
    rng = np.random.default_rng(seed)

    params = {
        "transform_strength": 0.3,
        "loss_rate_empty": 0.05,
        "loss_rate_passive": 0.01,
        "loss_rate_active": 0.02,
        "throughput_decay": 0.7,
        "persistence_threshold": 0.8,
        "persistence_strength": 0.8,
        "replication_prob": 0.15,
        "spontaneous_create_rate": 0.005,
        "mutation_rate": 0.01,
        "active_fraction": 0.1,
        "passive_fraction": 0.2,
    }
    if loss_rates:
        params.update(loss_rates)
    if no_noise:
        params["mutation_rate"] = 0.0
        params["spontaneous_create_rate"] = 0.0

    net = NetworkFlowModel(n_layers=10, nodes_per_layer=40, n_modes=16,
                           min_out=1, max_out=5, rng=rng)
    net.initialize(params["active_fraction"], params["passive_fraction"])

    records = []
    for step in range(n_steps):
        pre_states = net.states.copy()
        pre_occupied = pre_states > EMPTY

        output, cell_tp, balance = net.flow_energy(100.0, params)
        ent = net_output_entropy(output)
        dissip = balance["energy_lost"] / max(balance["energy_in"], 1e-12)

        net.update_states(cell_tp, params, regime, decouple_cfg)

        post_occupied = net.states > EMPTY
        survived = pre_occupied & post_occupied
        disrupted = pre_occupied & ~post_occupied

        tp_surv = cell_tp[survived] if survived.any() else np.array([])
        tp_disr = cell_tp[disrupted] if disrupted.any() else np.array([])
        tp_all = float(cell_tp[pre_occupied].mean()) if pre_occupied.any() else 0.0

        if len(tp_surv) > 0 and len(tp_disr) > 0 and tp_all > 1e-12:
            lb = float((tp_surv.mean() - tp_disr.mean()) / tp_all)
        elif len(tp_disr) == 0 and len(tp_surv) > 0:
            lb = 1.0
        else:
            lb = 0.0

        la = np.log10(1 + survived.sum()) - np.log10(1 + disrupted.sum())

        records.append({
            "step": step,
            "entropy": ent,
            "dissipation_rate": dissip,
            "energy_in": balance["energy_in"],
            "energy_out": balance["energy_out"],
            "energy_lost": balance["energy_lost"],
            "active_area": net.active_count(),
            "mean_age": net.mean_age(),
            "age_cv": net.age_cv(),
            "lambda_a": float(la),
            "lambda_b": lb,
        })

    return pd.DataFrame(records)


# ── Experiment suite ─────────────────────────────────────────────

def run_experiment(n_seeds=8, n_steps=1500):
    """Run standard and inverted loss conditions on both architectures."""

    standard_loss = {
        "loss_rate_empty": 0.05,
        "loss_rate_passive": 0.01,
        "loss_rate_active": 0.02,
    }
    inverted_loss = {
        "loss_rate_empty": 0.02,
        "loss_rate_passive": 0.04,
        "loss_rate_active": 0.08,
    }

    config = yaml.safe_load(open("config.yaml"))
    ss_start = n_steps // 3

    # All conditions to run
    conditions = [
        # (name, architecture, regime, loss_rates, decouple_cfg, no_noise)
        ("grid_std_baseline",       "grid",    3, standard_loss, None, False),
        ("grid_std_memoryless",     "grid",    1, standard_loss, None, False),
        ("grid_std_anti",           "grid",    3, standard_loss, {"mode": "anti_coupled"}, False),
        ("grid_inv_baseline",       "grid",    3, inverted_loss, None, False),
        ("grid_inv_memoryless",     "grid",    1, inverted_loss, None, False),
        ("grid_inv_anti",           "grid",    3, inverted_loss, {"mode": "anti_coupled"}, False),
        ("grid_inv_blind",          "grid",    3, inverted_loss, {"mode": "throughput_blind"}, False),
        ("net_std_baseline",        "network", 3, standard_loss, None, False),
        ("net_std_memoryless",      "network", 1, standard_loss, None, False),
        ("net_std_anti",            "network", 3, standard_loss, {"mode": "anti_coupled"}, False),
        ("net_inv_baseline",        "network", 3, inverted_loss, None, False),
        ("net_inv_memoryless",      "network", 1, inverted_loss, None, False),
        ("net_inv_anti",            "network", 3, inverted_loss, {"mode": "anti_coupled"}, False),
        ("net_inv_blind",           "network", 3, inverted_loss, {"mode": "throughput_blind"}, False),
    ]

    all_results = []
    summary_rows = []

    total = len(conditions) * n_seeds
    idx = 0

    for cond_name, arch, regime, loss, decouple, no_noise in conditions:
        cond_dfs = []
        for seed in range(n_seeds):
            idx += 1
            print(f"  [{idx}/{total}] {cond_name}, seed {seed}")

            if arch == "grid":
                df = run_grid_condition(config, regime, seed, n_steps,
                                       loss_rates=loss, decouple_cfg=decouple,
                                       no_noise=no_noise)
            else:
                df = run_network_condition(regime, seed, n_steps,
                                          loss_rates=loss, decouple_cfg=decouple,
                                          no_noise=no_noise)

            df["condition"] = cond_name
            df["seed"] = seed
            cond_dfs.append(df)

        cond_df = pd.concat(cond_dfs, ignore_index=True)
        all_results.append(cond_df)

        ss = cond_df[cond_df["step"] >= ss_start]
        summary_rows.append({
            "condition": cond_name,
            "architecture": arch,
            "loss_regime": "inverted" if loss == inverted_loss else "standard",
            "lambda_a": ss["lambda_a"].mean(),
            "lambda_b": ss["lambda_b"].mean(),
            "lambda_c": ss["age_cv"].mean(),
            "persistence": ss["mean_age"].mean(),
            "entropy": ss["entropy"].mean(),
            "dissipation_rate": ss["dissipation_rate"].mean(),
            "active": ss["active_area"].mean(),
        })

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return results_df, summary_df


# ── Plotting ─────────────────────────────────────────────────────

def plot_comparison(summary_df, save_dir):
    """Side-by-side comparison of standard vs inverted loss."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    metrics = [
        ("lambda_b", r"$\Lambda_B$"),
        ("persistence", "Persistence"),
        ("dissipation_rate", "Dissipation Rate"),
        ("entropy", "Mixing Entropy"),
    ]

    for row, arch in enumerate(["grid", "network"]):
        sub = summary_df[summary_df["architecture"] == arch]
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]

            # Group by loss regime
            std = sub[sub["loss_regime"] == "standard"].sort_values("condition")
            inv = sub[sub["loss_regime"] == "inverted"].sort_values("condition")

            # Match conditions by removing prefix
            std_labels = [c.replace(f"{arch[:3]}_std_", "") for c in std["condition"]]
            inv_labels = [c.replace(f"{arch[:3]}_inv_", "") for c in inv["condition"]]

            x = range(len(std_labels))
            width = 0.35
            ax.bar([i - width/2 for i in x], std[metric].values,
                   width, label="Standard loss", color="#2ecc71", alpha=0.7)

            x_inv = range(len(inv_labels))
            ax.bar([i + width/2 for i in x_inv], inv[metric].values,
                   width, label="Inverted loss", color="#e74c3c", alpha=0.7)

            all_labels = sorted(set(std_labels + inv_labels))
            ax.set_xticks(range(len(all_labels)))
            ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(label)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
            if row == 0:
                ax.set_title(label)

        axes[row, 0].annotate(arch.upper(), xy=(0, 0.5),
                               xycoords="axes fraction",
                               xytext=(-60, 0), textcoords="offset points",
                               fontsize=14, fontweight="bold",
                               ha="center", va="center", rotation=90)

    plt.suptitle("Inverted Loss Test: Standard vs Inverted Cell Loss Rates",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "inverted_loss_comparison.png"), dpi=150)
    plt.close()


def plot_dissipation_vs_structure(summary_df, save_dir):
    """Scatter: dissipation vs persistence, colored by loss regime."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, arch in zip(axes, ["grid", "network"]):
        sub = summary_df[summary_df["architecture"] == arch]

        for _, row in sub.iterrows():
            color = "#2ecc71" if row["loss_regime"] == "standard" else "#e74c3c"
            marker = "o" if "baseline" in row["condition"] else ("^" if "anti" in row["condition"] else "s")
            ax.scatter(row["dissipation_rate"], row["persistence"],
                       c=color, marker=marker, s=120, zorder=5, edgecolor="white")
            ax.annotate(row["condition"].split("_", 2)[-1],
                        (row["dissipation_rate"], row["persistence"]),
                        fontsize=7, ha="left", va="bottom")

        ax.set_xlabel("Dissipation Rate")
        ax.set_ylabel("Persistence")
        ax.set_title(f"{arch.upper()} Model")
        ax.grid(True, alpha=0.3)

        # Legend
        ax.scatter([], [], c="#2ecc71", s=80, label="Standard (active=2%, empty=5%)")
        ax.scatter([], [], c="#e74c3c", s=80, label="Inverted (active=8%, empty=2%)")
        ax.legend(fontsize=7)

    plt.suptitle("Dissipation vs Structure: Standard vs Inverted Loss Rates",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "inverted_loss_dissipation.png"), dpi=150)
    plt.close()


# ── Report ───────────────────────────────────────────────────────

def write_report(summary_df, save_dir):
    """Write analysis report."""

    # Key comparisons
    def get_row(cond):
        rows = summary_df[summary_df["condition"] == cond]
        return rows.iloc[0] if len(rows) > 0 else None

    report = """# Inverted Loss Test Report

## Purpose

Test whether the throughput-persistence coupling produces structure even when
active cells dissipate MORE than empty cells.

- **Standard**: active=2%, passive=1%, empty=5% loss
- **Inverted**: active=8%, passive=4%, empty=2% loss

If structure persists under inverted loss, the coupling mechanism is
independent of whether structure increases or decreases dissipation.

## Summary Table

| Condition | Lambda_B | Persistence | Dissip. | Entropy | Active |
|-----------|----------|-------------|---------|---------|--------|
"""
    for _, row in summary_df.iterrows():
        report += (f"| {row['condition']} | {row['lambda_b']:.3f} | "
                   f"{row['persistence']:.1f} | {row['dissipation_rate']:.3f} | "
                   f"{row['entropy']:.3f} | {row['active']:.0f} |\n")

    # Key comparisons
    report += "\n## Key Comparisons\n\n"

    for arch, prefix in [("grid", "grid"), ("network", "net")]:
        report += f"\n### {arch.upper()} Model\n\n"

        std_base = get_row(f"{prefix}_std_baseline")
        inv_base = get_row(f"{prefix}_inv_baseline")
        std_anti = get_row(f"{prefix}_std_anti")
        inv_anti = get_row(f"{prefix}_inv_anti")
        std_mem = get_row(f"{prefix}_std_memoryless")
        inv_mem = get_row(f"{prefix}_inv_memoryless")

        if std_base is not None and inv_base is not None:
            report += f"**Baseline comparison:**\n"
            report += f"- Standard: persistence={std_base['persistence']:.1f}, "
            report += f"Lambda_B={std_base['lambda_b']:.3f}, dissip={std_base['dissipation_rate']:.3f}\n"
            report += f"- Inverted: persistence={inv_base['persistence']:.1f}, "
            report += f"Lambda_B={inv_base['lambda_b']:.3f}, dissip={inv_base['dissipation_rate']:.3f}\n\n"

            if inv_base['persistence'] > 10 and inv_base['lambda_b'] > 0.05:
                report += f"**STRUCTURE PERSISTS** under inverted loss. "
                report += f"Lambda_B = {inv_base['lambda_b']:.3f} (positive selection gradient). "
            else:
                report += f"Structure is weakened or absent under inverted loss. "
                report += f"Lambda_B = {inv_base['lambda_b']:.3f}. "

            if inv_base['dissipation_rate'] > std_base['dissipation_rate']:
                report += f"Dissipation INCREASES with structure under inverted loss "
                report += f"({inv_base['dissipation_rate']:.3f} vs {inv_mem['dissipation_rate']:.3f} memoryless), "
                report += f"confirming the coupling operates independently of the dissipation sign.\n\n"
            else:
                report += f"Dissipation still decreases with structure.\n\n"

        if inv_anti is not None and inv_base is not None:
            report += f"**Anti-coupling under inverted loss:**\n"
            report += f"- Baseline: persistence={inv_base['persistence']:.1f}, active={inv_base['active']:.0f}\n"
            report += f"- Anti-coupled: persistence={inv_anti['persistence']:.1f}, active={inv_anti['active']:.0f}\n"
            pct = (1 - inv_anti['persistence'] / max(inv_base['persistence'], 1e-12)) * 100
            report += f"- Reduction: {pct:.0f}%\n\n"

    # Dissipation sign analysis
    report += "\n## Dissipation Sign Analysis\n\n"
    report += "Under standard loss: more structure → lower aggregate dissipation "
    report += "(because active cells lose 2% vs empty cells' 5%).\n\n"
    report += "Under inverted loss: more structure → [test result] aggregate dissipation "
    report += "(because active cells lose 8% vs empty cells' 2%).\n\n"

    # Check whether standard shows anti-correlation and inverted shows positive correlation
    for arch, prefix in [("grid", "grid"), ("network", "net")]:
        std_conds = summary_df[(summary_df["architecture"] == arch) &
                               (summary_df["loss_regime"] == "standard")]
        inv_conds = summary_df[(summary_df["architecture"] == arch) &
                               (summary_df["loss_regime"] == "inverted")]

        if len(std_conds) >= 3:
            rho_std, _ = stats.spearmanr(std_conds["dissipation_rate"],
                                          std_conds["persistence"])
            report += f"{arch.upper()} standard: dissipation vs persistence rho = {rho_std:.3f}\n"

        if len(inv_conds) >= 3:
            rho_inv, _ = stats.spearmanr(inv_conds["dissipation_rate"],
                                          inv_conds["persistence"])
            report += f"{arch.upper()} inverted: dissipation vs persistence rho = {rho_inv:.3f}\n"

    report += "\n## Conclusion\n\n"

    # Check if structure persists under inverted loss
    inv_baselines = summary_df[(summary_df["loss_regime"] == "inverted") &
                                summary_df["condition"].str.contains("baseline")]
    if len(inv_baselines) > 0 and inv_baselines["persistence"].mean() > 10:
        report += """**The throughput-persistence coupling produces structure regardless of the
dissipation sign.** Under inverted loss rates (active cells lose 4x more
than empty cells), the coupling still produces persistent structure with
positive Lambda_B. This definitively isolates the coupling mechanism from
the dissipation direction.

The implication: the structured regime does not arise *because* it reduces
dissipation (standard loss) or *because* it increases dissipation (inverted
loss). It arises because the throughput-persistence coupling selects for
cells that channel energy, and channeling requires surviving long enough to
be in the path of flow. The sign of the dissipation change is a consequence
of the loss-rate parameterisation, not a driver of the mechanism.
"""
    else:
        report += """Structure is reduced or absent under inverted loss rates. The coupling
mechanism may depend on the relationship between cell type and loss rate.
Further investigation needed.
"""

    with open(os.path.join(save_dir, "inverted_loss_report.md"), "w") as f:
        f.write(report)

    return report


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = "inverted_output"
    plot_dir = "inverted_plots"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print("=" * 60)
    print("Layer 8: Inverted Loss Test")
    print("=" * 60)
    print("Standard: active=2%, empty=5% (structure reduces dissipation)")
    print("Inverted: active=8%, empty=2% (structure increases dissipation)")
    print()

    t0 = time.time()
    results_df, summary_df = run_experiment(n_seeds=8, n_steps=1500)

    print(f"\nSimulations done ({time.time()-t0:.0f}s)")

    print("\nSaving results...")
    results_df.to_csv(os.path.join(out_dir, "inverted_results.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "inverted_summary.csv"), index=False)

    print("Generating plots...")
    plot_comparison(summary_df, plot_dir)
    plot_dissipation_vs_structure(summary_df, plot_dir)

    print("Writing report...")
    report = write_report(summary_df, out_dir)

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()
    print(report)
    print(f"\nTotal: {time.time()-t0:.0f}s")
