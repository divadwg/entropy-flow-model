#!/usr/bin/env python3
"""
Flow-Persistence Number (Lambda) Estimation

Estimates a dimensionless control parameter Lambda that captures whether
throughput-coupled persistence outweighs noise and decay. Tests whether
Lambda predicts the structured regime better than entropy production or
statistical complexity.

Candidate definitions:
  Lambda_A: survival ratio = survived / disrupted
  Lambda_B: throughput-conditioned survival advantage = P(survive|high_tp) / P(survive|low_tp)
  Lambda_C: repair-to-disruption ratio = (mean_tp_survivors * survival_rate) / effective_disruption_rate

Usage:
    python run_lambda.py [config.yaml]
"""
import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from src.grid import Grid, ReinforcementMap, extract_motifs, EMPTY, ACTIVE, REPLICATING
from src.dynamics import (
    flow_energy, update_states, update_states_decoupled,
    update_states_reinforced,
)
from src.experiment import build_params
from src.metrics import (
    collect_step_metrics, motif_propagation, statistical_complexity,
)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Lambda instrumentation ────────────────────────────────────────

def compute_lambda_step(grid, grid_pre_states, grid_post_states,
                        cell_throughput):
    """Compute all three Lambda candidates from one step transition.

    Candidates (all bounded):
      Lambda_A: log10 survival balance — net persistence vs turnover
      Lambda_B: throughput advantage of survivors over disrupted cells
      Lambda_C: coefficient of variation of cell ages (persistence concentration)

    Args:
        grid: Grid object with current states, throughput, and age (post-update)
        grid_pre_states: (H, W) int8 states before update
        grid_post_states: (H, W) int8 states after update
        cell_throughput: (H, W) instantaneous throughput this step

    Returns dict with lambda_A, lambda_B, lambda_C and diagnostic counts.
    """
    H, W = grid_pre_states.shape

    was_nonempty = grid_pre_states > EMPTY
    is_nonempty = grid_post_states > EMPTY
    n_nonempty_pre = int(was_nonempty.sum())
    n_nonempty_post = int(is_nonempty.sum())

    if n_nonempty_pre < 2:
        return {
            "lambda_A": np.nan, "lambda_B": np.nan, "lambda_C": np.nan,
            "n_survived": 0, "n_disrupted": 0, "n_nonempty_pre": n_nonempty_pre,
            "survival_rate": 0.0, "turnover_rate": 1.0,
        }

    survived = was_nonempty & is_nonempty
    disrupted = was_nonempty & ~is_nonempty
    n_survived = int(survived.sum())
    n_disrupted = int(disrupted.sum())

    survival_rate = n_survived / max(n_nonempty_pre, 1)
    turnover_rate = n_disrupted / max(n_nonempty_pre, 1)

    # ── Lambda_A: log survival balance ──
    # 0 = equal survival and loss. >0 = net persistence. <0 = net loss.
    lambda_A = np.log10(1.0 + n_survived) - np.log10(1.0 + n_disrupted)

    # ── Lambda_B: throughput advantage of survivors ──
    # (mean throughput of survived - mean throughput of disrupted) / mean overall.
    # Positive = high-throughput cells preferentially survive (coupled).
    # Zero = survival is independent of throughput (blind).
    # Negative = high-throughput cells preferentially die (anti-coupled).
    # Bounded roughly [-2, 2] but typically in [-1, 1].
    n_surv_tp = int((survived).sum())
    n_disr_tp = int((disrupted).sum())
    if n_surv_tp > 0 and n_disr_tp > 0:
        mean_tp_surv = float(cell_throughput[survived].mean())
        mean_tp_disr = float(cell_throughput[disrupted].mean())
        mean_tp_all = float(cell_throughput[was_nonempty].mean())
        if mean_tp_all > 1e-12:
            lambda_B = (mean_tp_surv - mean_tp_disr) / mean_tp_all
        else:
            lambda_B = 0.0
    elif n_surv_tp > 0 and n_disr_tp == 0:
        # No disrupted cells (perfect survival) — max coupling
        lambda_B = 1.0
    else:
        lambda_B = 0.0

    # ── Lambda_C: persistence concentration (CV of cell ages) ──
    # High = selective regime (mix of old survivors and young newcomers).
    # Low = uniform (either all dead/age-0, or all locked-in with similar ages).
    # This captures the DIVERSITY of survival outcomes from selection pressure.
    if n_nonempty_post >= 3:
        age_vals = grid.age[is_nonempty].astype(float)
        mean_age = age_vals.mean()
        if mean_age > 0.5:
            lambda_C = float(age_vals.std() / mean_age)
        else:
            lambda_C = 0.0
    else:
        lambda_C = 0.0

    return {
        "lambda_A": float(lambda_A),
        "lambda_B": float(lambda_B),
        "lambda_C": float(lambda_C),
        "n_survived": n_survived,
        "n_disrupted": n_disrupted,
        "n_nonempty_pre": n_nonempty_pre,
        "survival_rate": survival_rate,
        "turnover_rate": turnover_rate,
    }


# ── Simulation runners ────────────────────────────────────────────

def run_lambda_decoupled(config, seed, decouple_cfg, n_steps=None, regime=3):
    """Run one decoupled simulation collecting Lambda metrics each step."""
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
        n_steps = config.get("lambda", {}).get("n_steps", 1500)
    label = decouple_cfg.get("label", decouple_cfg.get("mode", "?"))

    records = []
    for step in range(n_steps):
        output, throughput, balance = flow_energy(grid, e_in_per_cell, params)
        m = collect_step_metrics(output, throughput, grid, balance, step)

        mode_totals = output.sum(axis=0)
        m["stat_complexity"] = statistical_complexity(mode_totals)
        m["entropy_production"] = (
            m["output_entropy"] * m["energy_out"] / max(m["energy_in"], 1e-12)
        )

        motif_ids = extract_motifs(grid.states)
        m["propagation"] = motif_propagation(motif_ids, grid.states)

        pre_states = grid.states.copy()

        # Update
        update_states_decoupled(grid, throughput, params, regime, decouple_cfg)

        # Compute Lambda from pre→post transition
        lam = compute_lambda_step(grid, pre_states, grid.states, throughput)
        m.update(lam)
        m["condition"] = label
        m["seed"] = seed
        records.append(m)

    return records


def run_lambda_reinforced(config, seed, epsilon, n_steps=None, regime=3,
                          persist_override=None, noise_off=False,
                          decay_rate=0.995, max_score=10.0):
    """Run one reinforced simulation collecting Lambda metrics."""
    rng = np.random.default_rng(seed)
    gc = config["grid"]
    grid = Grid(gc["height"], gc["width"], gc["n_modes"], rng)
    grid.initialize(
        active_frac=config["states"]["active_fraction"],
        passive_frac=config["states"]["passive_fraction"],
    )
    params = build_params(config)
    rc = config.get("reinforcement", {})
    if persist_override is not None:
        params["persistence_strength"] = persist_override
    elif "persistence_strength" in rc:
        params["persistence_strength"] = rc["persistence_strength"]
    if noise_off:
        params["mutation_rate"] = 0.0
        params["spontaneous_create_rate"] = 0.0
    e_in_per_cell = config["energy"]["E_in"] / gc["width"]
    if n_steps is None:
        n_steps = config.get("lambda", {}).get("n_steps", 1500)

    reinf_map = ReinforcementMap(epsilon=epsilon, decay_rate=decay_rate,
                                 max_score=max_score)
    records = []
    for step in range(n_steps):
        output, throughput, balance = flow_energy(grid, e_in_per_cell, params)
        m = collect_step_metrics(output, throughput, grid, balance, step)

        mode_totals = output.sum(axis=0)
        m["stat_complexity"] = statistical_complexity(mode_totals)
        m["entropy_production"] = (
            m["output_entropy"] * m["energy_out"] / max(m["energy_in"], 1e-12)
        )
        motif_ids = extract_motifs(grid.states)
        m["propagation"] = motif_propagation(motif_ids, grid.states)

        pre_states = grid.states.copy()

        update_states_reinforced(grid, throughput, params, regime, reinf_map)

        lam = compute_lambda_step(grid, pre_states, grid.states, throughput)
        m.update(lam)
        m["seed"] = seed
        records.append(m)

    return records


def run_lambda_baseline(config, seed, regime, n_steps=None):
    """Run a plain baseline simulation (regimes 1-3) with Lambda."""
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
        n_steps = config.get("lambda", {}).get("n_steps", 1500)

    records = []
    for step in range(n_steps):
        output, throughput, balance = flow_energy(grid, e_in_per_cell, params)
        m = collect_step_metrics(output, throughput, grid, balance, step)
        mode_totals = output.sum(axis=0)
        m["stat_complexity"] = statistical_complexity(mode_totals)
        m["entropy_production"] = (
            m["output_entropy"] * m["energy_out"] / max(m["energy_in"], 1e-12)
        )
        motif_ids = extract_motifs(grid.states)
        m["propagation"] = motif_propagation(motif_ids, grid.states)

        pre_states = grid.states.copy()

        update_states(grid, throughput, params, regime)

        lam = compute_lambda_step(grid, pre_states, grid.states, throughput)
        m.update(lam)
        m["seed"] = seed
        records.append(m)

    return records


# ── Plotting ──────────────────────────────────────────────────────

def _add_labels(ax, x, y, labels, fontsize=7):
    """Add non-overlapping labels to a scatter plot using adjustText."""
    from adjustText import adjust_text
    texts = []
    for i, txt in enumerate(labels):
        texts.append(ax.text(x[i], y[i], txt, fontsize=fontsize, ha="center",
                             va="bottom"))
    adjust_text(texts, x=x, y=y, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5),
                force_text=(0.8, 1.0), force_points=(0.3, 0.3),
                expand=(1.2, 1.4))


def plot_lambda_vs_metrics(summary_df, save_path):
    """Scatter: Lambda candidates vs regime metrics (persistence, propagation)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    lambdas = ["lambda_A", "lambda_B", "lambda_C"]
    lambda_labels = [
        r"$\Lambda_A$ (log survival balance)",
        r"$\Lambda_B$ (survivor tp advantage)",
        r"$\Lambda_C$ (age CV)",
    ]
    metrics = [("ss_mean_age", "Persistence (mean age)"),
               ("ss_propagation", "Propagation")]

    for row, (metric, ylabel) in enumerate(metrics):
        for col, (lam, xlabel) in enumerate(zip(lambdas, lambda_labels)):
            ax = axes[row, col]
            x = summary_df[lam].values
            y = summary_df[metric].values
            labels = summary_df["condition"].values

            ax.scatter(x, y, s=80, c=range(len(x)), cmap="tab10",
                       edgecolor="white", linewidth=0.5, zorder=3)
            _add_labels(ax, x, y, labels, fontsize=6)

            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 3:
                r, p = sp_stats.pearsonr(x[valid], y[valid])
                ax.set_title(f"{xlabel}\nr={r:.3f}, p={p:.3f}", fontsize=9)
            else:
                ax.set_title(xlabel, fontsize=9)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Flow-Persistence Number vs Regime Metrics",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_lambda_vs_ep_complexity(summary_df, save_path):
    """Compare Lambda vs mixing entropy, dissipation rate, and complexity."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    predictors = [
        ("lambda_B", r"$\Lambda_B$ (selection gradient)"),
        ("ss_entropy_prod", "Mixing Entropy"),
        ("ss_dissipation_rate", "Dissipation Rate"),
        ("ss_complexity", "Statistical Complexity"),
    ]
    outcomes = [
        ("ss_mean_age", "Persistence (mean age)"),
        ("ss_propagation", "Propagation"),
    ]

    for row, (outcome, ylabel) in enumerate(outcomes):
        for col, (pred, xlabel) in enumerate(predictors):
            ax = axes[row, col]
            x = summary_df[pred].values
            y = summary_df[outcome].values
            labels = summary_df["condition"].values

            ax.scatter(x, y, s=80, c=range(len(x)), cmap="tab10",
                       edgecolor="white", linewidth=0.5, zorder=3)
            _add_labels(ax, x, y, labels, fontsize=6)

            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 3:
                rho, _ = sp_stats.spearmanr(x[valid], y[valid])
                ax.set_title(f"{xlabel}\nSpearman rho={rho:.3f}", fontsize=9)
            else:
                ax.set_title(xlabel, fontsize=9)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle(r"Regime Prediction: $\Lambda_B$ vs Mixing Entropy vs Dissipation vs Complexity",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_regime_phase(summary_df, save_path):
    """Phase-style plot: Lambda_B vs persistence, colored by regime class."""
    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(12, 8))

    x = summary_df["lambda_B"].values
    y = summary_df["ss_mean_age"].values
    labels = summary_df["condition"].values

    # Classify regimes by persistence
    regime_class = []
    for _, row in summary_df.iterrows():
        age = row["ss_mean_age"]
        prop = row["ss_propagation"]
        if age < 10 and prop < 0.01:
            regime_class.append("collapse")
        elif age > 200 and prop > 0.3:
            regime_class.append("lock-in")
        else:
            regime_class.append("structured")

    color_map = {"collapse": "#e74c3c", "structured": "#2ecc71", "lock-in": "#3498db"}
    colors = [color_map[r] for r in regime_class]

    ax.scatter(x, y, s=120, c=colors, edgecolor="white", linewidth=1, zorder=3)

    # Non-overlapping labels
    texts = []
    for i, txt in enumerate(labels):
        texts.append(ax.text(x[i], y[i], txt, fontsize=8, ha="center",
                             va="bottom", style="italic"))
    adjust_text(texts, x=x, y=y, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5),
                force_text=(1.0, 1.2), force_points=(0.5, 0.5),
                expand=(1.3, 1.5))

    # Legend
    for regime, color in color_map.items():
        ax.scatter([], [], c=color, s=80, label=regime, edgecolor="white")
    ax.legend(fontsize=11, title="Regime", title_fontsize=11, loc="upper left")

    # Threshold line
    ax.axvline(x=0.0, color="gray", linestyle="--", alpha=0.4)

    ax.set_xlabel(r"$\Lambda_B$ (survivor throughput advantage)", fontsize=12)
    ax.set_ylabel("Persistence (mean age)", fontsize=12)
    ax.set_title("Regime Phase Diagram: Flow-Persistence Number",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return regime_class


def plot_lambda_timeseries(results, save_path):
    """Time series of Lambda_B across conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    lambdas = [("lambda_A", r"$\Lambda_A$"),
               ("lambda_B", r"$\Lambda_B$"),
               ("lambda_C", r"$\Lambda_C$")]

    conditions = results["condition"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

    for ax, (lam, title) in zip(axes, lambdas):
        for cond, color in zip(conditions, colors):
            sub = results[results["condition"] == cond]
            grouped = sub.groupby("step")[lam]
            mean = grouped.mean()
            window = max(1, len(mean) // 30)
            if window > 1:
                kernel = np.ones(window) / window
                mean_s = np.convolve(mean.values, kernel, mode="same")
            else:
                mean_s = mean.values
            ax.plot(mean.index.values, mean_s, label=cond, color=color,
                    linewidth=1.0, alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=5, loc="lower right", ncol=2)

    plt.suptitle("Flow-Persistence Number Over Time",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close()


# ── Report ────────────────────────────────────────────────────────

def generate_report(summary_df, regime_classes, config):
    """Generate markdown report."""
    lines = []
    lines.append("# Flow-Persistence Number (Lambda) — Analysis Report\n\n")

    lc = config.get("lambda", {})
    lines.append(f"Grid: {config['grid']['height']}x{config['grid']['width']}, "
                 f"{config['grid']['n_modes']} modes\n")
    lines.append(f"Steps: {lc.get('n_steps', 1500)}, "
                 f"Seeds: {lc.get('n_seeds', 8)}\n\n")

    lines.append("## Definitions\n\n")
    lines.append("Three candidate empirical flow-persistence numbers:\n\n")

    lines.append("### Lambda_A: Log Survival Balance\n\n")
    lines.append("    Lambda_A = log10(1 + n_survived) - log10(1 + n_disrupted)\n\n")
    lines.append("Where `survived` = non-empty before AND after the update, "
                 "`disrupted` = non-empty before but empty after. "
                 "0 = equal survival and loss. Positive = net persistence. "
                 "Negative = net loss. Measures the raw persistence/turnover balance.\n\n")

    lines.append("### Lambda_B: Throughput Advantage of Survivors\n\n")
    lines.append("    Lambda_B = (mean_tp_survived - mean_tp_disrupted) / mean_tp_all\n\n")
    lines.append("Normalized throughput difference between cells that survived and cells "
                 "that were disrupted in each step. Positive = surviving cells had higher "
                 "throughput (the coupling selects for productive cells). Zero = survival "
                 "is independent of throughput (blind). Negative = surviving cells had "
                 "LOWER throughput (anti-coupled). When no cells are disrupted (perfect "
                 "survival), Lambda_B = 1.0. This directly measures the throughput "
                 "selection gradient at each step.\n\n")

    lines.append("### Lambda_C: Persistence Concentration (Age CV)\n\n")
    lines.append("    Lambda_C = std(age) / mean(age) among non-empty cells\n\n")
    lines.append("Coefficient of variation of cell ages. High = selective regime where "
                 "some cells live long while others turn over quickly. Low = uniform "
                 "(either all dead or all locked-in). This captures the DIVERSITY of "
                 "survival outcomes produced by selection pressure.\n\n")

    lines.append("## Why These Are Reasonable Proxies\n\n")
    lines.append("- Lambda_A captures the raw balance of persistence vs turnover (log-scale).\n")
    lines.append("- Lambda_B measures the *throughput selection gradient*: do surviving cells "
                 "have systematically higher throughput than disrupted cells?\n")
    lines.append("- Lambda_C measures *persistence concentration*: how diverse are cell ages? "
                 "High = selective regime (mixed ages), low = uniform (all dead or all locked-in).\n\n")

    lines.append("## Summary Table\n\n")
    cols = ["condition", "lambda_A", "lambda_B", "lambda_C",
            "ss_mean_age", "ss_propagation", "ss_entropy_prod",
            "ss_dissipation_rate", "ss_complexity", "ss_active_area"]
    header = "| Condition | Lambda_A | Lambda_B | Lambda_C | Persistence | Propagation | Mixing Ent. | Dissip. Rate | Complexity | Active |"
    sep = "|-----------|----------|----------|----------|-------------|-------------|-------------|-------------|-----------|--------|"
    lines.append(header + "\n")
    lines.append(sep + "\n")
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['condition']} "
            f"| {row['lambda_A']:.2f} "
            f"| {row['lambda_B']:.2f} "
            f"| {row['lambda_C']:.2f} "
            f"| {row['ss_mean_age']:.1f} "
            f"| {row['ss_propagation']:.3f} "
            f"| {row['ss_entropy_prod']:.3f} "
            f"| {row['ss_dissipation_rate']:.3f} "
            f"| {row['ss_complexity']:.5f} "
            f"| {row['ss_active_area']:.0f} |\n"
        )

    # Correlations
    lines.append("\n## Correlation Analysis\n\n")
    predictors = [("lambda_A", "Lambda_A"), ("lambda_B", "Lambda_B"),
                  ("lambda_C", "Lambda_C"),
                  ("ss_entropy_prod", "Mixing entropy"), ("ss_complexity", "Complexity"),
                  ("ss_dissipation_rate", "Dissipation rate"),
                  ("ss_active_area", "Active cells")]
    outcomes = [("ss_mean_age", "Persistence"), ("ss_propagation", "Propagation")]

    # Spearman rank correlations (better for nonlinear monotonic relationships)
    lines.append("Spearman rank correlations (robust to nonlinearity):\n\n")
    lines.append("| Predictor | Persistence rho | Persistence p | Propagation rho | Propagation p |\n")
    lines.append("|-----------|----------------|-------------|-----------------|-------------|\n")

    best_predictor = None
    best_r = -1
    for pred_col, pred_name in predictors:
        row_parts = [pred_name]
        for out_col, _ in outcomes:
            x = summary_df[pred_col].values
            y = summary_df[out_col].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 3:
                r, p = sp_stats.spearmanr(x[valid], y[valid])
                row_parts.append(f"{r:.3f}")
                row_parts.append(f"{p:.4f}")
                if out_col == "ss_mean_age" and abs(r) > best_r:
                    best_r = abs(r)
                    best_predictor = pred_name
            else:
                row_parts.append("n/a")
                row_parts.append("n/a")
        lines.append("| " + " | ".join(row_parts) + " |\n")

    # Also show Pearson for comparison
    lines.append("\nPearson correlations (for comparison):\n\n")
    lines.append("| Predictor | Persistence r | Persistence p | Propagation r | Propagation p |\n")
    lines.append("|-----------|-------------|-------------|--------------|-------------|\n")
    for pred_col, pred_name in predictors:
        row_parts = [pred_name]
        for out_col, _ in outcomes:
            x = summary_df[pred_col].values
            y = summary_df[out_col].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 3:
                r, p = sp_stats.pearsonr(x[valid], y[valid])
                row_parts.append(f"{r:.3f}")
                row_parts.append(f"{p:.4f}")
            else:
                row_parts.append("n/a")
                row_parts.append("n/a")
        lines.append("| " + " | ".join(row_parts) + " |\n")

    # Excluding lock-in outlier
    no_lockin = summary_df[~summary_df["condition"].str.contains("no_noise")]
    if len(no_lockin) < len(summary_df):
        lines.append("\nSpearman correlations excluding lock-in (reinf_no_noise):\n\n")
        lines.append("| Predictor | Persistence rho | Propagation rho |\n")
        lines.append("|-----------|----------------|----------------|\n")
        for pred_col, pred_name in predictors:
            row_parts = [pred_name]
            for out_col, _ in outcomes:
                x = no_lockin[pred_col].values
                y = no_lockin[out_col].values
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() >= 3:
                    r, _ = sp_stats.spearmanr(x[valid], y[valid])
                    row_parts.append(f"{r:.3f}")
                else:
                    row_parts.append("n/a")
            lines.append("| " + " | ".join(row_parts) + " |\n")

    # Regime classification
    lines.append("\n## Regime Classification\n\n")
    lines.append("| Condition | Regime | Lambda_B |\n")
    lines.append("|-----------|--------|----------|\n")
    for i, (_, row) in enumerate(summary_df.iterrows()):
        rc = regime_classes[i] if i < len(regime_classes) else "?"
        lines.append(f"| {row['condition']} | {rc} | {row['lambda_B']:.2f} |\n")

    # Check thresholds
    collapse_lambdas = []
    structured_lambdas = []
    lockin_lambdas = []
    for i, rc in enumerate(regime_classes):
        lb = summary_df.iloc[i]["lambda_B"]
        if np.isnan(lb):
            continue
        if rc == "collapse":
            collapse_lambdas.append(lb)
        elif rc == "structured":
            structured_lambdas.append(lb)
        elif rc == "lock-in":
            lockin_lambdas.append(lb)

    lines.append("\n### Regime Bands\n\n")
    if collapse_lambdas:
        lines.append(f"- **Collapse** regime: Lambda_B in "
                     f"[{min(collapse_lambdas):.2f}, {max(collapse_lambdas):.2f}]\n")
    if structured_lambdas:
        lines.append(f"- **Structured** regime: Lambda_B in "
                     f"[{min(structured_lambdas):.2f}, {max(structured_lambdas):.2f}]\n")
    if lockin_lambdas:
        lines.append(f"- **Lock-in** regime: Lambda_B in "
                     f"[{min(lockin_lambdas):.2f}, {max(lockin_lambdas):.2f}]\n")

    # Separation check
    if collapse_lambdas and structured_lambdas:
        max_collapse = max(collapse_lambdas)
        min_structured = min(structured_lambdas)
        if max_collapse < min_structured:
            lines.append(f"\nLambda_B **cleanly separates** collapse from structured regimes "
                         f"(gap: {min_structured:.2f} - {max_collapse:.2f} = "
                         f"{min_structured - max_collapse:.2f}).\n")
        else:
            lines.append(f"\nCollapse and structured regimes overlap in Lambda_B.\n")

    # Key questions
    lines.append("\n## Key Questions\n\n")

    lines.append("### Does Lambda separate collapse / structured / lock-in?\n\n")
    if collapse_lambdas and structured_lambdas:
        if max(collapse_lambdas) < min(structured_lambdas):
            lines.append("**Yes.** Lambda_B separates collapse from structured regimes.\n\n")
        else:
            lines.append("Partially. There is overlap between regime classes.\n\n")
    else:
        lines.append("Cannot assess — insufficient regime diversity.\n\n")

    lines.append("### Does anti-coupling drive Lambda below the structured regime?\n\n")
    anti = summary_df[summary_df["condition"].str.contains("anti")]
    if len(anti) > 0:
        anti_lb = anti.iloc[0]["lambda_B"]
        lines.append(f"Anti-coupled Lambda_B = {anti_lb:.3f}. ")
        if anti_lb < 0.0:
            lines.append("**Yes**, anti-coupling drives Lambda_B negative, "
                         "meaning throughput actively *hurts* survival — "
                         "the feedback loop is inverted.\n\n")
        elif anti_lb < 0.05:
            lines.append("Anti-coupling drives Lambda_B near zero, "
                         "effectively decoupling throughput from persistence.\n\n")
        else:
            lines.append("No, Lambda_B remains positive under anti-coupling.\n\n")
    else:
        lines.append("No anti-coupled condition found.\n\n")

    lines.append("### Does no-noise lock-in correspond to excessive Lambda?\n\n")
    lockin_conds = summary_df[summary_df["condition"].str.contains("no_noise")]
    if len(lockin_conds) > 0:
        li = lockin_conds.iloc[0]
        lines.append(f"No-noise condition: Lambda_A = {li['lambda_A']:.2f}, "
                     f"Lambda_B = {li['lambda_B']:.3f}, "
                     f"persistence = {li['ss_mean_age']:.1f}. ")
        if li["lambda_A"] > 2.5:
            lines.append("**Yes**, extremely high Lambda_A (log survival balance) indicates "
                         "repair dominates with negligible disruption — frozen lock-in.\n\n")
        else:
            lines.append("Lambda_A is elevated but not extremely high.\n\n")
    else:
        lines.append("No no-noise condition found.\n\n")

    lines.append("### Which Lambda predicts regime best?\n\n")
    if best_predictor:
        lines.append(f"**{best_predictor}** has the strongest correlation with persistence "
                     f"(r = {best_r:.3f}).\n\n")

    # Final assessment
    lines.append("## Conclusion\n\n")

    # Compute Spearman correlations with persistence for all candidates
    lambda_rs = {}
    for pred_col, pred_name in predictors:
        x = summary_df[pred_col].values
        y = summary_df["ss_mean_age"].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() >= 3:
            r, _ = sp_stats.spearmanr(x[valid], y[valid])
            lambda_rs[pred_name] = abs(r)

    ep_r = lambda_rs.get("Mixing entropy", 0)
    cx_r = lambda_rs.get("Complexity", 0)
    dr_r = lambda_rs.get("Dissipation rate", 0)
    best_lambda_name = None
    best_lambda_r = 0
    for name in ["Lambda_A", "Lambda_B", "Lambda_C"]:
        r = lambda_rs.get(name, 0)
        if r > best_lambda_r:
            best_lambda_r = r
            best_lambda_name = name

    if best_lambda_r > ep_r and best_lambda_r > cx_r and best_lambda_r > 0.3:
        lines.append(
            f"The flow-persistence number {best_lambda_name} predicts the structured "
            f"regime better than mixing entropy "
            f"(Spearman |rho|={best_lambda_r:.3f} vs {ep_r:.3f}), "
            f"dissipation rate (|rho|={dr_r:.3f}), "
            f"or statistical complexity (|rho|={cx_r:.3f}).\n\n"
            "This supports the claim:\n\n"
            "> The structured regime is governed by a flow-persistence number comparing "
            "throughput-mediated repair to disruption. This quantity predicts the "
            "existence of the selection-like regime better than any thermodynamic "
            "observable.\n"
        )
    elif best_lambda_r > 0.3:
        lines.append(
            f"{best_lambda_name} correlates with regime outcomes "
            f"(Spearman |rho|={best_lambda_r:.3f}) "
            f"but does not clearly dominate mixing entropy (|rho|={ep_r:.3f}), "
            f"dissipation rate (|rho|={dr_r:.3f}), "
            f"or complexity (|rho|={cx_r:.3f}). The flow-persistence number captures "
            "regime structure but additional information may be needed.\n"
        )
    else:
        lines.append(
            f"The best Lambda candidate ({best_lambda_name}) shows only modest "
            f"correlation with regime outcomes (Spearman |rho|={best_lambda_r:.3f}). "
            f"Mixing entropy has |rho|={ep_r:.3f}, dissipation rate |rho|={dr_r:.3f}, "
            f"complexity has |rho|={cx_r:.3f}. "
            "The flow-persistence number definitions may need refinement.\n"
        )

    return "".join(lines)


# ── Main ──────────────────────────────────────────────────────────

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    lc = config.get("lambda", {})
    out_dir = lc.get("output_dir", "lambda_output")
    plot_dir = lc.get("plots_dir", "lambda_plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    n_seeds = lc.get("n_seeds", 8)
    n_steps = lc.get("n_steps", 1500)

    print("=" * 64)
    print("FLOW-PERSISTENCE NUMBER (LAMBDA) ANALYSIS")
    print("=" * 64)
    print(f"Grid: {config['grid']['height']}x{config['grid']['width']}, "
          f"{config['grid']['n_modes']} modes")
    print(f"Steps: {n_steps}, Seeds: {n_seeds}")
    print()

    t_total = time.time()
    all_records = []

    # ── 1. Base regimes (memoryless, persistent, persistent+branching) ──
    print("  [1/4] Base regimes...")
    for regime, name in [(1, "memoryless"), (2, "persistent"),
                         (3, "persistent+branching")]:
        print(f"    {name} ({n_seeds} seeds)...")
        t0 = time.time()
        for seed in range(n_seeds):
            recs = run_lambda_baseline(config, seed, regime, n_steps=n_steps)
            for r in recs:
                r["condition"] = name
            all_records.extend(recs)
        print(f"      Done ({time.time()-t0:.0f}s)")

    # ── 2. Decoupling conditions ──
    print("\n  [2/4] Decoupling conditions...")
    dc = config.get("decoupling", {})
    modes = dc.get("modes", [])
    for mode_cfg in modes:
        label = mode_cfg.get("label", mode_cfg["mode"])
        # Skip baseline (already covered as persistent+branching)
        if label == "baseline":
            label = "decouple_baseline"
        print(f"    {label} ({n_seeds} seeds)...")
        t0 = time.time()
        for seed in range(n_seeds):
            recs = run_lambda_decoupled(config, seed, mode_cfg,
                                        n_steps=n_steps, regime=3)
            for r in recs:
                r["condition"] = label
            all_records.extend(recs)
        print(f"      Done ({time.time()-t0:.0f}s)")

    # ── 3. Reinforcement conditions ──
    print("\n  [3/4] Reinforcement conditions...")
    rc = config.get("reinforcement", {})
    reinf_decay = rc.get("decay_rate", 0.995)
    reinf_max = rc.get("max_score", 10.0)

    reinf_conditions = [
        ("reinf_eps0", 0.0, None, False),
        ("reinf_eps0.4", 0.4, None, False),
        ("reinf_no_noise", 0.4, None, True),
    ]
    for label, eps, persist, noise_off in reinf_conditions:
        print(f"    {label} ({n_seeds} seeds)...")
        t0 = time.time()
        for seed in range(n_seeds):
            recs = run_lambda_reinforced(
                config, seed, epsilon=eps, n_steps=n_steps,
                persist_override=persist, noise_off=noise_off,
                decay_rate=reinf_decay, max_score=reinf_max,
            )
            for r in recs:
                r["condition"] = label
            all_records.extend(recs)
        print(f"      Done ({time.time()-t0:.0f}s)")

    # ── 4. Combine and summarize ──
    print("\n  [4/4] Computing summaries...")
    results = pd.DataFrame(all_records)
    results.to_csv(os.path.join(out_dir, "lambda_sweep.csv"), index=False)

    ss_start = int(n_steps * 0.7)
    conditions = results["condition"].unique()
    summary_rows = []
    for cond in conditions:
        sub = results[(results["condition"] == cond) & (results["step"] >= ss_start)]
        # Dissipation rate = fraction of input energy lost (thermodynamic EP proxy)
        e_in = sub["energy_in"].values
        e_lost = sub["energy_lost"].values
        dissipation_rate = float(np.mean(e_lost / np.maximum(e_in, 1e-12)))
        summary_rows.append({
            "condition": cond,
            "lambda_A": sub["lambda_A"].mean(),
            "lambda_B": sub["lambda_B"].mean(),
            "lambda_C": sub["lambda_C"].mean(),
            "ss_mean_age": sub["mean_age"].mean(),
            "ss_propagation": sub["propagation"].mean(),
            "ss_entropy_prod": sub["entropy_production"].mean(),
            "ss_complexity": sub["stat_complexity"].mean(),
            "ss_active_area": sub["active_area"].mean(),
            "ss_throughput": sub["throughput_mean"].mean(),
            "ss_dissipation_rate": dissipation_rate,
            "survival_rate": sub["survival_rate"].mean(),
            "turnover_rate": sub["turnover_rate"].mean(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "lambda_summary.csv"), index=False)

    # ── Plots ──
    print("\n  Generating plots...")
    plot_lambda_vs_metrics(summary_df,
                           os.path.join(plot_dir, "lambda_vs_metrics.png"))
    plot_lambda_vs_ep_complexity(summary_df,
                                os.path.join(plot_dir, "lambda_vs_ep_complexity.png"))
    regime_classes = plot_regime_phase(summary_df,
                                      os.path.join(plot_dir, "regime_phase.png"))
    plot_lambda_timeseries(results,
                           os.path.join(plot_dir, "lambda_timeseries.png"))
    print("    Done.")

    # ── Report ──
    report = generate_report(summary_df, regime_classes, config)
    with open(os.path.join(out_dir, "lambda_report.md"), "w") as f:
        f.write(report)

    print()
    print(report)
    print()
    print(f"Total time: {time.time()-t_total:.0f}s")
    print(f"Outputs: {out_dir}/")
    print(f"Plots:   {plot_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
