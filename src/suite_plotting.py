"""
Plotting for the experiment suite.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .grid import TRAIT_NAMES, TRAIT_DEFAULTS, TRAIT_ALPHA, TRAIT_SPLIT, TRAIT_THRESHOLD

COLORS = {"best_frozen": "#e74c3c", "default_frozen": "#3498db", "evolving": "#2ecc71"}


def _cum_ep_per_seed(df):
    return df.groupby("seed")["entropy_production"].sum()


def _ep_timeseries(df, window=50):
    """Rolling mean EP per step, averaged across seeds."""
    pivot = df.pivot_table(index="step", columns="seed", values="entropy_production")
    rolled = pivot.rolling(window, min_periods=1).mean()
    return rolled.mean(axis=1), rolled.std(axis=1)


# ── Experiment 1 Plots ──────────────────────────────────────────

def plot_frozen_landscape(search_df, path):
    """Heatmap of grid search results: alpha vs split for best threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    best_thresh = search_df.loc[search_df["cum_ep_mean"].idxmax(), "threshold"]
    sub = search_df[search_df["threshold"] == best_thresh]
    piv = sub.pivot_table(index="split", columns="alpha", values="cum_ep_mean")
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd",
                   extent=[piv.columns.min(), piv.columns.max(),
                           piv.index.min(), piv.index.max()],
                   origin="lower")
    ax.set_xlabel("transform_strength (alpha)")
    ax.set_ylabel("split_factor")
    ax.set_title(f"Frozen Trait Search (threshold={best_thresh})")
    plt.colorbar(im, ax=ax, label="Cumulative EP")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_exp1_comparison(exp1, path):
    """Bar chart: default frozen vs best frozen vs evolving."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of cumulative EP
    regimes = ["default_frozen", "best_frozen", "evolving"]
    labels = ["Default Frozen", "Best Frozen", "Evolving"]
    means, stds = [], []
    for r in regimes:
        if r in exp1:
            c = _cum_ep_per_seed(exp1[r])
            means.append(c.mean())
            stds.append(c.std())
        else:
            means.append(0)
            stds.append(0)

    x = np.arange(len(regimes))
    colors = [COLORS[r] for r in regimes]
    ax1.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Cumulative EP")
    ax1.set_title("Exp 1: Cumulative Entropy Production")
    ax1.grid(True, alpha=0.3, axis="y")

    # Time series
    for r, label in zip(regimes, labels):
        if r in exp1:
            m, s = _ep_timeseries(exp1[r])
            ax2.plot(m.index, m.values, color=COLORS[r], label=label, linewidth=1.5)
            ax2.fill_between(m.index, m - s, m + s, color=COLORS[r], alpha=0.1)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("EP (rolling mean)")
    ax2.set_title("Per-Step Entropy Production")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Experiment 1: Evolving vs Best Frozen", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Experiment 2 Plots ──────────────────────────────────────────

def plot_exp2_environments(exp2, path):
    """Bar chart: frozen vs evolving per heterogeneous environment."""
    env_names = list(exp2.keys())
    n_envs = len(env_names)

    fig, axes = plt.subplots(1, n_envs + 1, figsize=(5 * (n_envs + 1), 5))
    if n_envs + 1 == 1:
        axes = [axes]

    # Per-environment bars
    for i, env_name in enumerate(env_names):
        ax = axes[i]
        data = exp2[env_name]
        regimes = ["best_frozen", "evolving"]
        labels = ["Best Frozen", "Evolving"]
        means, stds = [], []
        for r in regimes:
            if r in data:
                c = _cum_ep_per_seed(data[r])
                means.append(c.mean())
                stds.append(c.std())
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(len(regimes))
        ax.bar(x, means, yerr=stds,
               color=[COLORS[r] for r in regimes], alpha=0.8, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Cumulative EP")
        ax.set_title(f"{env_name}")
        ax.grid(True, alpha=0.3, axis="y")

    # Summary: advantage by environment
    ax = axes[-1]
    advantages = []
    for env_name in env_names:
        data = exp2[env_name]
        if "best_frozen" in data and "evolving" in data:
            bf = _cum_ep_per_seed(data["best_frozen"]).mean()
            ev = _cum_ep_per_seed(data["evolving"]).mean()
            advantages.append((ev - bf) / max(bf, 1) * 100)
        else:
            advantages.append(0)
    colors = ["#2ecc71" if a > 0 else "#e74c3c" for a in advantages]
    ax.barh(range(len(env_names)), advantages, color=colors, alpha=0.8)
    ax.set_yticks(range(len(env_names)))
    ax.set_yticklabels(env_names)
    ax.set_xlabel("Evolving advantage (%)")
    ax.set_title("Relative Advantage")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Experiment 2: Heterogeneous Environments", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_exp2_traits(exp2, path):
    """Trait evolution by environment."""
    env_names = list(exp2.keys())
    trait_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(len(env_names), len(TRAIT_NAMES),
                             figsize=(4 * len(TRAIT_NAMES), 4 * len(env_names)))
    if len(env_names) == 1:
        axes = axes[np.newaxis, :]

    for row, env_name in enumerate(env_names):
        data = exp2[env_name]
        if "evolving" not in data:
            continue
        df = data["evolving"]
        for col, tname in enumerate(TRAIT_NAMES):
            ax = axes[row, col]
            mcol = f"trait_{tname}_mean"
            if mcol not in df.columns:
                continue
            pivot = df.pivot_table(index="step", columns="seed", values=mcol)
            m = pivot.mean(axis=1)
            s = pivot.std(axis=1)
            ax.plot(m.index, m.values, color=trait_colors[col], linewidth=1.5)
            ax.fill_between(m.index, m - s, m + s, color=trait_colors[col], alpha=0.15)
            ax.axhline(TRAIT_DEFAULTS[col], color="gray", linestyle=":", alpha=0.5)
            if row == 0:
                ax.set_title(tname, fontsize=9)
            if col == 0:
                ax.set_ylabel(env_name)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Experiment 2: Trait Evolution by Environment", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Experiment 3 Plots ──────────────────────────────────────────

def plot_exp3_timeseries(exp3, path):
    """Time series for changing environments with env condition overlay."""
    env_names = list(exp3.keys())
    fig, axes = plt.subplots(len(env_names), 2,
                             figsize=(14, 4 * len(env_names)))
    if len(env_names) == 1:
        axes = axes[np.newaxis, :]

    for row, env_name in enumerate(env_names):
        data = exp3[env_name]
        env = data.get("env")

        # Left: EP time series
        ax = axes[row, 0]
        for regime in ["best_frozen", "evolving"]:
            if regime in data:
                m, s = _ep_timeseries(data[regime])
                ax.plot(m.index, m.values, color=COLORS[regime],
                        label=regime.replace("_", " ").title(), linewidth=1.5)
                ax.fill_between(m.index, m - s, m + s, color=COLORS[regime], alpha=0.1)
        ax.set_xlabel("Step")
        ax.set_ylabel("EP")
        ax.set_title(f"{env_name}: Entropy Production")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: Trait evolution (alpha)
        ax2 = axes[row, 1]
        if "evolving" in data:
            df = data["evolving"]
            col = "trait_transform_strength_mean"
            if col in df.columns:
                pivot = df.pivot_table(index="step", columns="seed", values=col)
                m = pivot.mean(axis=1)
                s = pivot.std(axis=1)
                ax2.plot(m.index, m.values, color="#2ecc71", linewidth=1.5, label="Evolved alpha")
                ax2.fill_between(m.index, m - s, m + s, color="#2ecc71", alpha=0.15)
                ax2.axhline(TRAIT_DEFAULTS[0], color="gray", linestyle=":", alpha=0.5,
                            label=f"Default ({TRAIT_DEFAULTS[0]})")

        # Overlay environment condition on secondary axis
        if env is not None:
            ax3 = ax2.twinx()
            steps = range(int(df["step"].max()) + 1) if "evolving" in data else range(1500)
            tc = [env.get_transform_cost(s) for s in steps]
            if any(np.ndim(t) > 0 for t in tc):
                tc = [np.mean(t) for t in tc]
            ax3.plot(steps, tc, color="#f39c12", alpha=0.4, linewidth=1, label="Transform cost")
            ax3.set_ylabel("Transform cost", color="#f39c12")

        ax2.set_xlabel("Step")
        ax2.set_ylabel("Alpha")
        ax2.set_title(f"{env_name}: Trait Adaptation")
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(True, alpha=0.3)

    plt.suptitle("Experiment 3: Changing Environments", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Experiment 4 Plots ──────────────────────────────────────────

def plot_exp4_ablation(exp4, path):
    """Bar chart of ablation results."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(exp4.keys())
    means, stds = [], []
    for name in names:
        c = _cum_ep_per_seed(exp4[name])
        means.append(c.mean())
        stds.append(c.std())

    x = np.arange(len(names))
    colors = []
    for n in names:
        if n == "full_evolving":
            colors.append("#2ecc71")
        else:
            colors.append("#95a5a6")

    ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Cumulative EP")
    ax.set_title("Experiment 4: Ablation Tests")
    ax.grid(True, alpha=0.3, axis="y")

    # Add percentage labels
    if means[0] > 0:
        for i in range(1, len(means)):
            pct = (means[i] - means[0]) / means[0] * 100
            ax.text(i, means[i] + stds[i] + means[0] * 0.01,
                    f"{pct:+.1f}%", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Experiment 5 Plots ──────────────────────────────────────────

def plot_exp5_longrun(exp5, path):
    """Long-run time series and trait evolution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ev = exp5["evolving_df"]
    fx = exp5["fixed_df"]
    window = 200

    # Top-left: Cumulative EP
    ax = axes[0, 0]
    for label, df, color in [("Fixed", fx, "#3498db"), ("Evolving", ev, "#2ecc71")]:
        pivot = df.pivot_table(index="step", columns="seed", values="entropy_production")
        cum = pivot.cumsum()
        m = cum.mean(axis=1)
        s = cum.std(axis=1)
        ax.plot(m.index, m.values, color=color, label=label, linewidth=1.5)
        ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative EP")
    ax.set_title("Cumulative Entropy Production")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Rolling EP (moving window)
    ax = axes[0, 1]
    for label, df, color in [("Fixed", fx, "#3498db"), ("Evolving", ev, "#2ecc71")]:
        m, s = _ep_timeseries(df, window=window)
        ax.plot(m.index, m.values, color=color, label=label, linewidth=1.5)
        ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.1)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"EP (rolling {window})")
    ax.set_title("Rolling Mean EP")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Trait evolution
    ax = axes[1, 0]
    trait_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for i, tname in enumerate(TRAIT_NAMES):
        col = f"trait_{tname}_mean"
        if col not in ev.columns:
            continue
        pivot = ev.pivot_table(index="step", columns="seed", values=col)
        m = pivot.mean(axis=1)
        ax.plot(m.index, m.values, color=trait_colors[i], linewidth=1.5, label=tname)
        ax.axhline(TRAIT_DEFAULTS[i], color=trait_colors[i], linestyle=":", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Trait value")
    ax.set_title("Trait Evolution (Long Run)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Lineage diversity
    ax = axes[1, 1]
    if "n_lineages" in ev.columns:
        pivot = ev.pivot_table(index="step", columns="seed", values="n_lineages")
        m = pivot.mean(axis=1)
        s = pivot.std(axis=1)
        ax.plot(m.index, m.values, color="#9b59b6", linewidth=1.5)
        ax.fill_between(m.index, m - s, m + s, color="#9b59b6", alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distinct lineages")
    ax.set_title("Lineage Diversity")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Experiment 5: Long-Run Behavior", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Master plot ─────────────────────────────────────────────────

def plot_summary_figure(exp1, exp2, exp3, path):
    """Single summary figure for paper: advantage across all conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = []
    advantages = []
    errors = []

    # Exp 1: homogeneous
    for regime in ["best_frozen"]:
        if regime in exp1:
            bf_c = _cum_ep_per_seed(exp1[regime])
            ev_c = _cum_ep_per_seed(exp1["evolving"])
            diff = ev_c.mean() - bf_c.mean()
            pct = diff / max(bf_c.mean(), 1) * 100
            se = np.sqrt(ev_c.var() + bf_c.var()) / max(bf_c.mean(), 1) * 100
            conditions.append("Homogeneous")
            advantages.append(pct)
            errors.append(se)

    # Exp 2: heterogeneous
    for env_name, data in exp2.items():
        if "best_frozen" in data and "evolving" in data:
            bf_c = _cum_ep_per_seed(data["best_frozen"])
            ev_c = _cum_ep_per_seed(data["evolving"])
            pct = (ev_c.mean() - bf_c.mean()) / max(bf_c.mean(), 1) * 100
            se = np.sqrt(ev_c.var() + bf_c.var()) / max(bf_c.mean(), 1) * 100
            conditions.append(f"Hetero: {env_name}")
            advantages.append(pct)
            errors.append(se)

    # Exp 3: changing
    for env_name, data in exp3.items():
        if "best_frozen" in data and "evolving" in data:
            bf_c = _cum_ep_per_seed(data["best_frozen"])
            ev_c = _cum_ep_per_seed(data["evolving"])
            pct = (ev_c.mean() - bf_c.mean()) / max(bf_c.mean(), 1) * 100
            se = np.sqrt(ev_c.var() + bf_c.var()) / max(bf_c.mean(), 1) * 100
            conditions.append(f"Change: {env_name}")
            advantages.append(pct)
            errors.append(se)

    y = np.arange(len(conditions))
    colors = ["#2ecc71" if a > 0 else "#e74c3c" for a in advantages]
    ax.barh(y, advantages, xerr=errors, color=colors, alpha=0.8, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(conditions)
    ax.set_xlabel("Evolving advantage over best frozen (%)")
    ax.set_title("Evolution vs Best Static System Across Conditions")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
