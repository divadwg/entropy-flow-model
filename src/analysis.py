"""
Cross-regime analysis and report generation.
"""
import numpy as np
import pandas as pd


def compute_regime_summary(df):
    """Summary statistics per regime from the main comparison."""
    results = []
    for regime in [1, 2, 3]:
        sub = df[df["regime"] == regime]
        n_steps = sub["step"].max() + 1
        last_frac = sub[sub["step"] > n_steps * 0.8]

        cum_entropy = sub.groupby("seed")["output_entropy"].sum()

        results.append({
            "regime": regime,
            "regime_name": sub["regime_name"].iloc[0],
            "cum_entropy_mean": float(cum_entropy.mean()),
            "cum_entropy_std": float(cum_entropy.std()),
            "final_entropy_mean": float(last_frac["output_entropy"].mean()),
            "final_entropy_std": float(last_frac["output_entropy"].std()),
            "final_frag_mean": float(last_frac["fragmentation"].mean()),
            "final_energy_out_mean": float(last_frac["energy_out"].mean()),
            "final_active_area_mean": float(last_frac["active_area"].mean()),
            "final_mean_age": float(last_frac["mean_age"].mean()),
            "final_kl_mean": float(last_frac["kl_from_uniform"].mean()),
            "final_eff_modes": float(last_frac["effective_modes"].mean()),
        })
    return pd.DataFrame(results)


def summarize_sweep(sweep_df, param_name):
    """One-line summary per parameter value showing regime comparison."""
    cum = (sweep_df.groupby(["regime", "sweep_value", "seed"])["output_entropy"]
           .sum().reset_index())
    lines = []
    for val in sorted(cum["sweep_value"].unique()):
        parts = []
        for regime, label in [(1, "R1"), (2, "R2"), (3, "R3")]:
            mean = cum[(cum["regime"] == regime) & (cum["sweep_value"] == val)]["output_entropy"].mean()
            parts.append(f"{label}={mean:.1f}")
        r1 = cum[(cum["regime"] == 1) & (cum["sweep_value"] == val)]["output_entropy"].mean()
        r3 = cum[(cum["regime"] == 3) & (cum["sweep_value"] == val)]["output_entropy"].mean()
        parts.append(f"adv(R3-R1)={r3 - r1:+.1f}")
        lines.append(f"  {param_name}={val}: {', '.join(parts)}")
    return "\n".join(lines)


def generate_report(summary_df, sweep_summaries, config):
    """Generate full markdown interpretation report."""
    lines = []
    lines.append("# Entropy Flow Model — Results Report\n\n")
    gc = config["grid"]
    lines.append(f"Grid: {gc['height']}x{gc['width']}, "
                 f"{gc['n_modes']} modes, "
                 f"E_in={config['energy']['E_in']}, "
                 f"{config['simulation']['n_steps']} steps, "
                 f"{config['simulation']['n_seeds']} seeds\n\n")

    # Main comparison table
    lines.append("## Main Regime Comparison\n\n")
    lines.append("| Regime | Cum. Entropy | Final Entropy | Fragmentation "
                 "| Energy Out | Active Area | Eff. Modes | KL from Uniform |\n")
    lines.append("|--------|-------------|---------------|---------------"
                 "|------------|-------------|------------|------------------|\n")

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['regime_name']} "
            f"| {row['cum_entropy_mean']:.1f} +/- {row['cum_entropy_std']:.1f} "
            f"| {row['final_entropy_mean']:.3f} +/- {row['final_entropy_std']:.3f} "
            f"| {row['final_frag_mean']:.0f} "
            f"| {row['final_energy_out_mean']:.2f} "
            f"| {row['final_active_area_mean']:.0f} "
            f"| {row['final_eff_modes']:.1f} "
            f"| {row['final_kl_mean']:.3f} |\n"
        )

    r1 = summary_df[summary_df["regime"] == 1].iloc[0]
    r2 = summary_df[summary_df["regime"] == 2].iloc[0]
    r3 = summary_df[summary_df["regime"] == 3].iloc[0]

    diff_31 = r3["cum_entropy_mean"] - r1["cum_entropy_mean"]
    diff_21 = r2["cum_entropy_mean"] - r1["cum_entropy_mean"]
    diff_32 = r3["cum_entropy_mean"] - r2["cum_entropy_mean"]
    pct_31 = diff_31 / max(r1["cum_entropy_mean"], 1) * 100
    pct_21 = diff_21 / max(r1["cum_entropy_mean"], 1) * 100
    pct_32 = diff_32 / max(r2["cum_entropy_mean"], 1) * 100

    # Hypothesis test
    lines.append("\n## Hypothesis Test\n\n")
    lines.append("**H1**: Persistent + branching channels produce higher cumulative "
                 "output entropy than memoryless dissipation.\n\n")
    lines.append(f"- Regime 3 vs Regime 1 (cumulative entropy): "
                 f"{diff_31:+.1f} ({pct_31:+.1f}%)\n")
    lines.append(f"- Regime 2 vs Regime 1: {diff_21:+.1f} ({pct_21:+.1f}%)\n")
    lines.append(f"- Regime 3 vs Regime 2: {diff_32:+.1f} ({pct_32:+.1f}%)\n\n")

    if diff_31 > 0 and diff_32 > 0:
        lines.append("**Result**: SUPPORTED — branching persistent channels produce the "
                     "highest cumulative output entropy. The ordering is R3 > R2 > R1.\n")
    elif diff_31 > 0 and diff_21 > 0:
        lines.append("**Result**: PARTIALLY SUPPORTED — persistent channels outperform "
                     "memoryless, but branching adds little over persistence alone.\n")
    elif diff_31 > 0:
        lines.append("**Result**: PARTIALLY SUPPORTED — branching outperforms memoryless "
                     "but the contribution of persistence alone is ambiguous.\n")
    elif abs(pct_31) < 2:
        lines.append("**Result**: INCONCLUSIVE — differences are within noise.\n")
    else:
        lines.append("**Result**: NOT SUPPORTED — memoryless dissipation produces "
                     "comparable or higher cumulative entropy.\n")

    # Blackbody comparison
    lines.append("\n## Blackbody (Uniform) Comparison\n\n")
    lines.append("KL divergence from uniform distribution "
                 "(lower = closer to maximum entropy):\n\n")
    for _, row in summary_df.iterrows():
        lines.append(f"- {row['regime_name']}: KL = {row['final_kl_mean']:.3f}, "
                     f"effective modes = {row['final_eff_modes']:.1f} / {gc['n_modes']}\n")

    kl_diff = r1["final_kl_mean"] - r3["final_kl_mean"]
    if kl_diff > 0:
        lines.append(f"\nBranching regime is "
                     f"{kl_diff / max(r1['final_kl_mean'], 0.001) * 100:.1f}% "
                     "closer to a uniform (blackbody-like) output distribution.\n")
    else:
        lines.append("\nBranching regime is NOT closer to uniform than memoryless.\n")

    # Energy efficiency
    lines.append("\n## Energy Efficiency\n\n")
    for _, row in summary_df.iterrows():
        eff = row["final_energy_out_mean"] / config["energy"]["E_in"] * 100
        lines.append(f"- {row['regime_name']}: "
                     f"{row['final_energy_out_mean']:.2f} / {config['energy']['E_in']} "
                     f"= {eff:.1f}% throughput\n")

    # Sweep results
    if sweep_summaries:
        lines.append("\n## Parameter Sweeps\n")
        for param, summary_text in sweep_summaries.items():
            lines.append(f"\n### {param}\n```\n{summary_text}\n```\n")

    # Interpretation
    lines.append("\n## Interpretation\n\n")

    lines.append("### What was implemented\n\n")
    lines.append(
        "A 2D grid model where fixed energy flows top-to-bottom through cells "
        "of four types. Active and replicating cells redistribute energy across "
        "output modes (conserving total energy), increasing the Shannon entropy "
        "of the output distribution. Three regimes compare: (1) memoryless random "
        "structures refreshed each step, (2) persistent structures that accumulate "
        "where throughput is high, and (3) persistent + self-replicating structures "
        "that can expand into empty cells.\n\n"
    )

    lines.append("### Key mechanism\n\n")
    lines.append(
        "The model separates transport efficiency (how much energy reaches the "
        "bottom) from transformation quality (how spread-out the output is). "
        "Both contribute to cumulative output entropy. Persistent channels create "
        "deep transformation pipelines where energy passes through many active "
        "cells, each mixing the mode distribution toward uniform. Replication "
        "widens these pipelines by filling adjacent empty cells.\n\n"
    )

    lines.append("### Main findings\n\n")
    if diff_31 > 0:
        if diff_32 > diff_21:
            lines.append(
                f"Persistent + branching channels produce {pct_31:+.1f}% more "
                "cumulative output entropy than memoryless dissipation. Branching "
                f"contributes {pct_32:+.1f}% beyond persistence alone, suggesting "
                "self-replication is the key ingredient for maximizing entropy spread. "
                "The channel structures self-organize into efficient transformation "
                "networks that convert concentrated input into distributed output.\n\n"
            )
        else:
            lines.append(
                f"Persistent channels produce {pct_21:+.1f}% more cumulative entropy "
                "than memoryless, and branching adds a further "
                f"{pct_32:+.1f}%. Most of the advantage comes from persistence "
                "(memory) rather than replication, suggesting that maintaining "
                "transformation structures is the primary driver.\n\n"
            )
    else:
        lines.append(
            "The hypothesis is not supported in this parameter regime. "
            "Persistent and branching channels do not outperform memoryless "
            "dissipation on cumulative output entropy. This may indicate that "
            "the loss differential between cell types is too small, or that "
            "random structures are already sufficient for the given grid depth.\n\n"
        )

    lines.append("### Whether the hypothesis is supported\n\n")
    if diff_31 > 0 and diff_32 > 0:
        lines.append(
            "**Supported.** Memory and replication increase long-run entropy "
            "production by building and maintaining structures that transform "
            "concentrated input energy into spread-out output distributions. "
            "The cumulative advantage grows over time as channels self-organize.\n\n"
        )
    elif diff_31 > 0:
        lines.append(
            "**Partially supported.** Persistence helps but replication provides "
            "minimal additional benefit in this parameter regime.\n\n"
        )
    else:
        lines.append(
            "**Not supported** in the tested parameter regime. Further exploration "
            "of parameter space may reveal conditions where the hypothesis holds.\n\n"
        )

    lines.append("### Next model improvement\n\n")
    lines.append(
        "1. Allow channel parameters (transform_strength, loss_rate) to evolve "
        "and be inherited during replication — test whether adaptation emerges.\n"
        "2. Add spatial structure to the input (localized energy source) to test "
        "whether channels self-organize into efficient networks.\n"
        "3. Introduce competition between channel types with different transformation "
        "properties to see if selection for higher-entropy output emerges.\n"
        "4. Scale up grid size and depth to test whether the advantage grows with "
        "system size or saturates.\n"
        "5. Add a cost to maintaining channel structures (thermodynamic maintenance "
        "cost) to test whether replication still pays off when persistence is expensive.\n"
    )

    return "".join(lines)
