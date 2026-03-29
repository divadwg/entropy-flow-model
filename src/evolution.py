"""
Evolution experiment: heritable variation + selection on entropy production.

Compares fixed-trait persistent channels (R2-fixed) against evolving
branching channels (R3-evolving) where traits are inherited with mutation
during replication and selected by throughput-based persistence.
"""
import copy
import numpy as np
import pandas as pd
from .grid import Grid, N_TRAITS, TRAIT_NAMES, TRAIT_DEFAULTS, ACTIVE, REPLICATING
from .dynamics import flow_energy_with_traits, update_states_evolving
from .metrics import collect_step_metrics
from .experiment import build_params


def run_with_traits(config, seed, fixed=False, n_steps=None, evolving_traits=None,
                    env=None, frozen_traits=None):
    """
    Run one simulation with per-cell traits.

    fixed=True: traits frozen at defaults, no mutation.
    fixed=False: traits mutate during replication (evolving).
    frozen_traits: dict {trait_index: value} to freeze specific traits.
    env: Environment object for spatially/temporally varying conditions.

    Returns (records, lineage_history, trait_snapshots).
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

    # Apply frozen trait overrides
    if frozen_traits is not None:
        for trait_idx, value in frozen_traits.items():
            grid.traits[:, :, trait_idx] = value

    params = build_params(config)
    params["transform_cost"] = evo.get("transform_cost", 0.50)

    if frozen_traits is not None:
        params["trait_mutation_std"] = 0.0
        regime_name = "frozen"
    elif fixed:
        params["trait_mutation_std"] = 0.0
        regime_name = "fixed"
    else:
        params["trait_mutation_std"] = evo.get("trait_mutation_std", 0.02)
        regime_name = "evolving"

    if n_steps is None:
        n_steps = evo.get("n_steps", 800)

    e_in_default = config["energy"]["E_in"] / gc["width"]

    # Snapshot schedule
    snap_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    snap_steps = set(int(f * (n_steps - 1)) for f in snap_fracs)

    records = []
    lineage_history = []
    trait_snapshots = {}

    for step in range(n_steps):
        # Query environment for step-specific conditions
        if env is not None:
            step_e_in = env.get_e_in(step)
            params["transform_cost"] = env.get_transform_cost(step)
        else:
            step_e_in = e_in_default

        output, throughput, balance = flow_energy_with_traits(grid, step_e_in, params)
        m = collect_step_metrics(output, throughput, grid, balance, step)

        # Composite metric: rewards both mixing AND throughput
        m["entropy_production"] = m["output_entropy"] * m["energy_out"] / max(m["energy_in"], 1e-12)

        # Trait statistics
        tstats = grid.trait_statistics()
        for name, (mean, std) in tstats.items():
            m[f"trait_{name}_mean"] = mean
            m[f"trait_{name}_std"] = std

        # Lineage info
        lin = grid.lineage_counts()
        if lin:
            total_cells = sum(lin.values())
            top_id = max(lin, key=lin.get)
            m["n_lineages"] = len(lin)
            m["top_lineage_frac"] = lin[top_id] / total_cells
        else:
            m["n_lineages"] = 0
            m["top_lineage_frac"] = 0.0

        m["regime_name"] = regime_name
        m["seed"] = seed
        records.append(m)

        # Snapshots
        lineage_history.append(lin)
        if step in snap_steps:
            trait_snapshots[step] = grid.trait_values()

        update_states_evolving(grid, throughput, params, evolving_traits=evolving_traits)

    return records, lineage_history, trait_snapshots


def run_fixed_vs_evolving(config):
    """Compare R2-fixed vs R3-evolving across multiple seeds."""
    evo = config.get("evolution", {})
    n_seeds = evo.get("n_seeds", 8)
    n_steps = evo.get("n_steps", 800)
    seeds = list(range(n_seeds))

    fixed_records = []
    evolving_records = []
    all_lineage = {}
    all_snapshots = {}

    total = 2 * n_seeds
    idx = 0
    for seed in seeds:
        idx += 1
        print(f"    [{idx}/{total}] R2-fixed, seed {seed}")
        recs, _, _ = run_with_traits(config, seed, fixed=True, n_steps=n_steps)
        fixed_records.extend(recs)

        idx += 1
        print(f"    [{idx}/{total}] R3-evolving, seed {seed}")
        recs, lin_hist, snaps = run_with_traits(config, seed, fixed=False, n_steps=n_steps)
        evolving_records.extend(recs)
        all_lineage[seed] = lin_hist
        all_snapshots[seed] = snaps

    df_fixed = pd.DataFrame(fixed_records)
    df_evolving = pd.DataFrame(evolving_records)

    return df_fixed, df_evolving, all_lineage, all_snapshots


def run_mutation_sweep(config, mutation_values=None):
    """Sweep trait_mutation_std for the evolving regime."""
    evo = config.get("evolution", {})
    if mutation_values is None:
        mutation_values = evo.get("mutation_sweep", [0.0, 0.01, 0.03, 0.1, 0.3])

    n_seeds = evo.get("ablation_n_seeds", 5)
    n_steps = evo.get("ablation_n_steps", 500)
    seeds = list(range(n_seeds))

    all_records = []
    for mut_std in mutation_values:
        cfg = copy.deepcopy(config)
        cfg["evolution"]["trait_mutation_std"] = mut_std

        for seed in seeds:
            recs, _, _ = run_with_traits(cfg, seed, fixed=False, n_steps=n_steps)
            for r in recs:
                r["sweep_value"] = mut_std
            all_records.extend(recs)

    return pd.DataFrame(all_records)


def run_trait_ablation(config):
    """
    Test which traits matter by evolving subsets.

    Variants:
    - alpha_only: only transform_strength evolves
    - alpha_split: transform_strength + split_factor evolve
    - all: all traits evolve
    """
    from .grid import TRAIT_ALPHA, TRAIT_SPLIT

    evo = config.get("evolution", {})
    n_seeds = evo.get("ablation_n_seeds", 5)
    n_steps = evo.get("ablation_n_steps", 500)
    seeds = list(range(n_seeds))

    variants = {
        "alpha_only": [TRAIT_ALPHA],
        "alpha_split": [TRAIT_ALPHA, TRAIT_SPLIT],
        "all_traits": None,
    }

    all_records = []
    for var_name, evolving in variants.items():
        for seed in seeds:
            recs, _, _ = run_with_traits(config, seed, fixed=False,
                                         n_steps=n_steps, evolving_traits=evolving)
            for r in recs:
                r["variant"] = var_name
            all_records.extend(recs)

    return pd.DataFrame(all_records)


def compute_evolution_summary(df_fixed, df_evolving):
    """Summary statistics for fixed vs evolving comparison."""
    results = {}
    for label, df in [("fixed", df_fixed), ("evolving", df_evolving)]:
        n_steps = df["step"].max() + 1
        last = df[df["step"] > n_steps * 0.8]
        cum = df.groupby("seed")["output_entropy"].sum()
        cum_ep = df.groupby("seed")["entropy_production"].sum()

        row = {
            "cum_entropy_mean": float(cum.mean()),
            "cum_entropy_std": float(cum.std()),
            "cum_ep_mean": float(cum_ep.mean()),
            "cum_ep_std": float(cum_ep.std()),
            "final_entropy_mean": float(last["output_entropy"].mean()),
            "final_entropy_std": float(last["output_entropy"].std()),
            "final_ep_mean": float(last["entropy_production"].mean()),
            "final_ep_std": float(last["entropy_production"].std()),
            "final_energy_out": float(last["energy_out"].mean()),
            "final_active_area": float(last["active_area"].mean()),
            "final_eff_modes": float(last["effective_modes"].mean()),
        }

        # Trait stats (last 20%)
        for tname in TRAIT_NAMES:
            col = f"trait_{tname}_mean"
            if col in last.columns:
                row[f"final_{tname}"] = float(last[col].mean())
                row[f"final_{tname}_var"] = float(last[f"trait_{tname}_std"].mean())

        if label == "evolving" and "n_lineages" in last.columns:
            row["final_n_lineages"] = float(last["n_lineages"].mean())
            row["final_top_lineage_frac"] = float(last["top_lineage_frac"].mean())

        results[label] = row

    return results


def generate_evolution_report(summary, sweep_df, ablation_df, config):
    """Generate markdown report for the evolution experiment."""
    lines = []
    lines.append("# Evolution Experiment — Results Report\n\n")

    gc = config["grid"]
    evo = config.get("evolution", {})
    lines.append(f"Grid: {gc['height']}x{gc['width']}, {gc['n_modes']} modes, "
                 f"E_in={config['energy']['E_in']}\n")
    lines.append(f"Transform cost: {evo.get('transform_cost', 0.05)}, "
                 f"Mutation std: {evo.get('trait_mutation_std', 0.05)}\n")
    lines.append(f"Steps: {evo.get('n_steps', 800)}, "
                 f"Seeds: {evo.get('n_seeds', 8)}\n\n")

    # Main comparison — use entropy_production as primary metric
    f = summary["fixed"]
    e = summary["evolving"]
    diff_ep = e["cum_ep_mean"] - f["cum_ep_mean"]
    pct_ep = diff_ep / max(f["cum_ep_mean"], 1) * 100
    diff = e["cum_entropy_mean"] - f["cum_entropy_mean"]
    pct = diff / max(f["cum_entropy_mean"], 1) * 100

    lines.append("## Fixed vs Evolving Comparison\n\n")
    lines.append("| Metric | Fixed | Evolving | Diff |\n")
    lines.append("|--------|-------|----------|------|\n")
    lines.append(f"| **Cum. entropy production** | {f['cum_ep_mean']:.1f} +/- {f['cum_ep_std']:.1f} "
                 f"| {e['cum_ep_mean']:.1f} +/- {e['cum_ep_std']:.1f} "
                 f"| {diff_ep:+.1f} ({pct_ep:+.1f}%) |\n")
    lines.append(f"| Cumulative entropy | {f['cum_entropy_mean']:.1f} +/- {f['cum_entropy_std']:.1f} "
                 f"| {e['cum_entropy_mean']:.1f} +/- {e['cum_entropy_std']:.1f} "
                 f"| {diff:+.1f} ({pct:+.1f}%) |\n")
    lines.append(f"| Final entropy prod. | {f['final_ep_mean']:.3f} "
                 f"| {e['final_ep_mean']:.3f} "
                 f"| {e['final_ep_mean'] - f['final_ep_mean']:+.3f} |\n")
    lines.append(f"| Final entropy | {f['final_entropy_mean']:.3f} "
                 f"| {e['final_entropy_mean']:.3f} "
                 f"| {e['final_entropy_mean'] - f['final_entropy_mean']:+.3f} |\n")
    lines.append(f"| Energy throughput | {f['final_energy_out']:.2f} "
                 f"| {e['final_energy_out']:.2f} "
                 f"| {e['final_energy_out'] - f['final_energy_out']:+.2f} |\n")
    lines.append(f"| Active area | {f['final_active_area']:.0f} "
                 f"| {e['final_active_area']:.0f} "
                 f"| {e['final_active_area'] - f['final_active_area']:+.0f} |\n")

    # Evolved trait values
    lines.append("\n## Evolved Trait Values (final 20%)\n\n")
    lines.append("| Trait | Fixed (default) | Evolved mean | Evolved std | Direction |\n")
    lines.append("|-------|----------------|-------------|------------|----------|\n")

    for i, tname in enumerate(TRAIT_NAMES):
        default = TRAIT_DEFAULTS[i]
        e_mean = e.get(f"final_{tname}", default)
        e_var = e.get(f"final_{tname}_var", 0)
        if e_mean > default * 1.05:
            direction = "INCREASED"
        elif e_mean < default * 0.95:
            direction = "DECREASED"
        else:
            direction = "~unchanged"
        lines.append(f"| {tname} | {default:.2f} | {e_mean:.3f} | {e_var:.3f} | {direction} |\n")

    # Lineage selection
    if "final_n_lineages" in e:
        lines.append("\n## Lineage Selection\n\n")
        lines.append(f"- Distinct lineages at end: {e['final_n_lineages']:.0f}\n")
        lines.append(f"- Top lineage fraction: {e['final_top_lineage_frac']:.2f}\n")
        if e["final_top_lineage_frac"] > 0.3:
            lines.append("- Strong lineage selection: one lineage dominates >30% of active cells.\n")
        elif e["final_top_lineage_frac"] > 0.15:
            lines.append("- Moderate lineage selection: partial dominance.\n")
        else:
            lines.append("- Weak lineage selection: no single lineage dominates.\n")

    # Mutation sweep
    if sweep_df is not None and len(sweep_df) > 0:
        lines.append("\n## Mutation Rate Sweep\n\n")
        cum = sweep_df.groupby(["sweep_value", "seed"])["output_entropy"].sum().reset_index()
        for val in sorted(cum["sweep_value"].unique()):
            mean = cum[cum["sweep_value"] == val]["output_entropy"].mean()
            std = cum[cum["sweep_value"] == val]["output_entropy"].std()
            lines.append(f"- mutation_std={val}: cum_entropy = {mean:.1f} +/- {std:.1f}\n")

        means = cum.groupby("sweep_value")["output_entropy"].mean()
        best_mut = means.idxmax()
        lines.append(f"\nOptimal mutation rate: {best_mut}\n")
        if best_mut > 0 and best_mut < means.index.max():
            lines.append("Evidence for edge-of-chaos: intermediate mutation rate is best.\n")

    # Ablation
    if ablation_df is not None and len(ablation_df) > 0:
        lines.append("\n## Trait Ablation\n\n")
        cum = ablation_df.groupby(["variant", "seed"])["output_entropy"].sum().reset_index()
        for var in sorted(cum["variant"].unique()):
            mean = cum[cum["variant"] == var]["output_entropy"].mean()
            std = cum[cum["variant"] == var]["output_entropy"].std()
            lines.append(f"- {var}: cum_entropy = {mean:.1f} +/- {std:.1f}\n")

    # Verdict — use entropy_production as primary metric
    lines.append("\n## Verdict\n\n")

    lines.append("### Does the evolving system increase entropy production over time?\n")
    if diff_ep > 0:
        lines.append(f"YES — evolving system produces {pct_ep:+.1f}% more cumulative entropy production.\n\n")
    else:
        lines.append("NO — evolving system does not outperform fixed channels.\n\n")

    lines.append("### Does the evolving system outperform fixed-parameter persistent systems?\n")
    if diff_ep > 0 and pct_ep > 2:
        lines.append(f"YES — {pct_ep:+.1f}% advantage over fixed channels.\n\n")
    elif diff_ep > 0:
        lines.append("MARGINAL — small positive advantage within noise.\n\n")
    else:
        lines.append("NO — fixed channels perform comparably or better.\n\n")

    lines.append("### Do traits show directional change (not just noise)?\n")
    directional = False
    for i, tname in enumerate(TRAIT_NAMES):
        e_mean = e.get(f"final_{tname}", TRAIT_DEFAULTS[i])
        e_var = e.get(f"final_{tname}_var", 0)
        shift = abs(e_mean - TRAIT_DEFAULTS[i])
        if shift > max(e_var, 0.01):
            lines.append(f"- {tname}: directional shift detected "
                         f"({TRAIT_DEFAULTS[i]:.2f} -> {e_mean:.3f})\n")
            directional = True
    if not directional:
        lines.append("No clear directional trait evolution detected.\n")
    lines.append("\n")

    # Overall
    lines.append("### Overall Assessment\n\n")
    if diff_ep > 0 and pct_ep > 2 and directional:
        lines.append(
            "**STRONG SUPPORT**: Heritable variation + selection discovers "
            "better entropy-producing structures. Traits evolve directionally "
            "and the evolving system outperforms fixed channels. This demonstrates "
            "that selection on thermodynamic performance can drive cumulative "
            "improvement — the key step from physics to life.\n"
        )
    elif diff_ep > 0 and (pct_ep > 2 or directional):
        lines.append(
            "**PARTIAL SUPPORT**: Some evidence for selection-driven improvement. "
            "The evolving system shows advantages but the signal is not overwhelming. "
            "Longer runs or different parameters might strengthen the result.\n"
        )
    elif diff_ep > 0:
        lines.append(
            "**WEAK SUPPORT**: The evolving system performs slightly better but the "
            "advantage is small and may not reflect genuine selection.\n"
        )
    else:
        lines.append(
            "**NOT SUPPORTED**: Evolution does not improve entropy production beyond "
            "fixed persistent channels in this parameter regime. The fixed parameters "
            "may already be near-optimal, or the trait space may not offer enough "
            "room for improvement.\n"
        )

    lines.append("\n### Next Steps\n\n")
    lines.append(
        "1. Increase transform_cost to create a stronger efficiency/mixing tradeoff.\n"
        "2. Add spatial structure to the input energy to reward spatial organization.\n"
        "3. Allow loss_rate to evolve as a heritable trait.\n"
        "4. Test whether trait combinations emerge that are non-obvious.\n"
        "5. Run longer simulations to see if late-stage evolution produces breakthroughs.\n"
    )

    return "".join(lines)
