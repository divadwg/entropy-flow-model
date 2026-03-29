#!/usr/bin/env python3
"""
Evolution Experiment — Main Runner

Tests whether heritable variation + selection can drive cumulative
improvements in entropy production beyond fixed persistent structures.

Usage:
    python run_evolution.py [config.yaml]
"""
import os
import sys
import time
import yaml

from src.evolution import (
    run_fixed_vs_evolving, run_mutation_sweep, run_trait_ablation,
    compute_evolution_summary, generate_evolution_report,
)
from src.evo_plotting import (
    plot_fixed_vs_evolving, plot_trait_evolution, plot_trait_histograms,
    plot_lineage_dominance, plot_mutation_sweep, plot_ablation,
    plot_energy_vs_entropy_tradeoff,
)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    evo = config.get("evolution", {})
    output_dir = config.get("evo_output_dir", "evo_output")
    plots_dir = config.get("evo_plots_dir", "evo_plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 64)
    print("EVOLUTION EXPERIMENT")
    print("=" * 64)
    gc = config["grid"]
    print(f"Grid: {gc['height']}x{gc['width']}, {gc['n_modes']} modes")
    print(f"Transform cost: {evo.get('transform_cost', 0.05)}")
    print(f"Mutation std: {evo.get('trait_mutation_std', 0.05)}")
    print(f"Steps: {evo.get('n_steps', 800)}, Seeds: {evo.get('n_seeds', 8)}")
    print()

    # ── Phase 1: Fixed vs Evolving ───────────────────────────────
    print("Phase 1: Fixed vs Evolving comparison")
    t0 = time.time()
    df_fixed, df_evolving, all_lineage, all_snapshots = run_fixed_vs_evolving(config)
    print(f"  Completed in {time.time() - t0:.1f}s\n")

    df_fixed.to_csv(os.path.join(output_dir, "fixed_results.csv"), index=False)
    df_evolving.to_csv(os.path.join(output_dir, "evolving_results.csv"), index=False)

    summary = compute_evolution_summary(df_fixed, df_evolving)

    print("Summary:")
    for label in ["fixed", "evolving"]:
        s = summary[label]
        print(f"  {label:10s}  cum_EP={s['cum_ep_mean']:7.1f} +/- {s['cum_ep_std']:5.1f}  "
              f"cum_S={s['cum_entropy_mean']:7.1f}  "
              f"final_EP={s['final_ep_mean']:.3f}  E_out={s['final_energy_out']:.2f}")
    diff_ep = summary["evolving"]["cum_ep_mean"] - summary["fixed"]["cum_ep_mean"]
    pct_ep = diff_ep / max(summary["fixed"]["cum_ep_mean"], 1) * 100
    print(f"  EP Advantage: {diff_ep:+.1f} ({pct_ep:+.1f}%)")
    print()

    # Evolved traits
    print("Evolved traits (final):")
    from src.grid import TRAIT_NAMES, TRAIT_DEFAULTS
    for i, tname in enumerate(TRAIT_NAMES):
        key = f"final_{tname}"
        if key in summary["evolving"]:
            e_val = summary["evolving"][key]
            print(f"  {tname:25s}  default={TRAIT_DEFAULTS[i]:.2f}  evolved={e_val:.3f}")
    print()

    # ── Phase 1b: Plots ──────────────────────────────────────────
    print("Generating plots...")
    plot_fixed_vs_evolving(df_fixed, df_evolving,
                           os.path.join(plots_dir, "fixed_vs_evolving.png"))
    plot_trait_evolution(df_evolving,
                        os.path.join(plots_dir, "trait_evolution.png"))
    if all_snapshots:
        plot_trait_histograms(all_snapshots,
                              os.path.join(plots_dir, "trait_histograms.png"))
    if all_lineage:
        plot_lineage_dominance(all_lineage,
                               os.path.join(plots_dir, "lineage_dominance.png"))
    plot_energy_vs_entropy_tradeoff(df_fixed, df_evolving,
                                    os.path.join(plots_dir, "energy_entropy_tradeoff.png"))
    print("  Done.\n")

    # ── Phase 2: Mutation rate sweep ─────────────────────────────
    mut_values = evo.get("mutation_sweep", [0.0, 0.01, 0.03, 0.1, 0.3])
    print(f"Phase 2: Mutation rate sweep = {mut_values}")
    t0 = time.time()
    sweep_df = run_mutation_sweep(config, mut_values)
    print(f"  Completed in {time.time() - t0:.1f}s")

    sweep_df.to_csv(os.path.join(output_dir, "mutation_sweep.csv"), index=False)
    plot_mutation_sweep(sweep_df, os.path.join(plots_dir, "mutation_sweep.png"))

    # Print sweep summary
    cum = sweep_df.groupby(["sweep_value", "seed"])["output_entropy"].sum().reset_index()
    for val in sorted(cum["sweep_value"].unique()):
        mean = cum[cum["sweep_value"] == val]["output_entropy"].mean()
        print(f"  mut_std={val}: cum_entropy={mean:.1f}")
    print()

    # ── Phase 3: Trait ablation ──────────────────────────────────
    print("Phase 3: Trait ablation")
    t0 = time.time()
    ablation_df = run_trait_ablation(config)
    print(f"  Completed in {time.time() - t0:.1f}s")

    ablation_df.to_csv(os.path.join(output_dir, "trait_ablation.csv"), index=False)
    plot_ablation(ablation_df, os.path.join(plots_dir, "trait_ablation.png"))

    # Print ablation summary
    cum = ablation_df.groupby(["variant", "seed"])["output_entropy"].sum().reset_index()
    for var in sorted(cum["variant"].unique()):
        mean = cum[cum["variant"] == var]["output_entropy"].mean()
        print(f"  {var}: cum_entropy={mean:.1f}")
    print()

    # ── Phase 4: Report ──────────────────────────────────────────
    print("Generating report...")
    report = generate_evolution_report(summary, sweep_df, ablation_df, config)
    with open(os.path.join(output_dir, "evolution_report.md"), "w") as f:
        f.write(report)

    print()
    print(report)
    print()
    print(f"Outputs: {output_dir}/")
    print(f"Plots:   {plots_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
