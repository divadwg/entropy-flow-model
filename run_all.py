#!/usr/bin/env python3
"""
Entropy Flow Model — Main Runner

Tests whether persistent and replicating channel structures can increase
long-run entropy production by transforming concentrated input energy into
a more spread-out output distribution.

Usage:
    python run_all.py [config.yaml]
"""
import os
import sys
import time
import yaml

from src.experiment import run_regime_comparison, run_parameter_sweep
from src.analysis import compute_regime_summary, summarize_sweep, generate_report
from src.plotting import (
    plot_cumulative_entropy, plot_output_entropy, plot_fragmentation,
    plot_structure, plot_energy_balance, plot_kl_divergence,
    plot_effective_modes, plot_sweep, plot_phase_diagram,
)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    output_dir = config["output_dir"]
    plots_dir = config["plots_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    gc = config["grid"]
    print("=" * 64)
    print("ENTROPY FLOW MODEL")
    print("=" * 64)
    print(f"Grid: {gc['height']}x{gc['width']}, {gc['n_modes']} modes")
    print(f"E_in: {config['energy']['E_in']}")
    print(f"Steps: {config['simulation']['n_steps']}, "
          f"Seeds: {config['simulation']['n_seeds']}")
    print()

    # ── Phase 1: Main regime comparison ──────────────────────────
    print("Phase 1: Main regime comparison")
    t0 = time.time()
    df = run_regime_comparison(config)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s\n")

    df.to_csv(os.path.join(output_dir, "main_comparison.csv"), index=False)

    summary = compute_regime_summary(df)
    summary.to_csv(os.path.join(output_dir, "regime_summary.csv"), index=False)

    print("Regime Summary:")
    for _, row in summary.iterrows():
        print(f"  {row['regime_name']:25s}  "
              f"cum_S={row['cum_entropy_mean']:7.1f} +/- {row['cum_entropy_std']:5.1f}  "
              f"final_S={row['final_entropy_mean']:.3f}  "
              f"area={row['final_active_area_mean']:5.0f}  "
              f"eff_modes={row['final_eff_modes']:.1f}")
    print()

    # ── Phase 1b: Generate plots ─────────────────────────────────
    print("Generating main plots...")
    plot_cumulative_entropy(df, os.path.join(plots_dir, "cumulative_entropy.png"))
    plot_output_entropy(df, os.path.join(plots_dir, "output_entropy.png"))
    plot_fragmentation(df, os.path.join(plots_dir, "fragmentation.png"))
    plot_structure(df, os.path.join(plots_dir, "structure_persistence.png"))
    plot_energy_balance(df, os.path.join(plots_dir, "energy_balance.png"))
    plot_kl_divergence(df, os.path.join(plots_dir, "kl_divergence.png"))
    plot_effective_modes(df, os.path.join(plots_dir, "effective_modes.png"))
    print("  Done.\n")

    # ── Phase 2: Parameter sweeps ────────────────────────────────
    sweep_cfg = config.get("sweeps", {})
    sweep_params = [
        "persistence_strength", "replication_prob", "transform_strength",
        "mutation_rate", "E_in",
    ]

    sweep_results = {}
    sweep_summaries = {}

    for param in sweep_params:
        values = sweep_cfg.get(param)
        if values is None:
            continue
        print(f"Phase 2: Sweep {param} = {values}")
        t0 = time.time()
        sweep_df = run_parameter_sweep(config, param, values)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        sweep_df.to_csv(os.path.join(output_dir, f"sweep_{param}.csv"), index=False)
        sweep_results[param] = sweep_df

        summary_text = summarize_sweep(sweep_df, param)
        sweep_summaries[param] = summary_text
        print(summary_text)
        print()

        plot_sweep(sweep_df, param, os.path.join(plots_dir, f"sweep_{param}.png"))

    # ── Phase 2b: Phase diagram ──────────────────────────────────
    if len(sweep_results) >= 2:
        print("Generating phase diagram...")
        plot_phase_diagram(sweep_results,
                           os.path.join(plots_dir, "phase_diagram.png"))
        print("  Done.\n")

    # ── Phase 3: Report ──────────────────────────────────────────
    print("Generating report...")
    report = generate_report(summary, sweep_summaries, config)
    report_path = os.path.join(output_dir, "results_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print()
    print(report)
    print()
    print(f"Outputs: {output_dir}/")
    print(f"Plots:   {plots_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
