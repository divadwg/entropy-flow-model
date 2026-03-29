#!/usr/bin/env python3
"""
Paper Experiment Suite — Main Runner

Runs 5 experiments testing whether heritable variation + selection
can discover better entropy-producing channel designs than the best
frozen static systems, across heterogeneous and changing environments.

Usage:
    python run_suite.py [config.yaml]
"""
import os
import sys
import time
import json
import yaml
import pandas as pd

from src.suite import (
    experiment_1, experiment_2, experiment_3,
    experiment_4, experiment_5,
    generate_suite_report,
)
from src.suite_plotting import (
    plot_frozen_landscape, plot_exp1_comparison,
    plot_exp2_environments, plot_exp2_traits,
    plot_exp3_timeseries,
    plot_exp4_ablation,
    plot_exp5_longrun,
    plot_summary_figure,
)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    out_dir = config.get("suite_output_dir", "suite_output")
    plot_dir = config.get("suite_plots_dir", "suite_plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    gc = config["grid"]
    evo = config.get("evolution", {})
    sc = config.get("suite", {})

    print("=" * 64)
    print("PAPER EXPERIMENT SUITE")
    print("=" * 64)
    print(f"Grid: {gc['height']}x{gc['width']}, {gc['n_modes']} modes")
    print(f"Transform cost: {evo.get('transform_cost', 0.50)}")
    print(f"Mutation std: {evo.get('trait_mutation_std', 0.02)}")
    print()

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    t_total = time.time()

    # ── Experiment 1 ────────────────────────────────────────────
    print("EXPERIMENT 1: Evolving vs Best Frozen Baseline")
    t0 = time.time()
    exp1 = experiment_1(config)
    print(f"  Completed in {time.time()-t0:.0f}s\n")

    exp1["search_df"].to_csv(os.path.join(out_dir, "exp1_frozen_search.csv"), index=False)
    for regime in ["best_frozen", "default_frozen", "evolving"]:
        if regime in exp1:
            exp1[regime].to_csv(
                os.path.join(out_dir, f"exp1_{regime}.csv"), index=False
            )

    plot_frozen_landscape(exp1["search_df"], os.path.join(plot_dir, "exp1_landscape.png"))
    plot_exp1_comparison(exp1, os.path.join(plot_dir, "exp1_comparison.png"))

    # ── Experiment 2 ────────────────────────────────────────────
    print("EXPERIMENT 2: Heterogeneous Environments")
    t0 = time.time()
    exp2 = experiment_2(config)
    print(f"  Completed in {time.time()-t0:.0f}s\n")

    for env_name, data in exp2.items():
        data.get("search_df", pd.DataFrame()).to_csv(
            os.path.join(out_dir, f"exp2_{env_name}_search.csv"), index=False
        )
        for regime in ["best_frozen", "evolving"]:
            if regime in data:
                data[regime].to_csv(
                    os.path.join(out_dir, f"exp2_{env_name}_{regime}.csv"), index=False
                )

    plot_exp2_environments(exp2, os.path.join(plot_dir, "exp2_environments.png"))
    plot_exp2_traits(exp2, os.path.join(plot_dir, "exp2_traits.png"))

    # ── Experiment 3 ────────────────────────────────────────────
    print("EXPERIMENT 3: Changing Environments")
    t0 = time.time()
    # Reuse exp1's best frozen for the "average" baseline
    homogeneous_best = exp1.get("best_frozen_traits")
    exp3 = experiment_3(config, homogeneous_best=homogeneous_best)
    print(f"  Completed in {time.time()-t0:.0f}s\n")

    for env_name, data in exp3.items():
        for regime in ["best_frozen", "evolving"]:
            if regime in data:
                data[regime].to_csv(
                    os.path.join(out_dir, f"exp3_{env_name}_{regime}.csv"), index=False
                )

    plot_exp3_timeseries(exp3, os.path.join(plot_dir, "exp3_timeseries.png"))

    # ── Experiment 4 ────────────────────────────────────────────
    print("EXPERIMENT 4: Ablation Tests")
    t0 = time.time()
    exp4 = experiment_4(config)
    print(f"  Completed in {time.time()-t0:.0f}s\n")

    for abl_name, df in exp4.items():
        df.to_csv(os.path.join(out_dir, f"exp4_{abl_name}.csv"), index=False)

    plot_exp4_ablation(exp4, os.path.join(plot_dir, "exp4_ablation.png"))

    # ── Experiment 5 ────────────────────────────────────────────
    print("EXPERIMENT 5: Long-Run Behavior")
    t0 = time.time()
    exp5 = experiment_5(config)
    print(f"  Completed in {time.time()-t0:.0f}s\n")

    exp5["evolving_df"].to_csv(os.path.join(out_dir, "exp5_evolving.csv"), index=False)
    exp5["fixed_df"].to_csv(os.path.join(out_dir, "exp5_fixed.csv"), index=False)

    plot_exp5_longrun(exp5, os.path.join(plot_dir, "exp5_longrun.png"))

    # ── Summary ─────────────────────────────────────────────────
    print("Generating summary figure and report...")
    plot_summary_figure(exp1, exp2, exp3, os.path.join(plot_dir, "summary_figure.png"))

    report = generate_suite_report(exp1, exp2, exp3, exp4, exp5, config)
    with open(os.path.join(out_dir, "suite_report.md"), "w") as f:
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
