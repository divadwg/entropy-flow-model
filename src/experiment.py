"""
Simulation runner and parameter sweeps.

Provides functions to run individual simulations, compare regimes,
and sweep parameters while collecting per-step metrics.
"""
import copy
import numpy as np
import pandas as pd
from .grid import Grid
from .dynamics import flow_energy, update_states
from .metrics import collect_step_metrics

REGIME_NAMES = {1: "memoryless", 2: "persistent", 3: "persistent+branching"}


def build_params(config):
    """Flatten config sections into a single parameter dict."""
    p = {}
    p.update(config["dynamics"])
    p.update(config["states"])
    return p


def run_single(config, regime, seed):
    """Run one simulation and return per-step metric records."""
    rng = np.random.default_rng(seed)
    gc = config["grid"]
    grid = Grid(gc["height"], gc["width"], gc["n_modes"], rng)
    grid.initialize(
        active_frac=config["states"]["active_fraction"],
        passive_frac=config["states"]["passive_fraction"],
    )

    params = build_params(config)
    e_in_per_cell = config["energy"]["E_in"] / gc["width"]
    n_steps = config["simulation"]["n_steps"]

    records = []
    for step in range(n_steps):
        output, throughput, balance = flow_energy(grid, e_in_per_cell, params)
        m = collect_step_metrics(output, throughput, grid, balance, step)
        m["regime"] = regime
        m["regime_name"] = REGIME_NAMES[regime]
        m["seed"] = seed
        records.append(m)
        update_states(grid, throughput, params, regime)

    return records


def run_regime_comparison(config, seeds=None):
    """Run all 3 regimes across multiple seeds."""
    n_seeds = config["simulation"]["n_seeds"]
    if seeds is None:
        seeds = list(range(n_seeds))

    all_records = []
    total = len(seeds) * 3
    idx = 0
    for regime in [1, 2, 3]:
        for seed in seeds:
            idx += 1
            print(f"    [{idx}/{total}] Regime {regime} ({REGIME_NAMES[regime]}), seed {seed}")
            records = run_single(config, regime, seed)
            all_records.extend(records)

    return pd.DataFrame(all_records)


def run_parameter_sweep(config, param_name, values, n_seeds=None, n_steps=None):
    """Sweep one parameter across values for all 3 regimes."""
    sweep_cfg = config.get("sweeps", {})
    if n_seeds is None:
        n_seeds = sweep_cfg.get("n_seeds", 5)
    if n_steps is None:
        n_steps = sweep_cfg.get("n_steps", 300)

    seeds = list(range(n_seeds))
    all_records = []

    for val in values:
        cfg = copy.deepcopy(config)
        cfg["simulation"]["n_steps"] = n_steps
        _set_param(cfg, param_name, val)

        for regime in [1, 2, 3]:
            for seed in seeds:
                records = run_single(cfg, regime, seed)
                for r in records:
                    r["sweep_param"] = param_name
                    r["sweep_value"] = val
                all_records.extend(records)

    return pd.DataFrame(all_records)


def _set_param(config, param_name, value):
    """Set a parameter value in the appropriate config section."""
    section_map = {
        "persistence_strength": "states",
        "replication_prob": "states",
        "transform_strength": "dynamics",
        "mutation_rate": "states",
        "E_in": "energy",
    }
    section = section_map.get(param_name)
    if section and section in config:
        config[section][param_name] = value
    else:
        for s in ["dynamics", "states", "energy"]:
            if param_name in config.get(s, {}):
                config[s][param_name] = value
                return
