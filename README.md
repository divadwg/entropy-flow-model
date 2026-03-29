# Entropy Flow Model

Tests whether persistent and replicating channel structures can increase long-run entropy production by transforming concentrated input energy into a more spread-out output distribution.

## Hypothesis

Fixed energy input flows through a 2D grid from top to bottom. Channel structures can redistribute energy across output modes (increasing distribution entropy) while conserving total energy. Memory (persistence) and replication may help create and maintain these transforming structures, leading to higher cumulative output entropy compared to memoryless dissipation.

The key insight: the degree of freedom is **distribution**, not energy creation. A system that spreads energy across more output modes produces higher entropy even if total throughput is the same.

## Model

### Grid
- 2D grid (height x width), energy flows top to bottom
- Each cell has one of 4 states: empty, passive, active, replicating
- Energy is distributed across K modes per cell (K=16 by default)
- Input: all energy concentrated in mode 0 (low entropy)
- Output: measured at bottom row across all modes

### Cell Types
| Type | Loss Rate | Transformation | Persistence | Replication |
|------|-----------|----------------|-------------|-------------|
| Empty | 5% | None | No | No |
| Passive | 1% | None | Regime 2/3 | No |
| Active | 2% | Mix toward uniform | Regime 2/3 | No |
| Replicating | 2% | Mix toward uniform | Regime 3 | Yes |

### Energy Transformation
Active and replicating cells mix energy across modes:
```
e_out = (1 - alpha) * e_in + alpha * (sum(e_in) / K) * ones(K)
```
This conserves total energy while increasing mode entropy. Multiple active cells in sequence compound the mixing, driving the distribution toward uniform.

### Three Regimes
1. **Memoryless**: All cell states re-randomized each step. No persistent structure.
2. **Persistent**: Cells with high throughput persist. Active channels accumulate in productive locations. No replication.
3. **Persistent + Branching**: Like regime 2, but active cells can copy into adjacent empty cells when throughput is high.

### Metrics
- **Output entropy**: Shannon entropy of mode distribution at bottom (bits)
- **Fragmentation**: Count of active (cell, mode) pairs at output
- **Effective modes**: exp(entropy in nats) -- how many modes carry significant energy
- **KL from uniform**: Distance from maximum-entropy (blackbody-like) distribution
- **Energy throughput**: Total energy exported at bottom vs input
- **Active area**: Count of transforming cells
- **Channel age**: Mean persistence of non-empty cells

### Energy Conservation
Total energy is conserved at each step except for explicit losses:
- Cell-type loss rates (dissipation at each cell)
- Boundary losses (energy spreading off grid edges)

energy_in = energy_out + energy_lost (verified per step)

## Three Layers

The model has three layers, each building on the previous:

### Layer 1: Base Model (`run_all.py`)
The core 3-regime simulation comparing memoryless, persistent, and branching channel structures. Includes parameter sweeps and phase diagrams.

### Layer 2: Evolution Extension (`run_evolution.py`)
Adds heritable per-cell traits (transform_strength, split_factor, persist_threshold, mode_bias) that are copied with mutation during replication. Compares fixed-trait vs evolving populations. Includes mutation sweeps and trait ablation.

### Layer 3: Paper Experiment Suite (`run_suite.py`)
Five experiments designed to produce paper-quality evidence:

| Experiment | Question |
|-----------|----------|
| **Exp 1**: Evolving vs Best Frozen | Can evolution match exhaustive parameter search? |
| **Exp 2**: Heterogeneous Environments | Does spatial variation favor evolution? |
| **Exp 3**: Changing Environments | Does temporal variation favor evolution? |
| **Exp 4**: Ablation Tests | Which ingredients (mutation, replication, persistence) matter? |
| **Exp 5**: Long-Run Behavior | Does evolution continue improving over 8000+ steps? |

Tests five hypotheses (H1–H5) with honest assessment and guardrails against overclaiming.

## Running

```bash
cd entropy_flow_model
pip install -r requirements.txt

# Layer 1: Base model (fast, ~2 min)
python run_all.py

# Layer 2: Evolution extension (~5 min)
python run_evolution.py

# Layer 3: Full paper suite (~12 min)
python run_suite.py
```

## Outputs

### Layer 1: Base Model
- `output/main_comparison.csv` -- per-step metrics for all regimes and seeds
- `output/regime_summary.csv` -- summary statistics per regime
- `output/sweep_*.csv` -- parameter sweep results
- `output/results_report.md` -- full interpretation report
- `plots/cumulative_entropy.png` -- main result: cumulative output entropy over time
- `plots/output_entropy.png`, `fragmentation.png`, `energy_balance.png`, etc.
- `plots/sweep_*.png` -- parameter sweep results
- `plots/phase_diagram.png` -- branching advantage across parameters

### Layer 2: Evolution Extension
- `evo_output/evolving_results.csv`, `fixed_results.csv` -- per-step metrics
- `evo_output/mutation_sweep.csv`, `trait_ablation.csv` -- sweep results
- `evo_output/evolution_report.md` -- interpretation report
- `evo_plots/fixed_vs_evolving.png` -- cumulative EP, per-step EP, entropy, throughput
- `evo_plots/trait_evolution.png`, `trait_histograms.png` -- trait dynamics
- `evo_plots/lineage_dominance.png` -- lineage diversity over time

### Layer 3: Paper Suite
- `suite_output/exp[1-5]_*.csv` -- per-experiment data
- `suite_output/suite_report.md` -- full hypothesis assessment report
- `suite_plots/exp[1-5]_*.png` -- per-experiment figures
- `suite_plots/summary_figure.png` -- master comparison across all conditions

## Parameter Sweeps (Layer 1)

| Parameter | Values | Tests |
|-----------|--------|-------|
| persistence_strength | 0.0 -- 0.95 | How much persistence matters |
| replication_prob | 0.0 -- 0.5 | How much branching matters |
| transform_strength | 0.05 -- 0.8 | How much mode mixing matters |
| mutation_rate | 0.0 -- 0.1 | Whether noise helps or hurts |
| E_in | 20 -- 500 | Whether advantage scales with energy |
