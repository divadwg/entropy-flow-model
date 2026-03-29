# Entropy Flow Model

A simulation framework that discovers how structured energy-channeling patterns emerge, persist, and self-reinforce in a thermodynamic flow system — and identifies the causal mechanism responsible.

## Central Finding

**Selection-like structure arises from a throughput–persistence feedback loop:** cells that channel energy well persist longer, and because they persist they continue to channel energy. This is not entropy maximization, not statistical complexity maximization, and not an implementation artefact. It is a minimal selection law operating through physics alone.

The evidence comes from six layers of experiments, culminating in a causal intervention that deliberately breaks the feedback loop and shows structured regimes collapse, and a dimensionless control parameter (the flow-persistence number) that predicts regime membership better than entropy production or statistical complexity.

## Model

### Grid
- 2D grid (10x40 by default), energy flows top to bottom
- Each cell has one of 4 states: empty, passive, active, replicating
- Energy is distributed across K=16 modes per cell
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
This conserves total energy while increasing mode entropy. A transform cost creates a tradeoff: higher alpha produces more entropy but reduces throughput.

### Energy Conservation
```
energy_in = energy_out + energy_lost  (verified per step)
```

## Six Layers of Experiments

### Layer 1: Base Model (`run_all.py`)
Compares three regimes — memoryless, persistent, and persistent+branching — across parameter sweeps. Establishes that persistence and replication create stable channel structures with higher cumulative entropy.

### Layer 2: Evolution Extension (`run_evolution.py`)
Adds heritable per-cell traits (transform_strength, split_factor, persist_threshold, mode_bias) that are copied with mutation during replication. Traits evolve directionally toward more efficient configurations.

### Layer 3: Paper Experiment Suite (`run_suite.py`)
Five experiments testing whether evolution discovers efficient entropy-producing configurations:

| Experiment | Question | Result |
|-----------|----------|--------|
| Evolving vs Best Frozen | Can evolution match exhaustive parameter search? | Within 4.4% despite no prior knowledge |
| Heterogeneous Environments | Does spatial variation favor evolution? | Gap widens; frozen search adapts per-environment |
| Changing Environments | Does temporal variation favor evolution? | Gap narrows to 1.7% under drift |
| Ablation Tests | Which ingredients matter? | Replication necessary; tradeoff is primary constraint |
| Long-Run Behavior | Does performance plateau? | Alpha evolves toward optimum (0.15→0.124) |

**3 of 5 hypotheses supported.** Moderate evidence for evolutionary competitiveness, strongest under environmental change.

### Layer 4: Statistical Complexity (`run_complexity.py`)
Adds the statistical complexity metric (normalized entropy × disequilibrium). The evolving system maintains intermediate complexity — neither frozen order nor random diffusion — consistent with a structured-but-not-trivial regime.

### Layer 5: Mechanism Identification

#### 5a. Local Reinforcement (`run_reinforcement.py`)
Tests whether explicit motif-level memory (reinforcing patterns that persist) improves the system. **Finding:** the baseline model already implements self-reinforcement through the throughput EMA — adding explicit reinforcement is largely redundant. Without noise, it causes complete lock-in.

#### 5b. Causal Decoupling (`run_decoupling.py`)
The decisive experiment. Systematically breaks the throughput → persistence feedback loop:

| Condition | Persistence | Propagation | EP | Active Cells |
|-----------|------------|-------------|------|------|
| **baseline** | 99.2 | 0.024 | 3.17 | 287 |
| lifetime cap (50) | 24.1 (-76%) | 0.114 (+383%) | 3.02 (-5%) | 296 |
| random override (5%) | 49.4 (-50%) | 0.101 (+328%) | 3.07 (-3%) | 304 |
| throughput-blind | 49.8 (-50%) | 0.102 (+333%) | 3.07 (-3%) | 303 |
| **anti-coupled** | **3.8 (-96%)** | **0.000 (-100%)** | **0.51 (-84%)** | **14** |

**Anti-coupling (inverting the loop) destroys everything.** This confirms the throughput–persistence coupling is the operative selection law, not an artefact.

**Entropy production is roughly constant** across mild interventions (3.02–3.17), confirming structure is NOT selected for its entropy-producing properties.

**Path dependence tracks coupling:** anti-coupled trajectories converge (Hamming 0.075); baseline trajectories diverge (0.657). The coupling creates contingent histories.

### Layer 6: Flow-Persistence Number (`run_lambda.py`)
Operationalises the throughput–persistence feedback as a dimensionless control parameter and tests whether it predicts the structured regime better than entropy production or statistical complexity.

Three candidates are measured empirically from the simulations:

| Candidate | Definition | What it measures |
|-----------|-----------|-----------------|
| Lambda_A | log10(1+survived) − log10(1+disrupted) | Net persistence vs turnover (log-scale) |
| Lambda_B | (mean_tp_survived − mean_tp_disrupted) / mean_tp_all | Throughput selection gradient |
| Lambda_C | std(age) / mean(age) | Persistence concentration (age CV) |

Key results (Spearman rank correlation with persistence):

| Predictor | rho | p |
|-----------|-----|---|
| **Lambda_A** | **0.925** | **<0.0001** |
| **Lambda_B** | **0.934** | **<0.0001** |
| EP | 0.687 | 0.007 |
| Complexity | −0.295 | 0.306 |

Lambda_B separates the three regime classes:
- **Collapse** (memoryless, anti-coupled): Lambda_B ≈ 0.00
- **Structured** (baseline, decoupled variants): Lambda_B ≈ 0.01–0.38
- **Lock-in** (no noise): Lambda_B = 1.00

Anti-coupling drives Lambda_B to near zero (0.005), confirming the throughput–persistence coupling is the operative mechanism. Lambda_C distinguishes lock-in (CV ≈ 0.01) from structured (CV ≈ 1.0) from collapse (CV ≈ 0.0).

## Running

```bash
pip install -r requirements.txt

# Layer 1: Base model (~2 min)
python run_all.py

# Layer 2: Evolution extension (~5 min)
python run_evolution.py

# Layer 3: Paper experiment suite (~12 min)
python run_suite.py

# Layer 4: Statistical complexity (~3 min)
python run_complexity.py

# Layer 5a: Reinforcement experiment (~3 min)
python run_reinforcement.py

# Layer 5b: Decoupling experiment (~2 min)
python run_decoupling.py

# Layer 6: Flow-persistence number (~2 min)
python run_lambda.py
```

## Outputs

### Layer 1: Base Model
- `output/` — CSVs, summary, interpretation report
- `plots/` — cumulative entropy, sweeps, phase diagram (13 plots)

### Layer 2: Evolution Extension
- `evo_output/` — fixed vs evolving results, mutation sweep, trait ablation
- `evo_plots/` — EP comparison, trait evolution, lineage dominance (7 plots)

### Layer 3: Paper Suite
- `suite_output/` — per-experiment CSVs, hypothesis assessment report
- `suite_plots/` — per-experiment figures, summary comparison (8 plots)
- `supplementary_material.pdf` — paper-ready supplementary material

### Layer 4: Complexity
- `figures/complexity_over_time.png` — complexity across all regimes
- `figures/complexity_vs_alpha.png` — complexity vs transform strength

### Layer 5a: Reinforcement
- `reinf_output/` — sweep results, crossover analysis, report
- `reinf_plots/` — sweep time series, summary bars, path dependence, crossover (4 plots)

### Layer 5b: Decoupling
- `decouple_output/` — sweep results, path dependence, causal report
- `decouple_plots/` — time series, structure panel, complexity/entropy panel, path dependence, structure ranking (5 plots)

### Layer 6: Flow-Persistence Number
- `lambda_output/` — sweep results, summary table, report
- `lambda_plots/` — Lambda vs metrics, Lambda vs EP/complexity, regime phase diagram, time series (4 plots)

### Combined
- `all_plots.pdf` — all plots from all layers in one PDF

## The Argument in Summary

1. Persistent channel structures produce more cumulative entropy than memoryless ones (Layer 1).
2. Heritable traits evolve directionally toward efficient configurations (Layers 2–3).
3. The system occupies an intermediate complexity regime, not maximizing entropy or complexity (Layer 4).
4. The throughput EMA already functions as local memory of success — explicit reinforcement is redundant (Layer 5a).
5. **Breaking the throughput–persistence coupling destroys structured regimes** (Layer 5b).
6. A dimensionless flow-persistence number predicts regime membership (Spearman rho > 0.92) better than entropy production (0.69) or statistical complexity (−0.30) (Layer 6).

The selection law is:

> pattern → throughput → persistence → continued channeling → more throughput → more persistence

This is not entropy maximization. It is self-reinforcing structure maintenance through energy flow.
