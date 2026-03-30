# Entropy Flow Model

A simulation framework that discovers how structured energy-channeling patterns emerge, persist, and self-reinforce in thermodynamic flow systems — and identifies the causal mechanism responsible.

Code: [github.com/divadwg/entropy-flow-model](https://github.com/divadwg/entropy-flow-model)

## Central Finding

**Selection operates independently of dissipation.** The throughput-persistence feedback loop produces identical structured regimes whether structure increases or decreases aggregate dissipation.

Under standard loss rates (active cells 2%, empty 5%), more structure means less waste. Under inverted loss rates (active 8%, empty 2%), more structure means *more* waste. In both cases: Lambda_B ~ 0.38, persistence ~ 99. The coupling is dissipation-agnostic.

This is inconsistent with frameworks that predict systems organise to extremise dissipation. The mechanism that produces structure is dissipation-agnostic: even if MEPP or dissipative adaptation are correct as thermodynamic principles, they do not explain why structure arises in this class of systems.

## Intellectual Context

### Kauffman's Autonomous Agents

The conceptual framework is Kauffman's (2000). He argued that "autonomous agents" — entities performing thermodynamic work cycles to maintain their own existence — constitute a form of selection emerging from thermodynamic constraints without any externally imposed fitness function. His core insight: *persistence is the fitness proxy*. Structures that channel energy effectively persist, and because they persist they continue to channel energy. This is throughput-persistence coupling, described in different language. Our model is a computational instantiation of this concept. Our contributions are measurement: (a) the coupling is causal (breaking it destroys structure), (b) it produces a bounded regime parameterised by Lambda_B, (c) Lambda_B is formally a Price covariance, and (d) the regime is independent of dissipation direction.

### The Price Equation

Price (1970) showed that selection is a statistical relationship — a covariance between character and reproductive success — requiring no optimand. We derive that:

```
Lambda_B = Cov(w, z) / [Var(w) * z_bar] = beta_{z<-w} / z_bar
```

where `w` is binary survival (0 or 1), `z` is throughput, and `beta_{z<-w}` is the regression of throughput on survival. This is an algebraic identity, not an analogy. No loss rates appear in the derivation: the Price equivalence holds identically under any loss-rate parameterisation (confirmed empirically: Lambda_B = 0.379 standard vs 0.389 inverted).

When Lambda_B > 0: survivors have higher throughput (functional selection).
When Lambda_B = 0: survival is independent of throughput (no selection).
When Lambda_B < 0: survivors have *lower* throughput (anti-selection).
When Lambda_B = 1: all cells survive, no disruption (lock-in).

The throughput-persistence coupling is not *like* selection — it *is* selection, in the precise statistical sense that Price defined.

### Langton's Edge of Chaos

The three-regime structure (collapse / structured / lock-in) maps onto Langton's (1990) classification (ordered / complex / chaotic). Lambda_B adds a physical interpretation (normalised Price covariance) and an empirical protocol (computed from survival statistics, not rule space).

### Self-Organised Criticality

The observation that driven-dissipative systems can naturally self-organise near critical transitions (Bak et al., 1987) provides another structural parallel. The throughput-persistence coupling is a driven-dissipative mechanism that produces bounded intermediate regimes.

### On Dissipative Adaptation

England (2013) showed formally that self-replicating structures are statistically more likely to arise in environments where they absorb and dissipate energy effectively. This is a rigorous thermodynamic result about the probability of observing replicators, given the constraints of microscopic reversibility and detailed balance.

The colloquial interpretation — "life exists because it dissipates energy" or "organisms are selected *for* dissipation" — reverses the causal arrow. England's result says dissipation is a *constraint on what is likely to be observed*, not a *cause of* or *selection pressure for* structure. Our inverted-loss test is consistent with England's formal result (structure can arise under many dissipation regimes) but inconsistent with the colloquial misinterpretation (that dissipation drives or explains the emergence of structure). In our model, structures arise because they channel energy, irrespective of how much they dissipate. Under inverted loss, structured states dissipate *more* than unstructured states, yet persist identically.

## Model

### Grid Architecture
- 2D grid (10x40 by default), energy flows top to bottom
- Each cell has one of 4 states: empty, passive, active, replicating
- Energy is distributed across K=16 modes per cell
- Input: all energy concentrated in mode 0 (low entropy)
- Output: measured at bottom row across all modes

### Network Architecture (Layer 7)
- Random directed acyclic graph (DAG), 10 layers x 40 nodes
- Variable out-degree (1-5 edges per node)
- Energy flows along weighted edges (Dirichlet-distributed)
- Same cell types, same physics, different topology

### Cell Types

| Type | Standard Loss | Inverted Loss | Transforms? | Replicates? |
|------|--------------|---------------|-------------|-------------|
| Empty | 5% | 2% | No | No |
| Passive | 1% | 4% | No | No |
| Active | 2% | 8% | Yes | No |
| Replicating | 2% | 8% | Yes | Yes |

Under inverted loss, passive cells lose 4% — intermediate between empty (2%) and active (8%). Passive cells do not transform energy, so they neither form nor participate in structured pipelines, and results are insensitive to this value.

### Energy Transformation
Active and replicating cells mix energy across modes:
```
e_out = (1 - alpha) * e_in + alpha * (sum(e_in) / K) * ones(K)
```
This conserves total energy while increasing mode entropy. The mixing operation is applied *before* the loss: spectral redistribution occurs on the full input energy, then a fraction is lost. This means mixing entropy is nearly identical under standard and inverted loss (3.94 vs 3.93 grid; 3.93 vs 3.91 network) despite active cells losing 4x more energy under inversion.

### Energy Conservation
```
energy_in = energy_out + energy_lost  (verified per step)
```

## Eight Layers of Experiments

### Layer 1: Base Model (`run_all.py`)

Compares three regimes — memoryless, persistent, and persistent+branching — across parameter sweeps.

| Regime | Cum. Entropy | Throughput | Active Cells |
|--------|-------------|------------|-------------|
| Memoryless | 892 | 65.0% | 40 |
| Persistent | 1667 (+87%) | 76.9% | 155 |
| Persistent+Branching | 1938 (+117%) | 80.5% | 287 |

**Finding:** Persistence and replication create stable channel structures with higher cumulative entropy. Most of the advantage comes from persistence (memory) rather than replication. The branching regime achieves near-uniform output distribution (KL from uniform = 0.065, using 15.3 of 16 modes).

### Layer 2: Evolution Extension (`run_evolution.py`)

Adds heritable per-cell traits (transform_strength, split_factor, persist_threshold, mode_bias) that are copied with mutation during replication.

**Finding:** Traits evolve directionally toward more efficient configurations. Transform strength (alpha) evolves from 0.15 toward the optimum at ~0.124, balancing mixing effectiveness against energy redistribution.

### Layer 3: Paper Experiment Suite (`run_suite.py`)

Five experiments testing whether evolution discovers efficient entropy-producing configurations:

| Experiment | Question | Result |
|-----------|----------|--------|
| Evolving vs Best Frozen | Can evolution match exhaustive parameter search? | Within 4.4% despite no prior knowledge |
| Heterogeneous Environments | Does spatial variation favor evolution? | Gap widens; frozen search adapts per-environment |
| Changing Environments | Does temporal variation favor evolution? | Gap narrows to 1.7% under drift |
| Ablation Tests | Which ingredients matter? | Replication necessary; tradeoff is primary constraint |
| Long-Run Behavior | Does performance plateau? | Alpha evolves toward optimum (0.15 -> 0.124) |

**3 of 5 hypotheses supported.** Moderate evidence for evolutionary competitiveness, strongest under environmental change.

### Layer 4: Statistical Complexity (`run_complexity.py`)

Adds the statistical complexity metric (normalised entropy x disequilibrium).

**Finding:** The evolving system maintains intermediate complexity — neither frozen order nor random diffusion — consistent with a structured-but-not-trivial regime. This maps onto the "edge of chaos" (Langton 1990): the structured regime occupies the complexity peak between collapse and lock-in.

### Layer 5: Mechanism Identification

#### 5a. Local Reinforcement (`run_reinforcement.py`)

Tests whether explicit motif-level memory improves the system.

**Finding:** The baseline model already implements self-reinforcement through the throughput EMA — adding explicit reinforcement is redundant. The EMA is the memory mechanism. Without noise, explicit reinforcement causes complete lock-in (Lambda_B = 1.0).

#### 5b. Causal Decoupling (`run_decoupling.py`)

The decisive experiment. Systematically breaks the throughput -> persistence feedback loop:

| Condition | Persistence | Dissip. Rate | Mix. Ent. | Active |
|-----------|------------|--------------|-----------|--------|
| **Baseline** | 99.2 | 0.195 | 3.17 | 287 |
| Lifetime cap (50) | 24.1 (-76%) | 0.234 | 3.02 | 296 |
| Lifetime cap (150) | 57.4 (-42%) | 0.204 | 3.14 | 296 |
| Random override (5%) | 49.4 (-50%) | 0.224 | 3.07 | 304 |
| Random override (15%) | 24.1 (-76%) | 0.269 | 2.84 | 254 |
| Throughput-blind | 49.8 (-50%) | 0.223 | 3.07 | 303 |
| Half-blind | 66.5 (-33%) | 0.211 | 3.12 | 303 |
| **Anti-coupled** | **3.8 (-96%)** | **0.409** | **0.51** | **14** |

**Anti-coupling destroys everything.** Persistence drops 96%, active cells drop 95%. The catastrophic collapse under anti-coupling — not gradual degradation — establishes the throughput-persistence coupling as the causal mechanism. Mixing entropy varies only 11% across mild interventions despite 4-fold persistence variation, showing the coupling controls structure, not thermodynamic output.

### Layer 6: Flow-Persistence Number (`run_lambda.py`)

Operationalises the throughput-persistence feedback as a dimensionless control parameter:

```
Lambda_B = (mean_tp_survived - mean_tp_disrupted) / mean_tp_all
         = Cov(w,z) / [Var(w) * z_bar]
         = beta_{z<-w} / z_bar
```

This is the normalised Price covariance — a dimensionless selection differential.

Key results (Spearman rank correlations with persistence):

| Predictor | Grid rho | Network rho |
|-----------|----------|-------------|
| **Lambda_B** (selection gradient) | **+0.93** | +0.75 |
| **Lambda_A** (log survival balance) | **+0.93** | **+0.93** |
| Dissipation rate | -0.73* | -0.75* |
| Mixing entropy | +0.69 | +0.39 |
| Complexity | -0.30 | --- |

*Parameterisation-dependent: these values reverse to rho = +0.60 (grid) and +0.80 (network) under inverted loss. The Lambda measures are invariant across parameterisations.

Lambda_B separates three regimes:
- **Collapse** (memoryless, anti-coupled): Lambda_B ~ 0.00
- **Structured** (baseline, decoupled variants): 0 < Lambda_B < 1
- **Lock-in** (no noise): Lambda_B = 1.00

Lambda_A (log survival balance) is the more robust empirical predictor (rho = 0.93 in both architectures). Lambda_B is preferred for its formal Price equation interpretation but is weaker in the network (0.75), likely because the DAG's heterogeneous connectivity creates a wider throughput distribution that makes mean-normalised differences less discriminating.

### Layer 7: Network Flow Model (`run_network.py`)

Universality test on a structurally different architecture: random DAG with variable connectivity (1-5 edges per node), Dirichlet-distributed edge weights.

| Condition | Lambda_B | Persistence | Dissip. | Active |
|-----------|----------|-------------|---------|--------|
| Memoryless | 0.002 | 0.0 | 0.328 | 40 |
| Persistent | 0.644 | 96.9 | 0.213 | 111 |
| Pers.+Branching | 0.770 | 87.9 | 0.177 | 228 |
| Anti-coupled | -1.667 | 70.7 | 0.352 | 130 |
| Throughput-blind | 0.021 | 11.9 | 0.300 | 142 |
| Lifetime cap (50) | 0.529 | 21.7 | 0.214 | 223 |
| No noise | 0.870 | 850.7 | 0.169 | 236 |

**Same three-regime structure replicates.** The key difference: anti-coupling in the network produces *parasitic persistence* (Lambda_B = -1.67, persistence = 70.7) rather than the grid's total collapse (persistence = 3.8). Non-functional nodes survive because the DAG's variable connectivity lets them occupy positions that receive energy passively. Lambda_B correctly identifies this as anti-selection (negative value).

### Layer 8: Inverted Loss Test (`run_inverted_loss.py`)

The critical experiment: does the coupling produce structure even when active cells dissipate MORE than empty cells?

- **Standard loss (Parameterisation A)**: active=2%, empty=5%. Filling the grid with active cells mechanically reduces aggregate dissipation.
- **Inverted loss (Parameterisation B)**: active=8%, empty=2%. Filling the grid with active cells mechanically *increases* aggregate dissipation.
- Neither parameterisation is privileged; the coupling is the invariant, and the dissipation sign is the free parameter.

| Loss Regime | Condition | Lambda_B | Persistence | Dissip. | Mix. Ent. | Active |
|-------------|-----------|----------|-------------|---------|-----------|--------|
| Standard | Baseline | 0.379 | 98.8 | 0.195 | 3.94 | 287 |
| Standard | Memoryless | 0.000 | 0.0 | 0.350 | 1.78 | 40 |
| Standard | Anti-coupled | 0.007 | 3.8 | 0.409 | 0.85 | 14 |
| Inverted | Baseline | 0.389 | 99.6 | 0.521 | 3.93 | 283 |
| Inverted | Memoryless | 0.000 | 0.0 | 0.287 | 1.70 | 40 |
| Inverted | Anti-coupled | 0.008 | 3.8 | 0.231 | 0.79 | 14 |

**Three findings:**

1. **The coupling is dissipation-agnostic.** Lambda_B is identical (0.379 vs 0.389) regardless of whether the structured regime dissipates more or less than the unstructured one. The coupling selects for throughput, not for dissipation.

2. **The dissipation sign is a parameterisation artefact.** Standard: structured baseline dissipates 0.195, memoryless dissipates 0.350 (Spearman rho = -0.50 between dissipation and persistence). Inverted: baseline dissipates 0.521, memoryless dissipates 0.287 (rho = +0.60 to +0.80). The sign of the dissipation-structure relationship is entirely determined by whether active or empty cells have higher loss rates.

3. **Anti-coupling still destroys structure.** Under inverted loss, anti-coupling reduces grid persistence from 99.6 to 3.8 (-96%), exactly as under standard loss. The causal mechanism is unchanged.

Replicates in both grid and network architectures.

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

# Layer 7: Network flow model (~5 min)
python run_network.py

# Layer 8: Inverted loss test (~8 min)
python run_inverted_loss.py
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

### Layer 7: Network Flow Model
- `network_output/` — results CSV, summary CSV, report
- `network_plots/` — time series, phase diagram, condition comparison (3 plots)

### Layer 8: Inverted Loss Test
- `inverted_output/` — results CSV, summary CSV, report
- `inverted_plots/` — comparison bars, dissipation vs structure scatter (2 plots)

## The Argument in Summary

1. Persistent channel structures produce more cumulative entropy than memoryless ones (Layer 1).
2. Heritable traits evolve directionally toward efficient configurations (Layers 2-3).
3. The system occupies an intermediate complexity regime (Layer 4).
4. The throughput EMA already functions as local memory of success — explicit reinforcement is redundant (Layer 5a).
5. **Breaking the throughput-persistence coupling destroys structured regimes** (Layer 5b).
6. A dimensionless flow-persistence number (normalised Price covariance) predicts regime membership better than any thermodynamic observable (Layer 6).
7. **The same regime structure replicates in a structurally different architecture** (Layer 7).
8. **The coupling operates identically whether structure increases or decreases dissipation** (Layer 8).

The key empirical finding:

> **Dissipation is irrelevant to the mechanism.** Under standard loss rates, structure reduces dissipation. Under inverted loss rates, structure *increases* dissipation. In both cases, the coupling produces identical structured regimes (Lambda_B ~ 0.38, persistence ~ 99). The coupling selects for throughput, and throughput is agnostic about waste. The mechanism that produces structure is dissipation-agnostic: even if MEPP or dissipative adaptation are correct as thermodynamic principles, they do not explain why structure arises in this class of systems.

The formal finding:

> **Lambda_B = Cov(w,z) / [Var(w) * z_bar]** — the flow-persistence number is a normalised Price covariance, connecting the throughput-persistence coupling directly to quantitative selection theory. This is an algebraic identity: the throughput-persistence coupling *is* selection, in the precise statistical sense that Price defined.

## References

- Bak, P., Tang, C., & Wiesenfeld, K. (1987). Self-organized criticality. *Physical Review Letters*, 59(4), 381-384.
- Bejan, A. (1997). Constructal-theory network of conducting paths for cooling a heat generating volume. *International Journal of Heat and Mass Transfer*, 40(4), 799-816.
- Dewar, R. (2003). Information theory explanation of the fluctuation theorem, maximum entropy production and self-organized criticality in non-equilibrium stationary states. *Journal of Physics A*, 36(3), 631.
- England, J. L. (2013). Statistical physics of self-replication. *Journal of Chemical Physics*, 139(12), 121923.
- Kauffman, S. A. (2000). *Investigations*. Oxford University Press.
- Langton, C. G. (1990). Computation at the edge of chaos: phase transitions and emergent computation. *Physica D*, 42(1-3), 12-37.
- Lopez-Ruiz, R., Mancini, H. L., & Calbet, X. (1995). A statistical measure of complexity. *Physics Letters A*, 209(5-6), 321-326.
- Martyushev, L. M. & Seleznev, V. D. (2006). Maximum entropy production principle in physics, chemistry and biology. *Physics Reports*, 426(1), 1-45.
- Price, G. R. (1970). Selection and covariance. *Nature*, 227, 520-521.
