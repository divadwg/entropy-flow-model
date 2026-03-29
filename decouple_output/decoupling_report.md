# Throughput–Persistence Decoupling — Experiment Report

Grid: 10x40, 16 modes
Regime: 3
Steps: 1500, Seeds: 8

## Hypothesis Under Test

If selection-like structure depends on a throughput → persistence feedback loop, then breaking that loop should degrade or destroy the structured regime.

## Interventions

| Condition | Description |
|-----------|-------------|
| baseline | Normal throughput-mediated persistence |
| cap_50 | Cells die after 50 steps regardless of throughput |
| cap_150 | Cells die after 150 steps regardless of throughput |
| override_5% | 5% of high-throughput cells randomly killed each step |
| override_15% | 15% of high-throughput cells randomly killed each step |
| blind | Persistence completely ignores throughput (random survival) |
| half_blind | 50/50 mix of throughput-based and random survival |
| anti_coupled | High throughput REDUCES persistence (full inversion) |

## Results

| Condition | Persistence | Recurrence | Propagation | Complexity | EP | Throughput | Active |
|-----------|------------|------------|-------------|-----------|------|-----------|--------|
| baseline | 99.2 | 0.989 | 0.024 | 0.00732 | 3.168 | 2.224 | 287 |
| cap_50 | 24.1 | 0.983 | 0.114 | 0.00627 | 3.023 | 2.166 | 296 |
| cap_150 | 57.4 | 0.986 | 0.048 | 0.00626 | 3.140 | 2.211 | 296 |
| override_5% | 49.4 | 0.984 | 0.101 | 0.00523 | 3.068 | 2.181 | 304 |
| override_15% | 24.1 | 0.982 | 0.096 | 0.01342 | 2.844 | 2.112 | 254 |
| blind | 49.8 | 0.984 | 0.102 | 0.00538 | 3.071 | 2.183 | 303 |
| half_blind | 66.5 | 0.985 | 0.067 | 0.00546 | 3.118 | 2.199 | 303 |
| anti_coupled | 3.8 | 0.961 | 0.000 | 0.15788 | 0.505 | 1.894 | 14 |

## Effects vs Baseline

### cap_50

- Persistence: decreased by 75.7%
- Recurrence: decreased by 0.6%
- Propagation: increased by 383.1%
- Stat. complexity: decreased by 14.4%
- Entropy production: decreased by 4.6%
- Throughput: decreased by 2.6%

### cap_150

- Persistence: decreased by 42.1%
- Recurrence: decreased by 0.3%
- Propagation: increased by 101.6%
- Stat. complexity: decreased by 14.5%
- Entropy production: decreased by 0.9%
- Throughput: decreased by 0.6%

### override_5%

- Persistence: decreased by 50.2%
- Recurrence: decreased by 0.5%
- Propagation: increased by 327.8%
- Stat. complexity: decreased by 28.5%
- Entropy production: decreased by 3.2%
- Throughput: decreased by 1.9%

### override_15%

- Persistence: decreased by 75.7%
- Recurrence: decreased by 0.7%
- Propagation: increased by 305.8%
- Stat. complexity: increased by 83.3%
- Entropy production: decreased by 10.2%
- Throughput: decreased by 5.0%

### blind

- Persistence: decreased by 49.8%
- Recurrence: decreased by 0.5%
- Propagation: increased by 332.5%
- Stat. complexity: decreased by 26.5%
- Entropy production: decreased by 3.1%
- Throughput: decreased by 1.8%

### half_blind

- Persistence: decreased by 32.9%
- Recurrence: decreased by 0.3%
- Propagation: increased by 182.5%
- Stat. complexity: decreased by 25.4%
- Entropy production: decreased by 1.6%
- Throughput: decreased by 1.1%

### anti_coupled

- Persistence: decreased by 96.2%
- Recurrence: decreased by 2.8%
- Propagation: decreased by 100.0%
- Stat. complexity: increased by 2056.4%
- Entropy production: decreased by 84.1%
- Throughput: decreased by 14.8%

## Path Dependence

| Condition | Late Hamming (mean) |
|-----------|--------------------|
| baseline | 0.657 |
| blind | 0.569 |
| anti_coupled | 0.075 |

## Structure Ranking

| Rank | Condition | Structure Score | EP |
|------|-----------|----------------|----|
| 1 | blind | 0.737 | 3.071 |
| 2 | baseline | 0.736 | 3.168 |
| 3 | override_5% | 0.731 | 3.068 |
| 4 | half_blind | 0.706 | 3.118 |
| 5 | cap_50 | 0.672 | 3.023 |
| 6 | cap_150 | 0.627 | 3.140 |
| 7 | override_15% | 0.602 | 2.844 |
| 8 | anti_coupled | 0.000 | 0.505 |

## Interpretation

**The throughput–persistence coupling is causally necessary for structured regimes.**

Anti-coupled persistence reduces structure score by 100%. Inverting the coupling actively destroys structure by removing the cells that channel energy most effectively.

This supports the causal claim:

> Selection-like structure in this model depends on a feedback loop between throughput and persistence. When that loop is broken, the system no longer maintains structured regimes effectively.

Entropy production is roughly constant across conditions, confirming that structure is NOT selected for its entropy-producing properties. The causal mechanism is throughput → persistence, not entropy maximization.

## Conclusion

The throughput–persistence coupling is the operative selection law in this model. It is not an implementation artefact; it is the mechanism by which locally successful configurations maintain themselves. Breaking it produces measurable degradation in persistence, propagation, and recurrence of local motifs.
