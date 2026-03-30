# Inverted Loss Test Report

## Purpose

Test whether the throughput-persistence coupling produces structure even when
active cells dissipate MORE than empty cells.

- **Standard**: active=2%, passive=1%, empty=5% loss
- **Inverted**: active=8%, passive=4%, empty=2% loss

If structure persists under inverted loss, the coupling mechanism is
independent of whether structure increases or decreases dissipation.

## Summary Table

| Condition | Lambda_B | Persistence | Dissip. | Entropy | Active |
|-----------|----------|-------------|---------|---------|--------|
| grid_std_baseline | 0.379 | 98.8 | 0.195 | 3.935 | 287 |
| grid_std_memoryless | 0.000 | 0.0 | 0.350 | 1.782 | 40 |
| grid_std_anti | 0.007 | 3.8 | 0.409 | 0.850 | 14 |
| grid_inv_baseline | 0.389 | 99.6 | 0.521 | 3.928 | 283 |
| grid_inv_memoryless | 0.000 | 0.0 | 0.287 | 1.699 | 40 |
| grid_inv_anti | 0.008 | 3.8 | 0.231 | 0.792 | 14 |
| grid_inv_blind | 0.020 | 49.5 | 0.524 | 3.948 | 304 |
| net_std_baseline | 0.770 | 87.9 | 0.177 | 3.926 | 228 |
| net_std_memoryless | 0.002 | 0.0 | 0.328 | 1.779 | 40 |
| net_std_anti | -1.667 | 70.7 | 0.352 | 2.735 | 130 |
| net_inv_baseline | 0.760 | 84.6 | 0.494 | 3.911 | 215 |
| net_inv_memoryless | 0.002 | 0.0 | 0.263 | 1.698 | 40 |
| net_inv_anti | -1.697 | 71.2 | 0.281 | 2.567 | 126 |
| net_inv_blind | 0.015 | 11.9 | 0.378 | 3.546 | 139 |

## Key Comparisons


### GRID Model

**Baseline comparison:**
- Standard: persistence=98.8, Lambda_B=0.379, dissip=0.195
- Inverted: persistence=99.6, Lambda_B=0.389, dissip=0.521

**STRUCTURE PERSISTS** under inverted loss. Lambda_B = 0.389 (positive selection gradient). Dissipation INCREASES with structure under inverted loss (0.521 vs 0.287 memoryless), confirming the coupling operates independently of the dissipation sign.

**Anti-coupling under inverted loss:**
- Baseline: persistence=99.6, active=283
- Anti-coupled: persistence=3.8, active=14
- Reduction: 96%


### NETWORK Model

**Baseline comparison:**
- Standard: persistence=87.9, Lambda_B=0.770, dissip=0.177
- Inverted: persistence=84.6, Lambda_B=0.760, dissip=0.494

**STRUCTURE PERSISTS** under inverted loss. Lambda_B = 0.760 (positive selection gradient). Dissipation INCREASES with structure under inverted loss (0.494 vs 0.263 memoryless), confirming the coupling operates independently of the dissipation sign.

**Anti-coupling under inverted loss:**
- Baseline: persistence=84.6, active=215
- Anti-coupled: persistence=71.2, active=126
- Reduction: 16%


## Dissipation Sign Analysis

Under standard loss: more structure → lower aggregate dissipation (because active cells lose 2% vs empty cells' 5%).

Under inverted loss: more structure → [test result] aggregate dissipation (because active cells lose 8% vs empty cells' 2%).

GRID standard: dissipation vs persistence rho = -0.500
GRID inverted: dissipation vs persistence rho = 0.600
NETWORK standard: dissipation vs persistence rho = -0.500
NETWORK inverted: dissipation vs persistence rho = 0.800

## Conclusion

**The throughput-persistence coupling produces structure regardless of the
dissipation sign.** Under inverted loss rates (active cells lose 4x more
than empty cells), the coupling still produces persistent structure with
positive Lambda_B. This definitively isolates the coupling mechanism from
the dissipation direction.

The implication: the structured regime does not arise *because* it reduces
dissipation (standard loss) or *because* it increases dissipation (inverted
loss). It arises because the throughput-persistence coupling selects for
cells that channel energy, and channeling requires surviving long enough to
be in the path of flow. The sign of the dissipation change is a consequence
of the loss-rate parameterisation, not a driver of the mechanism.
