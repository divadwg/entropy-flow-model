"""
Energy flow dynamics and cell state update rules.

Contains both the original fixed-parameter functions (used by run_all.py)
and new trait-aware functions (used by run_evolution.py).
"""
import numpy as np
from .grid import (
    EMPTY, PASSIVE, ACTIVE, REPLICATING,
    TRAIT_ALPHA, TRAIT_SPLIT, TRAIT_THRESHOLD, TRAIT_BIAS,
    N_TRAITS, TRAIT_BOUNDS, TRAIT_DEFAULTS,
)


# ── Original fixed-parameter functions ────────────────────────────

def flow_energy(grid, e_in_per_cell, params):
    """
    Propagate energy top-to-bottom using global transform_strength.
    Used by the original 3-regime experiment (run_all.py).
    """
    H, W, K = grid.height, grid.width, grid.n_modes
    alpha = params["transform_strength"]
    spread = params["lateral_spread"]
    w_center = 1.0 - 2.0 * spread

    loss = np.full((H, W), params["loss_rate_empty"], dtype=np.float64)
    loss[grid.states == PASSIVE] = params["loss_rate_passive"]
    loss[(grid.states == ACTIVE) | (grid.states == REPLICATING)] = params["loss_rate_active"]

    is_transformer = (grid.states == ACTIVE) | (grid.states == REPLICATING)

    current = np.zeros((W, K), dtype=np.float64)
    current[:, 0] = e_in_per_cell

    cell_throughput = np.zeros((H, W), dtype=np.float64)
    total_lost = 0.0

    for r in range(H):
        totals = current.sum(axis=1, keepdims=True)
        uniform = totals / K
        mask = is_transformer[r][:, np.newaxis]
        processed = np.where(mask, (1 - alpha) * current + alpha * uniform, current)

        cell_loss = loss[r][:, np.newaxis]
        lost_here = processed * cell_loss
        total_lost += lost_here.sum()
        processed -= lost_here

        # Throughput = outgoing energy (after loss)
        cell_throughput[r] = processed.sum(axis=1)

        if r < H - 1:
            next_row = w_center * processed
            next_row[1:] += spread * processed[:-1]
            next_row[:-1] += spread * processed[1:]
            total_lost += spread * processed[0].sum()
            total_lost += spread * processed[-1].sum()
            current = next_row
        else:
            output = processed

    energy_in = e_in_per_cell * W
    energy_out = float(output.sum())
    return output, cell_throughput, {
        "energy_in": energy_in, "energy_out": energy_out, "energy_lost": total_lost,
    }


def update_states(grid, cell_throughput, params, regime):
    """
    Original state update for regimes 1-3 (no trait evolution).
    Used by run_all.py.
    """
    H, W = grid.height, grid.width
    rng = grid.rng

    if regime == 1:
        r = rng.random((H, W))
        grid.states[:] = EMPTY
        grid.states[r < params["active_fraction"] + params["passive_fraction"]] = PASSIVE
        grid.states[r < params["active_fraction"]] = ACTIVE
        grid.throughput[:] = 0.0
        grid.age[:] = 0
        return

    decay = params["throughput_decay"]
    grid.throughput = decay * grid.throughput + (1.0 - decay) * cell_throughput
    grid.age += 1

    high_tp = grid.throughput > params["persistence_threshold"]

    can_decay = (grid.states > EMPTY) & ~high_tp
    decay_prob = 1.0 - params["persistence_strength"]
    decayed = can_decay & (rng.random((H, W)) < decay_prob)
    grid.states[decayed] = EMPTY
    grid.age[decayed] = 0
    grid.throughput[decayed] = 0.0

    if regime == 3:
        repl_prob = params["replication_prob"]
        can_repl = high_tp & (grid.states >= ACTIVE)
        repl_cells = can_repl & (rng.random((H, W)) < repl_prob)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ys, xs = np.where(repl_cells)
        for y, x in zip(ys, xs):
            dy, dx = directions[rng.integers(4)]
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and grid.states[ny, nx] == EMPTY:
                grid.states[ny, nx] = REPLICATING
                grid.age[ny, nx] = 0

    create_rate = params["spontaneous_create_rate"]
    empty = grid.states == EMPTY
    created = empty & (rng.random((H, W)) < create_rate)
    n_created = int(created.sum())
    if n_created > 0:
        grid.states[created] = rng.choice(
            np.array([PASSIVE, ACTIVE], dtype=np.int8), size=n_created
        )
        grid.age[created] = 0

    mut_rate = params["mutation_rate"]
    mutated = rng.random((H, W)) < mut_rate
    n_mut = int(mutated.sum())
    if n_mut > 0:
        max_state = 4 if regime == 3 else 3
        grid.states[mutated] = rng.integers(0, max_state, size=n_mut).astype(np.int8)
        grid.age[mutated] = 0


# ── Trait-aware functions (evolution experiment) ──────────────────

def _build_row_targets(grid, r, K):
    """Build per-cell mixing targets for one row based on split_factor and mode_bias.

    Returns (W, K) array where each row is the target distribution for that cell.
    Cells with split_factor >= K get uniform targets. Others get energy concentrated
    in ceil(split_factor) contiguous modes starting at mode_bias * K.
    """
    W = grid.width
    targets = np.full((W, K), 1.0 / K, dtype=np.float64)

    active = (grid.states[r] == ACTIVE) | (grid.states[r] == REPLICATING)
    if not active.any() or grid.traits is None:
        return targets

    sf = grid.traits[r, :, TRAIT_SPLIT]
    mb = grid.traits[r, :, TRAIT_BIAS]

    for c in np.where(active)[0]:
        n = max(1, min(K, int(round(sf[c]))))
        if n >= K:
            continue  # already uniform
        center = int(mb[c] * K) % K
        t = np.zeros(K)
        for i in range(n):
            t[(center + i) % K] = 1.0 / n
        targets[c] = t

    return targets


def flow_energy_with_traits(grid, e_in_per_cell, params):
    """
    Energy flow using per-cell heritable traits.

    Each active cell uses its own transform_strength (alpha) and mixing target
    (from split_factor and mode_bias). Loss rate increases with alpha via
    transform_cost, creating a tradeoff between mixing and efficiency.
    """
    H, W, K = grid.height, grid.width, grid.n_modes
    spread = params["lateral_spread"]
    w_center = 1.0 - 2.0 * spread
    transform_cost = params.get("transform_cost", 0.0)
    base_active_loss = params["loss_rate_active"]

    # Per-cell loss rates
    loss = np.full((H, W), params["loss_rate_empty"], dtype=np.float64)
    loss[grid.states == PASSIVE] = params["loss_rate_passive"]

    active_mask = (grid.states == ACTIVE) | (grid.states == REPLICATING)
    if grid.traits is not None:
        # Loss = base + cost * alpha (transformation has a thermodynamic cost)
        # transform_cost can be scalar or (H, W) array for spatial heterogeneity
        tc = transform_cost
        if np.ndim(tc) == 0:
            loss[active_mask] = base_active_loss + float(tc) * grid.traits[active_mask, TRAIT_ALPHA]
        else:
            loss[active_mask] = base_active_loss + np.asarray(tc)[active_mask] * grid.traits[active_mask, TRAIT_ALPHA]
    else:
        loss[active_mask] = base_active_loss

    # e_in_per_cell can be scalar (uniform) or (W,) array (spatially varying)
    current = np.zeros((W, K), dtype=np.float64)
    current[:, 0] = e_in_per_cell

    cell_throughput = np.zeros((H, W), dtype=np.float64)
    total_lost = 0.0

    for r in range(H):
        totals = current.sum(axis=1, keepdims=True)  # (W, 1)
        is_active_row = active_mask[r]

        if grid.traits is not None and is_active_row.any():
            row_alpha = grid.traits[r, :, TRAIT_ALPHA][:, np.newaxis]  # (W, 1)
            targets = _build_row_targets(grid, r, K)  # (W, K)
            target_scaled = targets * totals  # (W, K)
            mask = is_active_row[:, np.newaxis]
            processed = np.where(mask,
                                 (1 - row_alpha) * current + row_alpha * target_scaled,
                                 current)
        else:
            alpha = params.get("transform_strength", 0.3)
            uniform = totals / K
            mask = is_active_row[:, np.newaxis]
            processed = np.where(mask, (1 - alpha) * current + alpha * uniform, current)

        cell_loss = loss[r][:, np.newaxis]
        lost_here = processed * cell_loss
        total_lost += lost_here.sum()
        processed -= lost_here

        # Throughput = outgoing energy (after loss), so cell's own alpha affects its fitness
        cell_throughput[r] = processed.sum(axis=1)

        if r < H - 1:
            next_row = w_center * processed
            next_row[1:] += spread * processed[:-1]
            next_row[:-1] += spread * processed[1:]
            total_lost += spread * processed[0].sum()
            total_lost += spread * processed[-1].sum()
            current = next_row
        else:
            output = processed

    energy_in = float(e_in_per_cell * W) if np.ndim(e_in_per_cell) == 0 else float(np.sum(e_in_per_cell))
    energy_out = float(output.sum())
    return output, cell_throughput, {
        "energy_in": energy_in, "energy_out": energy_out, "energy_lost": total_lost,
    }


def update_states_evolving(grid, cell_throughput, params, evolving_traits=None):
    """
    State update with heritable trait replication and mutation.

    When replication_prob > 0, active cells with high throughput can copy
    into empty neighbors, inheriting traits with Gaussian mutation noise.
    When trait_mutation_std == 0, traits are copied exactly (fixed regime).

    evolving_traits: list of trait indices that can mutate, or None for all.
        Non-evolving traits are reset to defaults in offspring.
    """
    H, W = grid.height, grid.width
    rng = grid.rng

    # Throughput EMA
    decay = params["throughput_decay"]
    grid.throughput = decay * grid.throughput + (1.0 - decay) * cell_throughput
    grid.age += 1

    # Persistence: use row-relative throughput to remove positional bias
    # A cell persists if its throughput > threshold * row_mean
    is_active = (grid.states == ACTIVE) | (grid.states == REPLICATING)
    global_threshold = params.get("persistence_threshold", 0.5)

    row_means = grid.throughput.mean(axis=1, keepdims=True)
    row_means = np.maximum(row_means, 1e-12)
    relative_tp = grid.throughput / row_means

    if grid.traits is not None:
        thresholds = np.full((H, W), global_threshold, dtype=np.float64)
        thresholds[is_active] = grid.traits[is_active, TRAIT_THRESHOLD]
        high_tp = relative_tp > thresholds
    else:
        high_tp = relative_tp > global_threshold

    # Decay low-throughput channels
    can_decay = (grid.states > EMPTY) & ~high_tp
    decay_prob = 1.0 - params["persistence_strength"]
    decayed = can_decay & (rng.random((H, W)) < decay_prob)
    grid.states[decayed] = EMPTY
    grid.age[decayed] = 0
    grid.throughput[decayed] = 0.0
    if grid.traits is not None:
        grid.traits[decayed] = 0.0
        grid.lineage[decayed] = -1

    # Replication with trait inheritance — proportional to throughput
    repl_prob = params.get("replication_prob", 0.0)
    mutation_std = params.get("trait_mutation_std", 0.0)

    if repl_prob > 0:
        can_repl = high_tp & (grid.states >= ACTIVE)
        # Proportional replication: higher relative throughput → higher replication rate
        rel_tp_active = relative_tp * can_repl
        max_rel = rel_tp_active.max() if rel_tp_active.any() else 1.0
        if max_rel > 0:
            scaled_prob = repl_prob * (relative_tp / max_rel)
        else:
            scaled_prob = np.full((H, W), repl_prob)
        repl_cells = can_repl & (rng.random((H, W)) < scaled_prob)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ys, xs = np.where(repl_cells)
        for y, x in zip(ys, xs):
            dy, dx = directions[rng.integers(4)]
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and grid.states[ny, nx] == EMPTY:
                grid.states[ny, nx] = REPLICATING
                grid.age[ny, nx] = 0

                if grid.traits is not None:
                    child = grid.traits[y, x].copy()
                    for t in range(N_TRAITS):
                        if mutation_std > 0 and (evolving_traits is None or t in evolving_traits):
                            child[t] += rng.normal(0, mutation_std)
                        elif evolving_traits is not None and t not in evolving_traits:
                            child[t] = TRAIT_DEFAULTS[t]
                        lo, hi = TRAIT_BOUNDS[t]
                        child[t] = np.clip(child[t], lo, hi)
                    grid.traits[ny, nx] = child
                    grid.lineage[ny, nx] = grid.lineage[y, x]

    # Spontaneous creation
    create_rate = params.get("spontaneous_create_rate", 0.005)
    empty = grid.states == EMPTY
    created = empty & (rng.random((H, W)) < create_rate)
    n_created = int(created.sum())
    if n_created > 0:
        grid.states[created] = rng.choice(
            np.array([PASSIVE, ACTIVE], dtype=np.int8), size=n_created
        )
        grid.age[created] = 0

        if grid.traits is not None:
            ys, xs = np.where(created)
            for y, x in zip(ys, xs):
                if grid.states[y, x] >= ACTIVE:
                    # Spontaneous cells always start with defaults;
                    # only inherited traits evolve via mutation during replication
                    for t in range(N_TRAITS):
                        grid.traits[y, x, t] = TRAIT_DEFAULTS[t]
                    grid.lineage[y, x] = grid._next_lineage_id
                    grid._next_lineage_id += 1

    # State mutation
    mut_rate = params.get("mutation_rate", 0.01)
    mutated = rng.random((H, W)) < mut_rate
    n_mut = int(mutated.sum())
    if n_mut > 0:
        grid.states[mutated] = rng.integers(0, 4, size=n_mut).astype(np.int8)
        grid.age[mutated] = 0
