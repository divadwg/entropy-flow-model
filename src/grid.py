"""
Grid state management for the entropy flow model.

Each cell has a state (empty/passive/active/replicating) and optional
heritable traits that define its transformation behavior. When traits
are enabled, each active cell carries parameters that are copied (with
mutation) during replication.
"""
import numpy as np

EMPTY = 0
PASSIVE = 1
ACTIVE = 2
REPLICATING = 3
STATE_NAMES = ["empty", "passive", "active", "replicating"]

# Heritable trait indices
TRAIT_ALPHA = 0       # transform_strength: mixing intensity per cell
TRAIT_SPLIT = 1       # split_factor: number of output modes targeted
TRAIT_THRESHOLD = 2   # persist_threshold: throughput needed to survive
TRAIT_BIAS = 3        # mode_bias: which modes to target (0-1 maps to mode index)
N_TRAITS = 4
TRAIT_NAMES = ["transform_strength", "split_factor", "persist_threshold", "mode_bias"]
TRAIT_BOUNDS = [
    (0.05, 0.95),   # transform_strength
    (1.0, 16.0),    # split_factor (1 mode to K modes)
    (0.1, 2.0),     # persist_threshold
    (0.0, 1.0),     # mode_bias
]
TRAIT_DEFAULTS = [0.15, 16.0, 0.8, 0.0]


class Grid:
    """2D grid with cell states, throughput history, and optional heritable traits."""

    def __init__(self, height, width, n_modes, rng=None):
        self.height = height
        self.width = width
        self.n_modes = n_modes
        self.rng = rng or np.random.default_rng()
        self.states = np.zeros((height, width), dtype=np.int8)
        self.throughput = np.zeros((height, width), dtype=np.float64)
        self.age = np.zeros((height, width), dtype=np.int32)
        self.traits = None       # (H, W, N_TRAITS) when enabled
        self.lineage = None      # (H, W) lineage IDs
        self._next_lineage_id = 0

    def initialize(self, active_frac=0.1, passive_frac=0.2):
        """Set random initial cell states."""
        r = self.rng.random((self.height, self.width))
        self.states[:] = EMPTY
        self.states[r < active_frac + passive_frac] = PASSIVE
        self.states[r < active_frac] = ACTIVE
        self.throughput[:] = 0.0
        self.age[:] = 0

    def enable_traits(self, randomize=True):
        """Enable per-cell heritable traits.

        If randomize=True, active cells get random traits and unique lineages.
        If randomize=False, all cells get default trait values.
        """
        H, W = self.height, self.width
        self.traits = np.zeros((H, W, N_TRAITS), dtype=np.float64)
        self.lineage = np.full((H, W), -1, dtype=np.int32)

        for t in range(N_TRAITS):
            self.traits[:, :, t] = TRAIT_DEFAULTS[t]

        active = (self.states == ACTIVE) | (self.states == REPLICATING)
        if randomize and active.any():
            ys, xs = np.where(active)
            for y, x in zip(ys, xs):
                for t in range(N_TRAITS):
                    lo, hi = TRAIT_BOUNDS[t]
                    self.traits[y, x, t] = self.rng.uniform(lo, hi)
                self.lineage[y, x] = self._next_lineage_id
                self._next_lineage_id += 1
        elif not randomize and active.any():
            ys, xs = np.where(active)
            for y, x in zip(ys, xs):
                self.lineage[y, x] = self._next_lineage_id
                self._next_lineage_id += 1

    def active_area(self):
        """Count cells in active or replicating state."""
        return int(((self.states == ACTIVE) | (self.states == REPLICATING)).sum())

    def mean_channel_age(self):
        """Mean age of non-empty cells."""
        mask = self.states > EMPTY
        return float(self.age[mask].mean()) if mask.any() else 0.0

    def count_by_state(self):
        return {STATE_NAMES[i]: int((self.states == i).sum()) for i in range(4)}

    def trait_statistics(self):
        """Mean and std of each trait across active cells."""
        active = (self.states == ACTIVE) | (self.states == REPLICATING)
        if not active.any() or self.traits is None:
            return {name: (0.0, 0.0) for name in TRAIT_NAMES}
        vals = self.traits[active]
        return {
            TRAIT_NAMES[i]: (float(vals[:, i].mean()), float(vals[:, i].std()))
            for i in range(N_TRAITS)
        }

    def trait_values(self):
        """Return (N_active, N_TRAITS) array of trait values for active cells."""
        active = (self.states == ACTIVE) | (self.states == REPLICATING)
        if not active.any() or self.traits is None:
            return np.empty((0, N_TRAITS))
        return self.traits[active].copy()

    def lineage_counts(self):
        """Count cells per lineage ID. Returns dict {lineage_id: count}."""
        if self.lineage is None:
            return {}
        active = (self.states == ACTIVE) | (self.states == REPLICATING)
        ids = self.lineage[active]
        ids = ids[ids >= 0]
        if len(ids) == 0:
            return {}
        unique, counts = np.unique(ids, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


# ── Local pattern reinforcement ──────────────────────────────────

N_STATES = 4
MOTIF_SIZE = N_STATES ** 5  # cross-neighborhood: center + 4 cardinal neighbors


def encode_motif(center, up, down, left, right):
    """Encode a cross-neighborhood of cell states as an integer in [0, 1024)."""
    return int(center + N_STATES * (up + N_STATES * (down + N_STATES * (left + N_STATES * right))))


def extract_motifs(states):
    """Extract the cross-neighborhood motif ID for every cell.

    Returns (H, W) int array of motif IDs. Boundary cells use EMPTY for
    out-of-bounds neighbors.
    """
    H, W = states.shape
    motifs = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            c = int(states[y, x])
            u = int(states[y - 1, x]) if y > 0 else EMPTY
            d = int(states[y + 1, x]) if y < H - 1 else EMPTY
            l = int(states[y, x - 1]) if x > 0 else EMPTY
            r = int(states[y, x + 1]) if x < W - 1 else EMPTY
            motifs[y, x] = encode_motif(c, u, d, l, r)
    return motifs


class ReinforcementMap:
    """Tracks reinforcement scores for local state-neighborhood motifs.

    After each step, motifs whose center cell persisted get a small increment.
    All scores decay each step. During state updates, the reinforcement score
    for a cell's current motif slightly biases its survival probability:

        survival_boost = epsilon * score / (1 + score)

    This is bounded in [0, epsilon), preventing runaway feedback.
    """

    def __init__(self, epsilon=0.1, decay_rate=0.99, max_score=10.0):
        self.scores = np.zeros(MOTIF_SIZE, dtype=np.float64)
        self.epsilon = epsilon        # strength of reinforcement bias
        self.decay_rate = decay_rate  # per-step multiplicative decay
        self.max_score = max_score    # clip to prevent overflow

    def get_boost(self, motif_ids):
        """Return survival boost array for given motif IDs.

        boost = epsilon * score / (1 + score), bounded in [0, epsilon).
        """
        scores = self.scores[motif_ids]
        return self.epsilon * scores / (1.0 + scores)

    def reinforce(self, motif_ids, mask):
        """Increment scores for motifs at positions where mask is True."""
        ids = motif_ids[mask]
        if len(ids) == 0:
            return
        # Accumulate counts per motif
        np.add.at(self.scores, ids, 1.0)
        np.clip(self.scores, 0.0, self.max_score, out=self.scores)

    def decay(self):
        """Apply multiplicative decay to all scores."""
        self.scores *= self.decay_rate

    def motif_stats(self):
        """Summary statistics of the reinforcement map."""
        active = self.scores > 0.01
        return {
            "n_reinforced_motifs": int(active.sum()),
            "mean_reinf_score": float(self.scores[active].mean()) if active.any() else 0.0,
            "max_reinf_score": float(self.scores.max()),
            "total_reinf_mass": float(self.scores.sum()),
        }
