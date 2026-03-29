"""
Environment definitions for the experiment suite.

Environments modify energy input and transform costs spatially and temporally,
creating heterogeneous and non-stationary conditions.
"""
import numpy as np


class Environment:
    """Base environment: homogeneous and static."""

    def __init__(self, config):
        self.base_e_in = config["energy"]["E_in"]
        self.width = config["grid"]["width"]
        self.height = config["grid"]["height"]
        self.base_tc = config.get("evolution", {}).get("transform_cost", 0.50)

    def get_e_in(self, step):
        return np.full(self.width, self.base_e_in / self.width)

    def get_transform_cost(self, step):
        return self.base_tc

    def describe(self):
        return "Homogeneous static"


class GradientEnv(Environment):
    """Spatial gradient: transform_cost varies left-to-right across columns."""

    def __init__(self, config, cost_range=(0.2, 0.8)):
        super().__init__(config)
        self.cost_range = cost_range
        col_costs = np.linspace(cost_range[0], cost_range[1], self.width)
        self.cost_map = np.broadcast_to(
            col_costs[np.newaxis, :], (self.height, self.width)
        ).copy()

    def get_transform_cost(self, step):
        return self.cost_map

    def describe(self):
        return f"Gradient: cost {self.cost_range[0]}-{self.cost_range[1]}"


class PatchyEnv(Environment):
    """Patchy environment: 4 patches with different transform costs and energy levels."""

    def __init__(self, config, n_patches=4):
        super().__init__(config)
        self.n_patches = n_patches
        pw = self.width // n_patches
        patch_costs = [0.1, 0.6, 0.15, 0.8][:n_patches]
        patch_energy = [1.5, 0.5, 1.2, 0.7][:n_patches]

        self.cost_map = np.full((self.height, self.width), self.base_tc)
        self.e_in_map = np.full(self.width, self.base_e_in / self.width)
        self.patch_ranges = []

        for i in range(n_patches):
            c0 = i * pw
            c1 = (i + 1) * pw if i < n_patches - 1 else self.width
            self.cost_map[:, c0:c1] = patch_costs[i]
            self.e_in_map[c0:c1] *= patch_energy[i]
            self.patch_ranges.append((c0, c1))

    def get_e_in(self, step):
        return self.e_in_map.copy()

    def get_transform_cost(self, step):
        return self.cost_map

    def describe(self):
        return f"Patchy: {self.n_patches} patches"


class SwitchingEnv(Environment):
    """Periodic switching between two transform cost settings."""

    def __init__(self, config, period=500, cost_A=0.3, cost_B=0.7):
        super().__init__(config)
        self.period = period
        self.cost_A = cost_A
        self.cost_B = cost_B

    def get_transform_cost(self, step):
        return self.cost_A if (step // self.period) % 2 == 0 else self.cost_B

    def describe(self):
        return f"Switching: {self.cost_A}/{self.cost_B} every {self.period}"


class DriftingEnv(Environment):
    """Slow drift: transform_cost changes linearly over time."""

    def __init__(self, config, cost_start=0.3, cost_end=0.7, n_steps=3000):
        super().__init__(config)
        self.cost_start = cost_start
        self.cost_end = cost_end
        self.n_steps = n_steps

    def get_transform_cost(self, step):
        frac = min(step / max(self.n_steps - 1, 1), 1.0)
        return self.cost_start + frac * (self.cost_end - self.cost_start)

    def describe(self):
        return f"Drift: cost {self.cost_start}->{self.cost_end}"


class ShockEnv(Environment):
    """Random shocks: abrupt changes at random times."""

    def __init__(self, config, shock_prob=0.003, cost_range=(0.2, 0.8), seed=42):
        super().__init__(config)
        self.shock_prob = shock_prob
        self.cost_range = cost_range
        self._schedule = {}
        self._rng_seed = seed

    def precompute(self, n_steps):
        rng = np.random.default_rng(self._rng_seed)
        self._schedule = {}
        for step in range(n_steps):
            if rng.random() < self.shock_prob:
                self._schedule[step] = rng.uniform(*self.cost_range)
        return self

    def get_transform_cost(self, step):
        current = self.base_tc
        for s in sorted(self._schedule.keys()):
            if s <= step:
                current = self._schedule[s]
            else:
                break
        return current

    def describe(self):
        return f"Shocks: prob={self.shock_prob}"
