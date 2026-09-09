"""Stable-Baselines3 adapter for the structured RCMPSP environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environments.rcmpsp_env import RCMPSPEnv
from src.environments.observation import (
    flatten_observation,
    observation_size,
)


class FlattenRCMPSPObservation(gym.ObservationWrapper):
    """Flatten and normalize RCMPSP observations to a float32 Box."""

    def __init__(self, env: RCMPSPEnv):
        super().__init__(env)
        size = observation_size(env.activity_count, env.resource_count)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(size,), dtype=np.float32)
        self._flat_buffer = np.empty(size, dtype=np.float32)
        self._capacity_scale = np.maximum(np.asarray(env.instance.capacities, dtype=np.float32), 1.0)

    def observation(self, observation):
        return flatten_observation(
            observation,
            self.env.instance.capacities,
            self.env.horizon,
            capacity_scale=self._capacity_scale,
            out=self._flat_buffer,
        )


def make_sb3_env(instance) -> FlattenRCMPSPObservation:
    return FlattenRCMPSPObservation(RCMPSPEnv(instance))
