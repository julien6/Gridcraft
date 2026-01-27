from __future__ import annotations

import numpy as np

try:
    from pettingzoo import ParallelEnv
except ImportError:  # pragma: no cover
    ParallelEnv = object  # type: ignore

from gymnasium import spaces

from .config import GridcraftConfig
from .constants import ACTION_NAMES, Item
from .render import PygameRenderer
from .world import GridcraftWorld


class GridcraftEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "gridcraft_v0"}

    def __init__(self, config: GridcraftConfig | None = None, render_mode: str | None = None):
        self.config = config or GridcraftConfig()
        self.render_mode = render_mode
        self.agents = [f"agent_{i}" for i in range(self.config.num_agents)]
        self.possible_agents = list(self.agents)
        self._rng = np.random.default_rng(self.config.seed)
        self.world = GridcraftWorld(self.config, self._rng)
        self.renderer = PygameRenderer(self.config) if render_mode else None

        self.action_spaces = {
            agent: spaces.Discrete(len(ACTION_NAMES)) for agent in self.possible_agents
        }
        inv_size = len(Item)
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "grid": spaces.Box(
                        low=0,
                        high=10,
                        shape=(3, self.config.view_size, self.config.view_size),
                        dtype=np.int8,
                    ),
                    "self": spaces.Box(
                        low=0,
                        high=max(self.config.hp_max, self.config.hunger_max, 99),
                        shape=(2 + inv_size,),
                        dtype=np.int16,
                    ),
                }
            )
            for agent in self.possible_agents
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.world = GridcraftWorld(self.config, self._rng)
        self.world.reset(self.possible_agents)
        observations = self.world.observations()
        infos = {agent: {} for agent in self.possible_agents}
        self.agents = list(self.possible_agents)
        return observations, infos

    def step(self, actions: dict[str, int]):
        result = self.world.step(actions)
        observations = self.world.observations()
        self.agents = [agent_id for agent_id, agent in self.world.agents.items() if agent.alive]
        return observations, result.rewards, result.terminations, result.truncations, result.infos

    def render(self):
        if self.render_mode is None:
            return None
        return self.renderer.render(self.world, self.render_mode)

    def close(self):
        if self.renderer:
            self.renderer.close()

    def state(self):
        return self.world.observations()
