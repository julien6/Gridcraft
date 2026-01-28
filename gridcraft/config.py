from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GridcraftConfig:
    width: int = 64
    height: int = 64
    num_agents: int = 2
    seed: int | None = None

    tree_density: float = 0.08
    stone_density: float = 0.06
    water_density: float = 0.06

    hp_max: int = 20
    hunger_max: int = 20
    hunger_decay_ticks: int = 5
    starvation_damage: int = 1

    mob_spawn_rate: int = 10
    max_mobs: int = 6
    mob_damage: int = 2
    mob_hp: int = 10
    mob_aggro_radius: int = 6
    mob_move_prob: float = 0.8

    view_size: int = 9
    max_steps: int = 500

    item_drop_chance: float = 0.2

    craft_anywhere: bool = True

    tile_size: int = 48
    fps: int = 12
    asset_path: str | None = None
