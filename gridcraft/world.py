from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GridcraftConfig
from .constants import ACTION_NAMES, ACTION_TO_RECIPE, Block, Item, Terrain
from .entities import AgentState, ItemDrop, MobState


@dataclass
class StepResult:
    rewards: dict[str, float]
    terminations: dict[str, bool]
    truncations: dict[str, bool]
    infos: dict[str, dict]


class GridcraftWorld:
    def __init__(self, config: GridcraftConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.terrain = np.full(
            (config.height, config.width), Terrain.GRASS, dtype=np.int8)
        self.blocks = np.full((config.height, config.width),
                              Block.EMPTY, dtype=np.int8)
        self.agents: dict[str, AgentState] = {}
        self.mobs: list[MobState] = []
        self.items: list[ItemDrop] = []
        self.step_count = 0
        self._mob_counter = 0

    def reset(self, agent_ids: list[str]) -> None:
        self.step_count = 0
        self._mob_counter = 0
        self.terrain[:] = Terrain.GRASS
        self.blocks[:] = Block.EMPTY
        self.items.clear()
        self.mobs.clear()
        self.agents = {}
        self._generate_terrain()
        self._spawn_blocks()
        for agent_id in agent_ids:
            x, y = self._find_open_cell()
            self.agents[agent_id] = AgentState(
                agent_id=agent_id,
                x=x,
                y=y,
                hp=self.config.hp_max,
                hunger=self.config.hunger_max,
                inventory={},
            )
        self._spawn_initial_mobs()

    def _generate_terrain(self) -> None:
        self.terrain[:] = Terrain.GRASS
        height, width = self.terrain.shape
        target_water = int(self.config.water_density * width * height)
        water_count = 0

        while water_count < target_water:
            start_x = int(self.rng.integers(0, width))
            start_y = int(self.rng.integers(0, height))
            if self.terrain[start_y, start_x] == Terrain.WATER:
                continue

            lake_size = int(self.rng.integers(4, 12))
            lake_cells: list[tuple[int, int]] = [(start_x, start_y)]
            self.terrain[start_y, start_x] = Terrain.WATER
            water_count += 1

            attempts = 0
            while len(lake_cells) < lake_size and attempts < lake_size * 10:
                attempts += 1
                base_x, base_y = lake_cells[int(
                    self.rng.integers(0, len(lake_cells)))]
                dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)
                          ][int(self.rng.integers(0, 4))]
                nx, ny = base_x + dx, base_y + dy
                if 0 <= nx < width and 0 <= ny < height and self.terrain[ny, nx] != Terrain.WATER:
                    self.terrain[ny, nx] = Terrain.WATER
                    lake_cells.append((nx, ny))
                    water_count += 1
                    if water_count >= target_water:
                        break

        dirt_mask = (self.rng.random(self.terrain.shape) < 0.05) & (
            self.terrain != Terrain.WATER)
        self.terrain[dirt_mask] = Terrain.DIRT

    def _spawn_blocks(self) -> None:
        tree_mask = self.rng.random(
            self.blocks.shape) < self.config.tree_density
        stone_mask = self.rng.random(
            self.blocks.shape) < self.config.stone_density
        self.blocks[tree_mask] = Block.TREE
        self.blocks[stone_mask] = Block.STONE
        self.blocks[self.terrain == Terrain.WATER] = Block.EMPTY

    def _find_open_cell(self) -> tuple[int, int]:
        for _ in range(1000):
            x = self.rng.integers(0, self.config.width)
            y = self.rng.integers(0, self.config.height)
            if self.is_walkable(x, y) and not self._agent_at(x, y) and not self._mob_at(x, y):
                return int(x), int(y)
        raise RuntimeError("Failed to find open cell")

    def _spawn_initial_mobs(self) -> None:
        for _ in range(min(self.config.max_mobs // 2, 2)):
            self.spawn_mob()

    def is_walkable(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.config.width or y >= self.config.height:
            return False
        if self.terrain[y, x] == Terrain.WATER:
            return False
        if self.blocks[y, x] in (Block.TREE, Block.STONE):
            return False
        return True

    def _agent_at(self, x: int, y: int) -> AgentState | None:
        for agent in self.agents.values():
            if agent.alive and agent.x == x and agent.y == y:
                return agent
        return None

    def _mob_at(self, x: int, y: int) -> MobState | None:
        for mob in self.mobs:
            if mob.alive and mob.x == x and mob.y == y:
                return mob
        return None

    def _items_at(self, x: int, y: int) -> list[ItemDrop]:
        return [item for item in self.items if item.x == x and item.y == y]

    def spawn_mob(self) -> None:
        if len(self.mobs) >= self.config.max_mobs:
            return
        for _ in range(200):
            x = self.rng.integers(0, self.config.width)
            y = self.rng.integers(0, self.config.height)
            if not self.is_walkable(int(x), int(y)):
                continue
            if self._agent_at(int(x), int(y)):
                continue
            if self._mob_at(int(x), int(y)):
                continue
            if self._distance_to_nearest_agent(int(x), int(y)) < self.config.mob_aggro_radius:
                continue
            self._mob_counter += 1
            self.mobs.append(
                MobState(mob_id=self._mob_counter, x=int(
                    x), y=int(y), hp=self.config.mob_hp)
            )
            return

    def _distance_to_nearest_agent(self, x: int, y: int) -> int:
        distances = [abs(agent.x - x) + abs(agent.y - y)
                     for agent in self.agents.values() if agent.alive]
        return min(distances) if distances else 999

    def step(self, actions: dict[str, int]) -> StepResult:
        self.step_count += 1
        rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
        terminations = {agent_id: False for agent_id in self.agents.keys()}
        truncations = {agent_id: False for agent_id in self.agents.keys()}
        infos = {agent_id: {} for agent_id in self.agents.keys()}

        for agent_id, action in actions.items():
            agent = self.agents[agent_id]
            if not agent.alive:
                continue
            self._apply_action(agent, action)

        self._move_mobs()
        self._resolve_mob_attacks(rewards)
        self._handle_hunger(rewards)
        self._cleanup_entities()

        if self.step_count % self.config.mob_spawn_rate == 0:
            self.spawn_mob()

        alive_agents = [agent for agent in self.agents.values() if agent.alive]
        if not alive_agents:
            for agent_id in terminations:
                terminations[agent_id] = True
        if self.step_count >= self.config.max_steps:
            for agent_id in truncations:
                truncations[agent_id] = True

        if alive_agents:
            for agent_id in rewards:
                rewards[agent_id] += 1.0
        else:
            for agent_id in rewards:
                rewards[agent_id] -= 100.0

        return StepResult(rewards=rewards, terminations=terminations, truncations=truncations, infos=infos)

    def _apply_action(self, agent: AgentState, action: int) -> None:
        action_name = ACTION_NAMES[action]
        if action_name.startswith("move"):
            self._move_agent(agent, action_name)
        elif action_name == "harvest":
            self._harvest(agent)
        elif action_name == "pickup":
            self._pickup(agent)
        elif action_name == "attack":
            self._attack(agent)
        elif action_name == "eat":
            self._eat(agent)
        elif action_name.startswith("craft"):
            recipe_name = ACTION_TO_RECIPE[action_name]
            self._craft(agent, recipe_name)

    def _move_agent(self, agent: AgentState, action_name: str) -> None:
        dx, dy = 0, 0
        if action_name == "move_n":
            dy = -1
        elif action_name == "move_s":
            dy = 1
        elif action_name == "move_w":
            dx = -1
        elif action_name == "move_e":
            dx = 1
        nx, ny = agent.x + dx, agent.y + dy
        if self.is_walkable(nx, ny) and not self._agent_at(nx, ny) and not self._mob_at(nx, ny):
            agent.x, agent.y = nx, ny

    def _harvest(self, agent: AgentState) -> None:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if abs(dx) + abs(dy) != 1:
                    continue
                nx, ny = agent.x + dx, agent.y + dy
                if 0 <= nx < self.config.width and 0 <= ny < self.config.height:
                    block = self.blocks[ny, nx]
                    if block == Block.TREE:
                        self.blocks[ny, nx] = Block.EMPTY
                        self._add_item(agent, Item.WOOD, 1)
                        return
                    elif block == Block.STONE:
                        if agent.equipped in (Item.WOOD_PICKAXE, Item.STONE_PICKAXE):
                            self.blocks[ny, nx] = Block.EMPTY
                            self._add_item(agent, Item.STONE, 1)
                            return

    def _pickup(self, agent: AgentState) -> None:
        items_here = self._items_at(agent.x, agent.y)
        if not items_here:
            return
        for item in items_here:
            self._add_item(agent, item.item, item.count)
        self.items = [item for item in self.items if item not in items_here]

    def _attack(self, agent: AgentState) -> None:
        for mob in self.mobs:
            if mob.alive and abs(mob.x - agent.x) + abs(mob.y - agent.y) == 1:
                damage = 2
                if agent.equipped in (Item.WOOD_SWORD, Item.STONE_SWORD):
                    damage = 4 if agent.equipped == Item.STONE_SWORD else 3
                mob.hp -= damage
                if mob.hp <= 0:
                    mob.alive = False
                    if self.rng.random() < self.config.item_drop_chance:
                        self.items.append(
                            ItemDrop(item=Item.APPLE, count=1, x=mob.x, y=mob.y))
                return

    def _eat(self, agent: AgentState) -> None:
        if agent.inventory.get(Item.APPLE, 0) > 0:
            agent.inventory[Item.APPLE] -= 1
            agent.hunger = min(self.config.hunger_max, agent.hunger + 6)

    def _craft(self, agent: AgentState, recipe_name: str) -> None:
        if not self.config.craft_anywhere and self.blocks[agent.y, agent.x] != Block.CRAFTING_TABLE:
            return
        recipe = {
            "plank": {"inputs": {Item.WOOD: 1}, "outputs": {Item.PLANK: 2}},
            "stick": {"inputs": {Item.PLANK: 2}, "outputs": {Item.STICK: 4}},
            "wood_sword": {"inputs": {Item.STICK: 1, Item.PLANK: 1}, "outputs": {Item.WOOD_SWORD: 1}},
            "stone_sword": {"inputs": {Item.STICK: 1, Item.STONE: 1}, "outputs": {Item.STONE_SWORD: 1}},
            "wood_pickaxe": {"inputs": {Item.STICK: 1, Item.PLANK: 1}, "outputs": {Item.WOOD_PICKAXE: 1}},
            "stone_pickaxe": {"inputs": {Item.STICK: 1, Item.STONE: 1}, "outputs": {Item.STONE_PICKAXE: 1}},
        }[recipe_name]
        if all(agent.inventory.get(item, 0) >= count for item, count in recipe["inputs"].items()):
            for item, count in recipe["inputs"].items():
                agent.inventory[item] -= count
            for item, count in recipe["outputs"].items():
                self._add_item(agent, item, count)

    def _add_item(self, agent: AgentState, item: Item, count: int) -> None:
        if item not in agent.inventory:
            agent.inventory_order.append(item)
        agent.inventory[item] = agent.inventory.get(item, 0) + count
        if item in (Item.WOOD_SWORD, Item.STONE_SWORD, Item.WOOD_PICKAXE, Item.STONE_PICKAXE):
            agent.equipped = item

    def _move_mobs(self) -> None:
        for mob in self.mobs:
            if not mob.alive:
                continue
            if self.rng.random() > self.config.mob_move_prob:
                continue
            target = self._nearest_agent(mob)
            if target and abs(target.x - mob.x) + abs(target.y - mob.y) <= self.config.mob_aggro_radius:
                dx = int(np.sign(target.x - mob.x))
                dy = int(np.sign(target.y - mob.y))
                nx, ny = mob.x + dx, mob.y + dy
                if self.is_walkable(nx, ny) and not self._mob_at(nx, ny) and not self._agent_at(nx, ny):
                    mob.x, mob.y = nx, ny
            else:
                dx, dy = self.rng.choice([-1, 0, 1], 2)
                nx, ny = mob.x + int(dx), mob.y + int(dy)
                if self.is_walkable(nx, ny) and not self._mob_at(nx, ny) and not self._agent_at(nx, ny):
                    mob.x, mob.y = nx, ny

    def _nearest_agent(self, mob: MobState) -> AgentState | None:
        alive_agents = [agent for agent in self.agents.values() if agent.alive]
        if not alive_agents:
            return None
        return min(alive_agents, key=lambda agent: abs(agent.x - mob.x) + abs(agent.y - mob.y))

    def _resolve_mob_attacks(self, rewards: dict[str, float]) -> None:
        for mob in self.mobs:
            if not mob.alive:
                continue
            for agent in self.agents.values():
                if agent.alive and abs(agent.x - mob.x) + abs(agent.y - mob.y) == 1:
                    agent.hp -= self.config.mob_damage
                    rewards[agent.agent_id] -= self.config.mob_damage
                    if agent.hp <= 0:
                        agent.alive = False

    def _handle_hunger(self, rewards: dict[str, float]) -> None:
        if self.step_count % self.config.hunger_decay_ticks != 0:
            return
        for agent in self.agents.values():
            if not agent.alive:
                continue
            agent.hunger = max(0, agent.hunger - 1)
            if agent.hunger == 0:
                agent.hp -= self.config.starvation_damage
                rewards[agent.agent_id] -= float(self.config.starvation_damage)
                if agent.hp <= 0:
                    agent.alive = False

    def _cleanup_entities(self) -> None:
        self.mobs = [mob for mob in self.mobs if mob.alive]

    def observations(self) -> dict[str, dict]:
        obs: dict[str, dict] = {}
        for agent_id, agent in self.agents.items():
            obs[agent_id] = {
                "grid": self._extract_view(agent),
                "self": self._agent_vector(agent),
            }
        return obs

    def _extract_view(self, agent: AgentState) -> np.ndarray:
        radius = self.config.view_size // 2
        size = self.config.view_size
        grid = np.zeros((3, size, size), dtype=np.int8)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = agent.x + dx
                y = agent.y + dy
                gx = dx + radius
                gy = dy + radius
                if 0 <= x < self.config.width and 0 <= y < self.config.height:
                    grid[0, gy, gx] = int(self.terrain[y, x])
                    grid[1, gy, gx] = int(self.blocks[y, x])
                    grid[2, gy, gx] = self._entity_code(x, y)
                else:
                    grid[0, gy, gx] = int(Terrain.WATER)
        return grid

    def _entity_code(self, x: int, y: int) -> int:
        if self._mob_at(x, y):
            return 2
        if self._agent_at(x, y):
            return 1
        if self._items_at(x, y):
            return 3
        return 0

    def _agent_vector(self, agent: AgentState) -> np.ndarray:
        item_counts = [agent.inventory.get(item, 0) for item in Item]
        return np.array([agent.hp, agent.hunger, *item_counts], dtype=np.int16)
