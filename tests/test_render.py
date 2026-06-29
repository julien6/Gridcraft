from __future__ import annotations

import os
from collections import deque

import numpy as np
import pytest

from gridcraft.config import GridcraftConfig
from gridcraft.constants import ACTION_NAMES, Block, EntityType, Item, Terrain
from gridcraft.entities import ItemDrop, MobState
from gridcraft.env import GridcraftEnv

pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("pygame")
Image = pytest.importorskip("PIL.Image")


def _inventory_panel_width(config: GridcraftConfig) -> int:
    cols = 4
    padding = max(2, config.tile_size // 4)
    observation_tile_size = max(2, (3 * config.tile_size) // config.view_size)
    observation_width = config.view_size * observation_tile_size
    return padding * 3 + cols * config.tile_size + observation_width


def _nearest_adjacent_cell(world, agent, block_type: Block) -> tuple[int, int] | None:
    positions = list(zip(*((world.blocks == block_type).nonzero())))
    best = None
    best_dist = None
    for y, x in positions:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sx, sy = int(x + dx), int(y + dy)
            if not world.is_walkable(sx, sy):
                continue
            dist = abs(agent.x - sx) + abs(agent.y - sy)
            if best_dist is None or dist < best_dist:
                best = (sx, sy)
                best_dist = dist
    return best


def _paint_minecraft_like_scene(world) -> None:
    world.terrain[:] = Terrain.GRASS
    world.blocks[:] = Block.EMPTY
    world.mobs.clear()
    world.items.clear()

    # Natural-looking water patches: one lake, two ponds, and a broken creek.
    lake = {
        (13, 2), (14, 2), (15, 2),
        (12, 3), (13, 3), (14, 3), (15, 3), (16, 3),
        (12, 4), (13, 4), (14, 4), (15, 4),
        (13, 5), (14, 5),
    }
    ponds = {
        (2, 11), (3, 11), (2, 12), (3, 12), (4, 12),
        (16, 10), (17, 10), (16, 11),
    }
    creek = {(9, 8), (10, 9), (10, 10), (11, 11), (10, 12), (9, 13)}
    for x, y in lake | ponds | creek:
        world.terrain[y, x] = Terrain.WATER

    dirt_patches = {
        (1, 6), (2, 6), (3, 7), (4, 6), (5, 6), (6, 5), (7, 6), (8, 6),
        (6, 3), (5, 4), (6, 4), (7, 5), (6, 7), (7, 8),
        (11, 8), (12, 8), (13, 9), (14, 8), (15, 8), (16, 9), (17, 8),
        (3, 10), (5, 11), (14, 12), (15, 13),
    }
    for x, y in dirt_patches:
        if world.terrain[y, x] != Terrain.WATER:
            world.terrain[y, x] = Terrain.DIRT

    forest = {
        (2, 2), (3, 2), (4, 2), (5, 2),
        (2, 3), (4, 3), (6, 3),
        (3, 4), (5, 4), (6, 5),
        (1, 8), (2, 8), (3, 9), (4, 9),
        (17, 5), (18, 5), (18, 6),
    }
    stone_ridge = {
        (8, 3), (9, 3), (8, 4), (9, 4),
        (11, 5), (12, 6), (13, 7),
        (6, 12), (7, 12),
    }
    for x, y in forest:
        if world.terrain[y, x] != Terrain.WATER:
            world.blocks[y, x] = Block.TREE
    for x, y in stone_ridge:
        if world.terrain[y, x] != Terrain.WATER:
            world.blocks[y, x] = Block.STONE


def _place_agent(world, agent, x: int, y: int) -> None:
    agent.x, agent.y = x, y
    agent.hp = world.config.hp_max
    agent.hunger = world.config.hunger_max
    agent.inventory.clear()
    agent.inventory_order.clear()
    agent.equipped = None
    agent.alive = True
    agent.visited_positions = {(x, y)}


def _move_towards(agent, target_x: int, target_y: int, action_index: dict[str, int]) -> int:
    dx = target_x - agent.x
    dy = target_y - agent.y
    if abs(dx) > abs(dy):
        return action_index["move_e" if dx > 0 else "move_w"]
    if dy != 0:
        return action_index["move_s" if dy > 0 else "move_n"]
    return action_index["stay"]


def _scripted_move_towards(world, agent, target: tuple[int, int], action_index: dict[str, int]) -> int:
    start = (agent.x, agent.y)
    queue = deque([(start, None)])
    visited = {start}
    directions = [
        (1, 0, "move_e"),
        (-1, 0, "move_w"),
        (0, 1, "move_s"),
        (0, -1, "move_n"),
    ]

    while queue:
        (x, y), first_action = queue.popleft()
        if (x, y) == target:
            return action_index[first_action] if first_action is not None else action_index["stay"]

        for dx, dy, action_name in directions:
            nx, ny = x + dx, y + dy
            position = (nx, ny)
            if position in visited:
                continue
            if not world.is_walkable(nx, ny):
                continue
            occupant = world._agent_at(nx, ny)
            if occupant is not None and occupant is not agent:
                continue
            if world._mob_at(nx, ny):
                continue
            visited.add(position)
            queue.append((position, first_action or action_name))

    return action_index["stay"]


def _nearest_adjacent_mob_cell(world, agent) -> tuple[int, int] | None:
    best = None
    best_dist = None
    for mob in world.mobs:
        if not mob.alive:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sx, sy = mob.x + dx, mob.y + dy
            if not world.is_walkable(sx, sy):
                continue
            if world._mob_at(sx, sy):
                continue
            occupant = world._agent_at(sx, sy)
            if occupant is not None and occupant is not agent:
                continue
            dist = abs(agent.x - sx) + abs(agent.y - sy)
            if best_dist is None or dist < best_dist:
                best = (sx, sy)
                best_dist = dist
    return best


def _flee_from_mobs(world, agent, action_index: dict[str, int], radius: int = 5) -> int | None:
    alive_mobs = [mob for mob in world.mobs if mob.alive]
    if not alive_mobs:
        return None

    nearest_distance = min(abs(mob.x - agent.x) + abs(mob.y - agent.y) for mob in alive_mobs)
    if nearest_distance > radius:
        return None

    candidates = [
        (0, 0, "stay"),
        (1, 0, "move_e"),
        (-1, 0, "move_w"),
        (0, 1, "move_s"),
        (0, -1, "move_n"),
    ]
    best_action = "stay"
    best_score = nearest_distance
    for dx, dy, action_name in candidates:
        nx, ny = agent.x + dx, agent.y + dy
        if action_name != "stay":
            if not world.is_walkable(nx, ny):
                continue
            occupant = world._agent_at(nx, ny)
            if occupant is not None and occupant is not agent:
                continue
            if world._mob_at(nx, ny):
                continue
        score = min(abs(mob.x - nx) + abs(mob.y - ny) for mob in alive_mobs)
        if score > best_score:
            best_action = action_name
            best_score = score

    return action_index[best_action] if best_action != "stay" else None


def _spawn_zombies(world, positions: list[tuple[int, int]], spawned_mobs: list[MobState]) -> None:
    for mob_id, (x, y) in enumerate(positions, start=1):
        spawn_cell = None
        for radius in range(0, 4):
            candidates = [(x, y)] if radius == 0 else [
                (x + dx, y + dy)
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
                if abs(dx) + abs(dy) == radius
            ]
            for sx, sy in candidates:
                if not world.is_walkable(sx, sy):
                    continue
                if world._agent_at(sx, sy) or world._mob_at(sx, sy):
                    continue
                spawn_cell = (sx, sy)
                break
            if spawn_cell is not None:
                break
        if spawn_cell is None:
            continue
        sx, sy = spawn_cell
        mob = MobState(mob_id=mob_id, x=x, y=y, hp=10)
        mob.x, mob.y = sx, sy
        world.mobs.append(mob)
        spawned_mobs.append(mob)


def test_render_rgb_array():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = GridcraftConfig(
        width=16, height=16, num_agents=2, seed=5, tile_size=8)
    env = GridcraftEnv(config=config, render_mode="rgb_array")
    env.reset(seed=5)

    frame = env.render()
    assert frame is not None
    assert frame.shape == (config.height * config.tile_size,
                           config.width * config.tile_size + _inventory_panel_width(config), 3)

    env.close()


def test_render_tabular_observations_rgb_array():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = GridcraftConfig(
        width=16, height=16, num_agents=1, seed=5, tile_size=8)
    env = GridcraftEnv(config=config, render_mode="rgb_array")
    obs, infos = env.reset(seed=5)

    frame = env.render(tabular_observations=obs)
    base_width = config.width * config.tile_size + _inventory_panel_width(config)
    assert frame is not None
    assert frame.shape == (config.height * config.tile_size,
                           base_width + _inventory_panel_width(config), 3)
    assert frame[:, :config.width * config.tile_size].max() > 0
    assert frame[:, config.width * config.tile_size:base_width].max() > 0
    assert frame[:, base_width:].max() > 0

    env.close()


def test_render_tabular_observations_without_world_rgb_array():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = GridcraftConfig(
        width=16, height=16, num_agents=1, seed=5, tile_size=8)
    env = GridcraftEnv(config=config, render_mode="rgb_array")
    obs, infos = env.reset(seed=5)

    frame = env.renderer.render(None, "rgb_array", tabular_observations=obs)
    assert frame is not None
    assert frame.shape == (config.height * config.tile_size,
                           config.width * config.tile_size + _inventory_panel_width(config), 3)
    assert frame[:, :config.width * config.tile_size].max() == 0
    assert frame[:, config.width * config.tile_size:].max() > 0

    env.close()


def test_render_ground_item_is_smaller_than_tile():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = GridcraftConfig(
        width=8, height=8, num_agents=1, max_mobs=0, seed=5, tile_size=32)
    env = GridcraftEnv(config=config, render_mode="rgb_array")
    env.reset(seed=5)
    env.world.terrain[:] = Terrain.GRASS
    env.world.blocks[:] = Block.EMPTY
    env.world.items.clear()
    env.world.items.append(ItemDrop(item=Item.WOOD, count=1, x=2, y=2))

    frame = env.render()
    ts = config.tile_size
    terrain_pixel = frame[2 * ts + 1, 2 * ts + 1].copy()
    center_pixel = frame[2 * ts + ts // 2, 2 * ts + ts // 2].copy()

    assert (terrain_pixel == frame[1, 1]).all()
    assert not (center_pixel == terrain_pixel).all()
    env.close()


def test_render_tabular_item_overlays_block_instead_of_inventory_slot():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = GridcraftConfig(
        width=8, height=8, num_agents=1, max_mobs=0, seed=5, tile_size=32)
    env = GridcraftEnv(config=config, render_mode="rgb_array")
    env.reset(seed=5)

    grid = np.zeros((3, config.view_size, config.view_size), dtype=np.int8)
    grid[0, :, :] = int(Terrain.GRASS)
    grid[1, 1, 1] = int(Block.STONE)
    grid[2, 1, 1] = int(EntityType.ITEM)
    obs = {
        "agent_0": {
            "grid": grid,
            "self": np.zeros(2 + len(Item), dtype=np.int16),
        }
    }

    frame = env.renderer.render(None, "rgb_array", tabular_observations=obs)
    ts = config.tile_size
    padding = max(2, ts // 4)
    obs_tile = max(2, (3 * ts) // config.view_size)
    x0 = config.width * ts + padding + 4 * ts + padding + obs_tile
    y0 = padding + ts + padding + obs_tile

    corner_pixel = frame[y0 + 1, x0 + 1].copy()
    center_pixel = frame[y0 + obs_tile // 2, x0 + obs_tile // 2].copy()

    assert np.abs(corner_pixel.astype(int) - np.array([120, 120, 120])).max() <= 1
    assert not (center_pixel == corner_pixel).all()
    assert not (center_pixel == np.array([60, 60, 60], dtype=np.uint8)).all()
    env.close()


def test_render_human():
    config = GridcraftConfig(
        width=20,
        height=16,
        num_agents=3,
        max_steps=320,
        max_mobs=8,
        mob_spawn_rate=9999,
        mob_aggro_radius=50,
        mob_move_prob=1.0,
        mob_damage=1,
        hp_max=40,
        tree_apple_drop_chance=0.0,
        item_drop_chance=0.0,
        tile_size=32,
        seed=42,
    )
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = GridcraftEnv(config=config, render_mode="rgb_array")
    obs, infos = env.reset(seed=42)
    assert len(env.possible_agents) == 3
    assert len(env.agents) == 3
    frame_list = [Image.fromarray(env.render())]
    assert set(obs.keys()) == set(env.possible_agents)
    action_index = {name: idx for idx, name in enumerate(ACTION_NAMES)}
    _paint_minecraft_like_scene(env.world)

    crafter_id, scout_id, guard_id = env.possible_agents
    crafter = env.world.agents[crafter_id]
    scout = env.world.agents[scout_id]
    guard = env.world.agents[guard_id]
    _place_agent(env.world, crafter, 2, 6)
    _place_agent(env.world, scout, 4, 8)
    _place_agent(env.world, guard, 15, 8)

    zombie_positions = [(18, 8), (17, 13), (11, 2)]
    spawned_mobs = []
    stone_sword_step = None
    zombie_spawn_delay = 8

    scout_targets = [(4, 8), (3, 9), (5, 7), (7, 6), (6, 3)]
    guard_targets = [(15, 8), (14, 8), (15, 9), (16, 9), (15, 8)]
    scout_target_index = 0
    guard_target_index = 0

    phase = "gather_wood"
    killed_mobs = False
    for step in range(config.max_steps):
        actions = {agent: action_index["stay"] for agent in env.agents}
        if crafter_id in env.agents:
            crafter = env.world.agents[crafter_id]
            inventory = crafter.inventory
            wood = inventory.get(Item.WOOD, 0)
            plank = inventory.get(Item.PLANK, 0)
            stick = inventory.get(Item.STICK, 0)
            stone = inventory.get(Item.STONE, 0)

            if phase == "gather_wood":
                if wood >= 2:
                    phase = "craft_planks"
                    actions[crafter_id] = action_index["craft_plank"]
                else:
                    target = _nearest_adjacent_cell(env.world, crafter, Block.TREE)
                    if target is not None and (crafter.x, crafter.y) == target:
                        actions[crafter_id] = action_index["harvest"]
                    elif target is not None:
                        actions[crafter_id] = _scripted_move_towards(
                            env.world, crafter, target, action_index)
            elif phase == "craft_planks":
                if plank >= 3:
                    phase = "craft_sticks"
                    actions[crafter_id] = action_index["craft_stick"]
                elif wood > 0:
                    actions[crafter_id] = action_index["craft_plank"]
                else:
                    phase = "gather_wood"
            elif phase == "craft_sticks":
                if stick >= 1:
                    phase = "craft_wood_pickaxe"
                    if plank >= 1:
                        actions[crafter_id] = action_index["craft_wood_pickaxe"]
                elif plank >= 2:
                    actions[crafter_id] = action_index["craft_stick"]
                else:
                    phase = "craft_planks"
            elif phase == "craft_wood_pickaxe":
                if crafter.equipped == Item.WOOD_PICKAXE:
                    phase = "mine_stone"
                    target = _nearest_adjacent_cell(env.world, crafter, Block.STONE)
                    if target is not None:
                        actions[crafter_id] = _scripted_move_towards(
                            env.world, crafter, target, action_index)
                elif stick >= 1 and plank >= 1:
                    actions[crafter_id] = action_index["craft_wood_pickaxe"]
                else:
                    phase = "craft_planks"
            elif phase == "mine_stone":
                if stone >= 1:
                    phase = "craft_stone_sword"
                    if stick >= 1:
                        actions[crafter_id] = action_index["craft_stone_sword"]
                else:
                    target = _nearest_adjacent_cell(env.world, crafter, Block.STONE)
                    if target is not None and (crafter.x, crafter.y) == target:
                        actions[crafter_id] = action_index["harvest"]
                    elif target is not None:
                        actions[crafter_id] = _scripted_move_towards(
                            env.world, crafter, target, action_index)
            elif phase == "craft_stone_sword":
                if crafter.equipped == Item.STONE_SWORD:
                    phase = "wait_zombie_spawn"
                    stone_sword_step = step
                elif stick >= 1 and stone >= 1:
                    actions[crafter_id] = action_index["craft_stone_sword"]
                else:
                    phase = "mine_stone"
            elif phase == "wait_zombie_spawn":
                if stone_sword_step is not None and step - stone_sword_step >= zombie_spawn_delay:
                    _spawn_zombies(env.world, zombie_positions, spawned_mobs)
                    phase = "hunt_zombie"
                else:
                    target = guard_targets[guard_target_index % len(guard_targets)]
                    actions[crafter_id] = _scripted_move_towards(
                        env.world, crafter, target, action_index)
            elif phase == "hunt_zombie":
                alive_mobs = [mob for mob in env.world.mobs if mob.alive]
                if not alive_mobs:
                    killed_mobs = True
                    phase = "done"
                else:
                    mob = alive_mobs[0]
                    if abs(mob.x - crafter.x) + abs(mob.y - crafter.y) == 1:
                        actions[crafter_id] = action_index["attack"]
                    else:
                        target = _nearest_adjacent_mob_cell(env.world, crafter)
                        if target is not None:
                            actions[crafter_id] = _scripted_move_towards(
                                env.world, crafter, target, action_index)

        if scout_id in env.agents:
            scout = env.world.agents[scout_id]
            target = scout_targets[scout_target_index % len(scout_targets)]
            if (scout.x, scout.y) == target:
                if _nearest_adjacent_cell(env.world, scout, Block.TREE):
                    actions[scout_id] = action_index["harvest"]
                else:
                    scout_target_index += 1
                    target = scout_targets[scout_target_index % len(scout_targets)]
                    actions[scout_id] = _scripted_move_towards(
                        env.world, scout, target, action_index)
            else:
                actions[scout_id] = _scripted_move_towards(
                    env.world, scout, target, action_index)

        if guard_id in env.agents:
            guard = env.world.agents[guard_id]
            alive_mobs = [mob for mob in env.world.mobs if mob.alive]
            if alive_mobs:
                nearest_mob = min(
                    alive_mobs,
                    key=lambda mob: abs(mob.x - guard.x) + abs(mob.y - guard.y),
                )
                if abs(nearest_mob.x - guard.x) + abs(nearest_mob.y - guard.y) == 1:
                    actions[guard_id] = action_index["attack"]
                else:
                    target = _nearest_adjacent_mob_cell(env.world, guard)
                    if target is not None:
                        actions[guard_id] = _scripted_move_towards(
                            env.world, guard, target, action_index)
            else:
                target = guard_targets[guard_target_index % len(guard_targets)]
                if (guard.x, guard.y) == target:
                    guard_target_index += 1
                    target = guard_targets[guard_target_index % len(guard_targets)]
                    actions[guard_id] = _scripted_move_towards(
                        env.world, guard, target, action_index)
                else:
                    actions[guard_id] = _scripted_move_towards(
                        env.world, guard, target, action_index)

        for agent_id in list(env.agents):
            agent = env.world.agents[agent_id]
            if agent.equipped == Item.STONE_SWORD:
                continue
            flee_action = _flee_from_mobs(env.world, agent, action_index)
            if flee_action is not None:
                actions[agent_id] = flee_action

        obs, rewards, terminations, truncations, infos = env.step(actions)
        frame_list.append(Image.fromarray(env.render()))
        assert all(isinstance(value, float) for value in rewards.values())
        if phase == "done":
            break
        if all(terminations.values()) or all(truncations.values()):
            break
    gif_path = os.environ.get("GRIDCRAFT_RENDER_GIF", "render_human.gif")
    frame_list[0].save(
        gif_path,
        save_all=True,
        append_images=frame_list[1:],
        duration=max(1, int(1000 / config.fps)),
        loop=0,
    )
    assert os.path.isfile(gif_path)
    assert crafter.equipped == Item.STONE_SWORD
    assert crafter.inventory.get(Item.STONE, 0) == 0
    assert len(spawned_mobs) == len(zombie_positions)
    assert killed_mobs
    assert all(env.world.agents[agent_id].alive for agent_id in env.possible_agents)
    env.close()


if __name__ == "__main__":
    test_render_human()
