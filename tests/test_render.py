import os
from collections import deque

import pytest

from gridcraft.config import GridcraftConfig
from gridcraft.constants import ACTION_NAMES, Block, Item, Terrain
from gridcraft.entities import MobState
from gridcraft.env import GridcraftEnv
# from PIL import Image

pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("pygame")


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


def test_render_human():
    config = GridcraftConfig(
        width=12,
        height=12,
        num_agents=2,
        max_steps=160,
        max_mobs=0,
        mob_move_prob=0.0,
        tree_apple_drop_chance=0.0,
        seed=42,
    )
    env = GridcraftEnv(config=config, render_mode="human")
    obs, infos = env.reset(seed=42)
    # frame_list = [Image.fromarray(env.render())]
    assert set(obs.keys()) == set(env.possible_agents)
    action_index = {name: idx for idx, name in enumerate(ACTION_NAMES)}
    target_agent = env.possible_agents[0]
    agent_state = env.world.agents[target_agent]

    env.world.terrain[:] = Terrain.GRASS
    env.world.blocks[:] = Block.EMPTY
    env.world.mobs.clear()
    env.world.items.clear()

    agent_state.x, agent_state.y = 2, 2
    agent_state.visited_positions = {(agent_state.x, agent_state.y)}
    other_agent = env.world.agents[env.possible_agents[1]]
    other_agent.x, other_agent.y = 10, 10
    other_agent.visited_positions = {(other_agent.x, other_agent.y)}

    env.world.blocks[2, 4] = Block.TREE
    env.world.blocks[4, 2] = Block.TREE

    phase = "gather_wood"
    spawned_mob = None
    killed_mob = False
    for _ in range(config.max_steps):
        actions = {agent: action_index["stay"] for agent in env.agents}
        if target_agent in env.agents:
            agent_state = env.world.agents[target_agent]
            inventory = agent_state.inventory
            wood = inventory.get(Item.WOOD, 0)
            plank = inventory.get(Item.PLANK, 0)
            stick = inventory.get(Item.STICK, 0)

            if phase == "gather_wood":
                if wood >= 2:
                    phase = "craft_planks"
                else:
                    target = _nearest_adjacent_cell(env.world, agent_state, Block.TREE)
                    if target is not None and (agent_state.x, agent_state.y) == target:
                        actions[target_agent] = action_index["harvest"]
                    elif target is not None:
                        actions[target_agent] = _scripted_move_towards(
                            env.world, agent_state, target, action_index)
            elif phase == "craft_planks":
                if plank >= 3:
                    phase = "craft_sticks"
                elif wood > 0:
                    actions[target_agent] = action_index["craft_plank"]
                else:
                    phase = "gather_wood"
            elif phase == "craft_sticks":
                if stick >= 1:
                    phase = "craft_sword"
                elif plank >= 2:
                    actions[target_agent] = action_index["craft_stick"]
                else:
                    phase = "craft_planks"
            elif phase == "craft_sword":
                if agent_state.equipped == Item.WOOD_SWORD:
                    phase = "spawn_zombie"
                elif stick >= 1 and plank >= 1:
                    actions[target_agent] = action_index["craft_wood_sword"]
                else:
                    phase = "craft_planks"
            elif phase == "spawn_zombie":
                spawned_mob = MobState(mob_id=1, x=8, y=2, hp=6)
                env.world.mobs.append(spawned_mob)
                phase = "hunt_zombie"
            elif phase == "hunt_zombie":
                alive_mobs = [mob for mob in env.world.mobs if mob.alive]
                if not alive_mobs:
                    killed_mob = True
                    phase = "done"
                else:
                    mob = alive_mobs[0]
                    if abs(mob.x - agent_state.x) + abs(mob.y - agent_state.y) == 1:
                        actions[target_agent] = action_index["attack"]
                    else:
                        target = _nearest_adjacent_mob_cell(env.world, agent_state)
                        if target is not None:
                            actions[target_agent] = _scripted_move_towards(
                                env.world, agent_state, target, action_index)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
        assert all(isinstance(value, float) for value in rewards.values())
        if phase == "done":
            break
        if all(terminations.values()) or all(truncations.values()):
            break
        # img = Image.fromarray(env.render())
        # frame_list.append(img)

    # frame_list[0].save("out.gif", save_all=True,
    #                    append_images=frame_list[1:], duration=5, loop=0)
    assert agent_state.equipped == Item.WOOD_SWORD
    assert spawned_mob is not None
    assert killed_mob
    env.close()


if __name__ == "__main__":
    test_render_human()
