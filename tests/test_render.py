import os
import random
import pytest

from gridcraft.config import GridcraftConfig
from gridcraft.constants import ACTION_NAMES, Block, Item, Terrain
from gridcraft.env import GridcraftEnv
# from PIL import Image

pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("pygame")


def _inventory_panel_width(config: GridcraftConfig) -> int:
    cols = 4
    padding = max(2, config.tile_size // 4)
    return padding * 2 + cols * config.tile_size


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


def test_render_human():
    config = GridcraftConfig(
        width=12, height=12, num_agents=2, max_steps=200, max_mobs=1, seed=42, asset_path="../gridcraft/assets/")
    env = GridcraftEnv(config=config, render_mode="human")
    obs, infos = env.reset(seed=42)
    # frame_list = [Image.fromarray(env.render())]
    assert set(obs.keys()) == set(env.possible_agents)
    action_index = {name: idx for idx, name in enumerate(ACTION_NAMES)}
    target_agent = env.possible_agents[0]
    agent_state = env.world.agents[target_agent]
    tree_positions = list(zip(*((env.world.blocks == Block.TREE).nonzero())))
    if len(tree_positions) < 4:
        needed = 4 - len(tree_positions)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if needed == 0:
                    break
                x = agent_state.x + dx
                y = agent_state.y + dy
                if 0 <= x < config.width and 0 <= y < config.height:
                    if env.world.terrain[y, x] != Terrain.WATER and env.world.blocks[y, x] == Block.EMPTY:
                        env.world.blocks[y, x] = Block.TREE
                        needed -= 1
            if needed == 0:
                break
    stone_positions = list(zip(*((env.world.blocks == Block.STONE).nonzero())))
    if not stone_positions:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x = agent_state.x + dx
            y = agent_state.y + dy
            if 0 <= x < config.width and 0 <= y < config.height:
                if env.world.terrain[y, x] != Terrain.WATER and env.world.blocks[y, x] == Block.EMPTY:
                    env.world.blocks[y, x] = Block.STONE
                    break
    phase = "gather_wood"
    prev_hp = {
        agent_id: env.world.agents[agent_id].hp for agent_id in env.world.agents}
    flee_ticks = {agent_id: 0 for agent_id in env.world.agents}
    for _ in range(200):
        actions = {agent: random.choice(
            list(action_index.values())) for agent in env.agents}
        if target_agent in env.agents:
            agent_state = env.world.agents[target_agent]
            inventory = agent_state.inventory
            wood = inventory.get(Item.WOOD, 0)
            plank = inventory.get(Item.PLANK, 0)
            stick = inventory.get(Item.STICK, 0)
            pickaxe = inventory.get(Item.WOOD_PICKAXE, 0)
            stone = inventory.get(Item.STONE, 0)

            took_damage = agent_state.hp < prev_hp.get(
                target_agent, agent_state.hp)
            prev_hp[target_agent] = agent_state.hp
            if took_damage:
                flee_ticks[target_agent] = 3

            if flee_ticks.get(target_agent, 0) > 0:
                adj_mob = None
                for mob in env.world.mobs:
                    if mob.alive and abs(mob.x - agent_state.x) + abs(mob.y - agent_state.y) == 1:
                        adj_mob = mob
                        break
                if adj_mob is not None:
                    actions[target_agent] = action_index["attack"]
                else:
                    nearest = None
                    best_dist = None
                    for mob in env.world.mobs:
                        if not mob.alive:
                            continue
                        dist = abs(mob.x - agent_state.x) + \
                            abs(mob.y - agent_state.y)
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            nearest = mob
                    if nearest is not None:
                        dx = agent_state.x - nearest.x
                        dy = agent_state.y - nearest.y
                        if abs(dx) > abs(dy):
                            actions[target_agent] = action_index["move_e" if dx >
                                                                 0 else "move_w"]
                        elif dy != 0:
                            actions[target_agent] = action_index["move_s" if dy >
                                                                 0 else "move_n"]
                flee_ticks[target_agent] -= 1
            elif phase == "gather_wood":
                if wood >= 4:
                    phase = "craft_plank"
                else:
                    target = _nearest_adjacent_cell(
                        env.world, agent_state, Block.TREE)
                    if target is not None:
                        tx, ty = target
                        if agent_state.x == tx and agent_state.y == ty:
                            actions[target_agent] = action_index["harvest"]
                        else:
                            actions[target_agent] = _move_towards(
                                agent_state, tx, ty, action_index)
            elif phase == "craft_plank":
                if plank >= 4:
                    phase = "craft_stick"
                elif wood > 0:
                    actions[target_agent] = action_index["craft_plank"]
                else:
                    phase = "gather_wood"
            elif phase == "craft_stick":
                if stick >= 4:
                    phase = "craft_pickaxe"
                elif plank >= 2:
                    actions[target_agent] = action_index["craft_stick"]
                else:
                    phase = "craft_plank"
            elif phase == "craft_pickaxe":
                if pickaxe > 0 or agent_state.equipped == Item.WOOD_PICKAXE:
                    phase = "move_to_stone"
                elif stick >= 1 and plank >= 1:
                    actions[target_agent] = action_index["craft_wood_pickaxe"]
                else:
                    phase = "craft_plank"
            elif phase == "move_to_stone":
                if stone >= 1:
                    phase = "done"
                else:
                    target = _nearest_adjacent_cell(
                        env.world, agent_state, Block.STONE)
                    if target is not None:
                        tx, ty = target
                        if agent_state.x == tx and agent_state.y == ty:
                            actions[target_agent] = action_index["harvest"]
                        else:
                            actions[target_agent] = _move_towards(
                                agent_state, tx, ty, action_index)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
        assert all(isinstance(value, float) for value in rewards.values())
        if all(terminations.values()) or all(truncations.values()):
            break
        # img = Image.fromarray(env.render())
        # frame_list.append(img)

    # frame_list[0].save("out.gif", save_all=True,
    #                    append_images=frame_list[1:], duration=5, loop=0)
    env.close()


if __name__ == "__main__":
    test_render_human()
