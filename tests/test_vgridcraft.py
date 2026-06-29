import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vGridcraft"))

from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv
from vgridcraft.env import BLOCK_EMPTY, BLOCK_TREE, ITEM_APPLE, ITEM_PLANK, ITEM_STONE, ITEM_WOOD, TERRAIN_GRASS, TERRAIN_WATER


def _env(**kwargs):
    config = VGridcraftConfig(width=8, height=8, max_mobs=0, num_agents=1, max_steps=20, **kwargs)
    env = VectorizedGridcraftEnv(num_envs=1, num_agents=1, device="cpu", seed=0, config=config)
    env.terrain[:] = TERRAIN_GRASS
    env.blocks[:] = BLOCK_EMPTY
    env.mob_alive[:] = False
    env.item_alive[:] = False
    env.agent_x[:] = 3
    env.agent_y[:] = 3
    env.visited[:] = False
    env.visited[0, 0, 3, 3] = True
    env.hp[:] = config.hp_max
    env.hunger[:] = config.hunger_max
    env.inventory[:] = 0
    return env


def _step(env, action):
    return env.step(torch.tensor([[action]], dtype=torch.long))


def test_vgridcraft_successful_move_rewards_new_cell():
    env = _env()
    _, reward, done, truncated, _ = _step(env, 4)

    assert int(env.agent_x[0, 0]) == 4
    assert int(env.agent_y[0, 0]) == 3
    assert float(reward[0, 0]) == pytest.approx(env.config.new_cell_reward + env.config.survival_reward)
    assert not bool(done[0])
    assert not bool(truncated[0])


def test_vgridcraft_blocked_move_does_not_reduce_hunger():
    env = _env(move_hunger_cost_interval=1)
    env.terrain[0, 3, 4] = TERRAIN_WATER

    for _ in range(3):
        _step(env, 4)

    assert int(env.agent_x[0, 0]) == 3
    assert int(env.hunger[0, 0]) == 20


def test_vgridcraft_harvest_tree_adds_wood_and_apple():
    env = _env(tree_apple_drop_chance=1.0)
    env.blocks[0, 3, 4] = BLOCK_TREE

    _, reward, _, _, _ = _step(env, 5)

    assert int(env.blocks[0, 3, 4]) == BLOCK_EMPTY
    assert int(env.inventory[0, 0, ITEM_WOOD]) == 1
    assert int(env.inventory[0, 0, ITEM_APPLE]) == 1
    assert float(reward[0, 0]) == pytest.approx(env.config.harvest_wood_reward + env.config.harvest_tree_apple_reward + env.config.survival_reward)


def test_vgridcraft_craft_plank_consumes_wood():
    env = _env()
    env.inventory[0, 0, ITEM_WOOD] = 1

    _, reward, _, _, _ = _step(env, 9)

    assert int(env.inventory[0, 0, ITEM_WOOD]) == 0
    assert int(env.inventory[0, 0, ITEM_PLANK]) == 2
    assert float(reward[0, 0]) == pytest.approx(env.config.craft_plank_reward + env.config.survival_reward)


def test_vgridcraft_task_level_tracks_progress_and_info():
    env = _env(tree_apple_drop_chance=0.0)

    _, _, _, _, info = _step(env, 4)
    assert int(env.task_level_max[0, 0]) == 1
    assert int(info["task_level_max"][0, 0]) == 1

    env.blocks[0, 3, 5] = BLOCK_TREE
    _, _, _, _, info = _step(env, 5)
    assert int(env.task_level_max[0, 0]) == 2
    assert int(info["task_level_max"][0, 0]) == 2

    _, _, _, _, info = _step(env, 9)
    assert int(env.task_level_max[0, 0]) == 3
    assert int(info["task_level_max"][0, 0]) == 3


def test_vgridcraft_pickup_collects_adjacent_item():
    env = _env()
    env.item_alive[0, 0] = True
    env.item_x[0, 0] = 4
    env.item_y[0, 0] = 3
    env.item_type[0, 0] = ITEM_STONE
    env.item_count[0, 0] = 2

    _, reward, _, _, _ = _step(env, 6)

    assert int(env.inventory[0, 0, ITEM_STONE]) == 2
    assert not bool(env.item_alive[0, 0])
    assert float(reward[0, 0]) == pytest.approx(env.config.pickup_item_reward * 2 + env.config.survival_reward)
    assert int(env.task_level_max[0, 0]) == 6


def test_vgridcraft_pickup_ignores_item_on_agent_cell():
    env = _env()
    env.item_alive[0, 0] = True
    env.item_x[0, 0] = 3
    env.item_y[0, 0] = 3
    env.item_type[0, 0] = ITEM_STONE
    env.item_count[0, 0] = 2

    _, reward, _, _, _ = _step(env, 6)

    assert int(env.inventory[0, 0, ITEM_STONE]) == 0
    assert bool(env.item_alive[0, 0])
    assert float(reward[0, 0]) == pytest.approx(env.config.survival_reward)


def test_vgridcraft_info_reports_task_level_max():
    env = _env()
    env.inventory[0, 0, ITEM_WOOD] = 1

    _, _, _, _, info = _step(env, 9)

    assert int(info["task_level_max"][0, 0]) == 3


def test_vgridcraft_render_rgb_array_uses_gridcraft_renderer():
    pytest.importorskip("pygame")
    env = _env()

    frame = env.render(env_index=0, mode="rgb_array")

    assert frame.ndim == 3
    assert frame.shape[-1] == 3
    assert frame.shape[0] == env.config.height * 48
    assert frame.shape[1] > env.config.width * 48
    env.close()
