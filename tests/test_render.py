import os
import pytest

from gridcraft.config import GridcraftConfig
from gridcraft.env import GridcraftEnv

pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("pygame")


def test_render_rgb_array():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = GridcraftConfig(
        width=16, height=16, num_agents=2, seed=5, tile_size=8)
    env = GridcraftEnv(config=config, render_mode="rgb_array")
    env.reset(seed=5)

    frame = env.render()
    assert frame is not None
    assert frame.shape == (config.height * config.tile_size,
                           config.width * config.tile_size, 3)

    env.close()


def test_render_human():
    config = GridcraftConfig(
        width=12, height=12, num_agents=2, max_steps=20, seed=42)
    env = GridcraftEnv(config=config, render_mode="human")
    obs, infos = env.reset(seed=42)
    assert set(obs.keys()) == set(env.possible_agents)
    for _ in range(100):
        actions = {agent: env.action_spaces[agent].sample()
                   for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
        assert all(isinstance(value, float) for value in rewards.values())
        if all(terminations.values()) or all(truncations.values()):
            break
    env.close()


if __name__ == "__main__":
    test_render_human()
