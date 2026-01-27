import os

import pytest

pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("pygame")

from gridcraft import GridcraftConfig, GridcraftEnv


def test_render_rgb_array():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = GridcraftConfig(width=16, height=16, num_agents=2, seed=5, tile_size=8)
    env = GridcraftEnv(config=config, render_mode="rgb_array")
    env.reset(seed=5)

    frame = env.render()
    assert frame is not None
    assert frame.shape == (config.height * config.tile_size, config.width * config.tile_size, 3)

    env.close()
