import pytest

numpy = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")

from gridcraft import GridcraftConfig, GridcraftEnv


def test_smoke_step():
    config = GridcraftConfig(width=12, height=12, num_agents=2, max_steps=20, seed=42)
    env = GridcraftEnv(config=config)
    obs, infos = env.reset(seed=42)
    assert set(obs.keys()) == set(env.possible_agents)
    for _ in range(10):
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        assert all(isinstance(value, float) for value in rewards.values())
        if all(terminations.values()) or all(truncations.values()):
            break
    env.close()


def test_reproducibility():
    config = GridcraftConfig(width=12, height=12, num_agents=2, seed=7)
    env_a = GridcraftEnv(config=config)
    obs_a, _ = env_a.reset(seed=7)
    env_b = GridcraftEnv(config=config)
    obs_b, _ = env_b.reset(seed=7)
    assert numpy.array_equal(obs_a["agent_0"]["grid"], obs_b["agent_0"]["grid"])
    env_a.close()
    env_b.close()
