# Gridcraft

Gridcraft is a simple 2D top-down gridworld inspired by Minecraft, built as a **multi-agent** PettingZoo-style environment with survival, hunger, hostile mobs, harvesting, and crafting. It includes a minimal **PyGame renderer** with a built-in fallback color palette for tiles and entities.

## Features

* Parallel multi-agent API (PettingZoo-style)
* Survival mechanics: HP + hunger decay
* Hostile mobs with simple aggro AI
* Harvesting trees/stone, crafting basic tools/weapons
* Local observations (grid window + agent state vector)
* Deterministic RNG via seeds
* Optional PyGame renderer (`human` / `rgb_array`)

## Quick start

```python
from gridcraft import GridcraftConfig, GridcraftEnv

config = GridcraftConfig(width=24, height=24, num_agents=2)

env = GridcraftEnv(config=config, render_mode="human")
obs, infos = env.reset(seed=123)

actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
for _ in range(50):
    obs, rewards, terminations, truncations, infos = env.step(actions)
    env.render()
    if all(terminations.values()) or all(truncations.values()):
        break

env.close()
```

## Rendering assets

To avoid shipping any copyrighted Minecraft textures, the renderer ships with solid-color tiles by default.
You can override assets by providing `GridcraftConfig(asset_path=...)` and placing PNGs in that folder
(e.g., `terrain_grass.png` , `block_tree.png` , `agent.png` , `mob.png` ).

## Notes

* This environment is intentionally small and simple to enable rapid iteration and RL experiments.
* Crafting is enabled everywhere by default; set `craft_anywhere=False` and place crafting tables for stricter rules.
