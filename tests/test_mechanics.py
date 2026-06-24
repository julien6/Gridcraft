from gridcraft import GridcraftConfig, GridcraftEnv
from gridcraft.constants import ACTION_NAMES, Block, Item, Terrain
from gridcraft.entities import MobState


ACTION = {name: index for index, name in enumerate(ACTION_NAMES)}


def _single_agent_env(**kwargs):
    config_kwargs = {"width": 8, "height": 8,
                     "num_agents": 1, "max_mobs": 0, **kwargs}
    config = GridcraftConfig(**config_kwargs)
    env = GridcraftEnv(config=config)
    env.reset(seed=0)
    env.world.terrain[:] = Terrain.GRASS
    env.world.blocks[:] = Block.EMPTY
    env.world.mobs.clear()
    env.world.items.clear()
    agent = env.world.agents["agent_0"]
    agent.x = 3
    agent.y = 3
    agent.visited_positions = {(agent.x, agent.y)}
    return env, agent


def _step(env, action_name: str):
    return env.step({"agent_0": ACTION[action_name]})


def _reward(env, action_name: str) -> float:
    return _step(env, action_name)[1]["agent_0"]


def test_harvesting_tree_adds_wood_and_direct_apple():
    env, agent = _single_agent_env(tree_apple_drop_chance=1.0)
    env.world.blocks[agent.y, agent.x + 1] = Block.TREE

    _step(env, "harvest")

    assert agent.inventory.get(Item.WOOD, 0) == 1
    assert agent.inventory.get(Item.APPLE, 0) == 1
    assert env.world.blocks[agent.y, agent.x + 1] == Block.EMPTY
    env.close()


def test_tree_harvest_reward_includes_wood_and_apple_bonus():
    env, agent = _single_agent_env(tree_apple_drop_chance=1.0)
    env.world.blocks[agent.y, agent.x + 1] = Block.TREE

    reward = _reward(env, "harvest")

    assert reward == (
        env.config.harvest_wood_reward
        + env.config.harvest_tree_apple_reward
        + env.config.survival_reward
    )
    env.close()


def test_harvesting_tree_can_skip_apple_drop():
    env, agent = _single_agent_env(tree_apple_drop_chance=0.0)
    env.world.blocks[agent.y, agent.x + 1] = Block.TREE

    _step(env, "harvest")

    assert agent.inventory.get(Item.WOOD, 0) == 1
    assert agent.inventory.get(Item.APPLE, 0) == 0
    env.close()


def test_stone_harvest_reward_is_higher_than_tree_wood_reward():
    tree_env, tree_agent = _single_agent_env(tree_apple_drop_chance=0.0)
    tree_env.world.blocks[tree_agent.y, tree_agent.x + 1] = Block.TREE
    tree_reward = _reward(tree_env, "harvest")
    tree_env.close()

    stone_env, stone_agent = _single_agent_env()
    stone_agent.equipped = Item.WOOD_PICKAXE
    stone_env.world.blocks[stone_agent.y, stone_agent.x + 1] = Block.STONE
    stone_reward = _reward(stone_env, "harvest")

    assert stone_reward > tree_reward
    stone_env.close()


def test_eat_does_not_consume_apple_when_hunger_full():
    env, agent = _single_agent_env()
    agent.inventory[Item.APPLE] = 1
    agent.hunger = env.config.hunger_max

    _step(env, "eat")

    assert agent.inventory[Item.APPLE] == 1
    assert agent.hunger == env.config.hunger_max
    env.close()


def test_eating_apple_has_reward_only_when_hunger_is_restored():
    env, agent = _single_agent_env()
    agent.inventory[Item.APPLE] = 1
    agent.hunger = env.config.hunger_max

    full_hunger_reward = _reward(env, "eat")
    assert full_hunger_reward == env.config.survival_reward

    agent.hunger = env.config.hunger_max - 1
    restored_reward = _reward(env, "eat")

    assert restored_reward == env.config.eat_apple_reward + env.config.survival_reward
    env.close()


def test_eat_consumes_apple_and_restores_hunger_when_not_full():
    env, agent = _single_agent_env()
    agent.inventory[Item.APPLE] = 1
    agent.hunger = 10

    _step(env, "eat")

    assert agent.inventory[Item.APPLE] == 0
    assert agent.hunger == 16
    env.close()


def test_stay_does_not_reduce_hunger():
    env, agent = _single_agent_env(hunger_decay_ticks=1)
    agent.hunger = 10

    for _ in range(8):
        _step(env, "stay")

    assert agent.hunger == 10
    env.close()


def test_new_cell_reward_encourages_exploration_without_repeated_cell_bonus():
    env, agent = _single_agent_env()

    first_move_reward = _reward(env, "move_e")
    return_to_start_reward = _reward(env, "move_w")

    assert first_move_reward == env.config.new_cell_reward + env.config.survival_reward
    assert return_to_start_reward == env.config.survival_reward
    assert (agent.x, agent.y) == (3, 3)
    env.close()


def test_successful_moves_reduce_hunger_on_interval():
    env, agent = _single_agent_env(move_hunger_cost_interval=5)
    agent.hunger = 10
    moves = ["move_e", "move_w", "move_e", "move_w", "move_e"]

    for action_name in moves:
        _step(env, action_name)

    assert agent.hunger == 9
    env.close()


def test_successful_harvests_reduce_hunger_on_interval():
    env, agent = _single_agent_env(
        harvest_hunger_cost_interval=2,
        tree_apple_drop_chance=0.0,
    )
    agent.hunger = 10

    env.world.blocks[agent.y, agent.x + 1] = Block.TREE
    _step(env, "harvest")
    assert agent.hunger == 10

    env.world.blocks[agent.y, agent.x + 1] = Block.TREE
    _step(env, "harvest")
    assert agent.hunger == 9
    env.close()


def test_craft_reward_increases_with_task_hierarchy():
    env, agent = _single_agent_env()
    agent.inventory[Item.WOOD] = 1
    plank_reward = _reward(env, "craft_plank")

    agent.inventory[Item.PLANK] = 2
    stick_reward = _reward(env, "craft_stick")

    agent.inventory[Item.STICK] = 1
    agent.inventory[Item.PLANK] = 1
    wood_tool_reward = _reward(env, "craft_wood_pickaxe")

    agent.inventory[Item.STICK] = 1
    agent.inventory[Item.STONE] = 1
    stone_tool_reward = _reward(env, "craft_stone_sword")

    assert plank_reward < stick_reward < wood_tool_reward < stone_tool_reward
    env.close()


def test_failed_craft_gets_only_survival_reward():
    env, _ = _single_agent_env()

    reward = _reward(env, "craft_stone_pickaxe")

    assert reward == env.config.survival_reward
    env.close()


def test_blocked_moves_do_not_reduce_hunger():
    env, agent = _single_agent_env(move_hunger_cost_interval=1)
    agent.hunger = 10
    env.world.terrain[agent.y, agent.x + 1] = Terrain.WATER

    for _ in range(3):
        _step(env, "move_e")

    assert agent.x == 3
    assert agent.hunger == 10
    env.close()


def test_mob_kill_reward_is_higher_than_attack_hit_reward():
    hit_env, hit_agent = _single_agent_env(mob_move_prob=0.0)
    hit_env.world.mobs.append(
        MobState(mob_id=1, x=hit_agent.x + 1, y=hit_agent.y, hp=10))
    hit_reward = _reward(hit_env, "attack")
    hit_env.close()

    kill_env, kill_agent = _single_agent_env(mob_move_prob=0.0)
    kill_env.world.mobs.append(
        MobState(mob_id=1, x=kill_agent.x + 1, y=kill_agent.y, hp=2))
    kill_reward = _reward(kill_env, "attack")

    assert kill_reward > hit_reward
    kill_env.close()


def test_successful_attacks_reduce_hunger_on_interval():
    env, agent = _single_agent_env(
        attack_hunger_cost_interval=2,
        mob_move_prob=0.0,
    )
    agent.hunger = 10
    env.world.mobs.append(MobState(mob_id=1, x=agent.x + 1, y=agent.y, hp=10))

    _step(env, "attack")
    assert agent.hunger == 10

    _step(env, "attack")
    assert agent.hunger == 9
    env.close()


def test_missed_attacks_do_not_reduce_hunger_or_block_future_damage():
    env, agent = _single_agent_env(
        attack_hunger_cost_interval=1,
        mob_move_prob=0.0,
    )
    agent.hunger = 10

    _step(env, "attack")

    assert agent.hunger == 10
    assert agent.last_attack_step == -1
    env.close()


def test_starvation_reduces_hp_to_minimum_without_killing_agent():
    env, agent = _single_agent_env(
        hunger_decay_ticks=1,
        starvation_damage=1,
        starvation_min_hp=1,
    )
    agent.hunger = 0
    agent.hp = 3

    for _ in range(5):
        _step(env, "stay")

    assert agent.hp == 1
    assert agent.alive
    env.close()


def test_zombie_can_kill_starving_agent_at_minimum_hp():
    env, agent = _single_agent_env(
        hunger_decay_ticks=1,
        starvation_min_hp=1,
        mob_move_prob=0.0,
    )
    agent.hunger = 0
    agent.hp = 1
    env.world.mobs.append(MobState(mob_id=1, x=agent.x + 1, y=agent.y, hp=10))

    _step(env, "stay")

    assert agent.hp <= 0
    assert not agent.alive
    env.close()


def test_full_hunger_regenerates_hp_progressively():
    env, agent = _single_agent_env(health_regen_ticks=2)
    agent.hunger = env.config.hunger_max
    agent.hp = env.config.hp_max - 2

    _step(env, "stay")
    assert agent.hp == env.config.hp_max - 2

    _step(env, "stay")
    assert agent.hp == env.config.hp_max - 1
    env.close()


def test_successful_attack_blocks_mob_damage_this_step():
    env, agent = _single_agent_env(mob_move_prob=0.0)
    agent.hp = 10
    env.world.mobs.append(MobState(mob_id=1, x=agent.x + 1, y=agent.y, hp=10))

    _step(env, "attack")

    assert agent.hp == 10
    assert env.world.mobs[0].hp == 8
    env.close()


def test_mob_apple_drop_still_uses_item_drop_chance_on_ground():
    env, agent = _single_agent_env(
        item_drop_chance=1.0,
        mob_move_prob=0.0,
    )
    agent.equipped = Item.STONE_SWORD
    env.world.mobs.append(MobState(mob_id=1, x=agent.x + 1, y=agent.y, hp=4))

    _step(env, "attack")

    assert agent.inventory.get(Item.APPLE, 0) == 0
    assert len(env.world.items) == 1
    assert env.world.items[0].item == Item.APPLE
    assert (env.world.items[0].x, env.world.items[0].y) == (agent.x + 1, agent.y)
    env.close()


def test_mob_bfs_moves_toward_agent_around_obstacle():
    env, agent = _single_agent_env(max_mobs=1, mob_move_prob=1.0, mob_aggro_radius=6)
    agent.x = 4
    agent.y = 2
    mob = MobState(mob_id=1, x=1, y=2, hp=10)
    env.world.mobs.append(mob)
    env.world.blocks[2, 2] = Block.STONE

    before = abs(agent.x - mob.x) + abs(agent.y - mob.y)
    for _ in range(4):
        _step(env, "stay")
    after = abs(agent.x - mob.x) + abs(agent.y - mob.y)

    assert after < before
    env.close()
