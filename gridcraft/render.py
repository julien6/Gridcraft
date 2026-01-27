from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import GridcraftConfig
from .constants import Block, Item, Terrain
from .world import GridcraftWorld


@dataclass
class RenderAssets:
    terrain: dict[int, np.ndarray]
    blocks: dict[int, np.ndarray]
    agent: np.ndarray
    mob: np.ndarray
    items: dict[int, np.ndarray]


class PygameRenderer:
    def __init__(self, config: GridcraftConfig):
        self.config = config
        self._pygame = None
        self.screen = None
        self.clock = None
        self.assets = None

    def _init_pygame(self) -> None:
        if self._pygame is None:
            import pygame

            pygame.init()
            pygame.display.set_caption("Gridcraft")
            self._pygame = pygame
            width = self.config.width * self.config.tile_size
            height = self.config.height * self.config.tile_size
            self.screen = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
            self.assets = self._load_assets()

    def _load_assets(self) -> RenderAssets:
        def color_tile(rgb: tuple[int, int, int]) -> np.ndarray:
            return np.full((self.config.tile_size, self.config.tile_size, 3), rgb, dtype=np.uint8)

        terrain = {
            Terrain.GRASS: color_tile((92, 168, 68)),
            Terrain.WATER: color_tile((52, 117, 200)),
            Terrain.DIRT: color_tile((134, 96, 67)),
        }
        blocks = {
            Block.EMPTY: None,
            Block.TREE: color_tile((45, 120, 52)),
            Block.STONE: color_tile((120, 120, 120)),
            Block.CRAFTING_TABLE: color_tile((160, 105, 45)),
        }
        items = {
            Item.WOOD: color_tile((140, 100, 60)),
            Item.STONE: color_tile((150, 150, 150)),
            Item.APPLE: color_tile((200, 60, 60)),
        }
        agent = color_tile((230, 220, 80))
        mob = color_tile((120, 200, 120))

        if self.config.asset_path and os.path.isdir(self.config.asset_path):
            # Assets can be overridden by placing PNGs named by enum.
            for terrain_id in Terrain:
                custom = self._load_png("assets" + os.sep +
                                        f"terrain_{terrain_id.name.lower()}.png")
                if custom is not None:
                    terrain[terrain_id] = custom
            for block_id in Block:
                if block_id == Block.EMPTY:
                    continue
                custom = self._load_png(
                    "assets" + os.sep + f"block_{block_id.name.lower()}.png")
                if custom is not None:
                    blocks[block_id] = custom
            custom_agent = self._load_png("assets" + os.sep + "agent.png")
            if custom_agent is not None:
                agent = custom_agent
            custom_mob = self._load_png("assets" + os.sep + "mob.png")
            if custom_mob is not None:
                mob = custom_mob

        return RenderAssets(terrain=terrain, blocks=blocks, agent=agent, mob=mob, items=items)

    def _load_png(self, filename: str) -> np.ndarray | None:
        path = os.path.join(self.config.asset_path or "", filename)
        if not os.path.isfile(path):
            return None
        pygame = self._pygame
        surface = pygame.image.load(path)
        surface = pygame.transform.scale(
            surface, (self.config.tile_size, self.config.tile_size))
        arr = pygame.surfarray.array3d(surface)
        return np.transpose(arr, (1, 0, 2))

    def render(self, world: GridcraftWorld, render_mode: str) -> np.ndarray | None:
        self._init_pygame()
        pygame = self._pygame
        assert pygame is not None
        assert self.screen is not None
        assert self.assets is not None

        frame = np.zeros(
            (self.config.height * self.config.tile_size,
             self.config.width * self.config.tile_size, 3),
            dtype=np.uint8,
        )

        for y in range(self.config.height):
            for x in range(self.config.width):
                terrain_tile = self.assets.terrain[Terrain(
                    world.terrain[y, x])]
                self._blit(frame, terrain_tile, x, y)
                block_id = Block(world.blocks[y, x])
                if block_id != Block.EMPTY:
                    block_tile = self.assets.blocks[block_id]
                    if block_tile is not None:
                        self._blit(frame, block_tile, x, y)

        for item in world.items:
            tile = self.assets.items.get(item.item)
            if tile is not None:
                self._blit(frame, tile, item.x, item.y)

        for mob in world.mobs:
            self._blit(frame, self.assets.mob, mob.x, mob.y)

        for agent in world.agents.values():
            if agent.alive:
                self._blit(frame, self.assets.agent, agent.x, agent.y)

        if render_mode == "human":
            self._pump_events()
            surf = pygame.surfarray.make_surface(
                np.transpose(frame, (1, 0, 2)))
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.config.fps)
            return None
        if render_mode == "rgb_array":
            return frame
        return None

    def _blit(self, frame: np.ndarray, tile: np.ndarray, x: int, y: int) -> None:
        ts = self.config.tile_size
        x0 = x * ts
        y0 = y * ts
        frame[y0: y0 + ts, x0: x0 + ts, :] = tile

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self.screen = None
            self.clock = None
            self.assets = None

    def _pump_events(self) -> None:
        pygame = self._pygame
        assert pygame is not None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
