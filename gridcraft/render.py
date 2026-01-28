from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import GridcraftConfig
from .constants import Block, Item, ITEM_NAMES, Terrain
from .world import GridcraftWorld


@dataclass
class RenderAssets:
    terrain: dict[int, np.ndarray]
    blocks: dict[int, np.ndarray]
    agent: np.ndarray
    mob: np.ndarray
    items: dict[int, np.ndarray]
    ui_slot: np.ndarray
    ui_selected: np.ndarray
    ui_heart: np.ndarray | None
    ui_hunger: np.ndarray | None
    agent_labels: dict[int, np.ndarray]


class PygameRenderer:
    def __init__(self, config: GridcraftConfig):
        self.config = config
        self._pygame = None
        self.screen = None
        self.clock = None
        self.assets = None
        self._text_cache: dict[tuple[str,
                                     tuple[int, int, int]], np.ndarray] = {}
        self._scaled_item_cache: dict[tuple[int, int], np.ndarray] = {}

    def _init_pygame(self) -> None:
        if self._pygame is None:
            import pygame

            pygame.init()
            pygame.display.set_caption("Gridcraft")
            self._pygame = pygame
            width = self.config.width * self.config.tile_size + self._inventory_panel_width()
            height = self.config.height * self.config.tile_size
            self.screen = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
            self.assets = self._load_assets()

    def _load_assets(self) -> RenderAssets:
        def color_tile(rgb: tuple[int, int, int]) -> np.ndarray:
            return np.full((self.config.tile_size, self.config.tile_size, 3), rgb, dtype=np.uint8)

        def label_tile(text: str) -> np.ndarray:
            pygame = self._pygame
            assert pygame is not None
            ts = self.config.tile_size
            font_size = max(12, ts // 2 + 2)
            font = pygame.font.Font(None, font_size)
            surface = pygame.Surface((ts, ts), pygame.SRCALPHA)
            text_surface = font.render(text, True, (255, 0, 0))
            shadow = font.render(text, True, (0, 0, 0))
            rect = text_surface.get_rect()
            rect.bottomright = (ts - 2, ts - 2)
            shadow_rect = shadow.get_rect()
            shadow_rect.bottomright = (rect.right + 1, rect.bottom + 1)
            surface.blit(shadow, shadow_rect)
            surface.blit(text_surface, rect)
            rgb = pygame.surfarray.array3d(surface)
            alpha = pygame.surfarray.array_alpha(surface)
            rgba = np.dstack((rgb, alpha))
            return np.transpose(rgba, (1, 0, 2))

        def slot_tile() -> np.ndarray:
            ts = self.config.tile_size
            tile = np.full((ts, ts, 3), (60, 60, 60), dtype=np.uint8)
            tile[0, :, :] = (120, 120, 120)
            tile[-1, :, :] = (120, 120, 120)
            tile[:, 0, :] = (120, 120, 120)
            tile[:, -1, :] = (120, 120, 120)
            return tile

        def selected_tile() -> np.ndarray:
            ts = self.config.tile_size
            tile = np.zeros((ts, ts, 4), dtype=np.uint8)
            tile[0, :, :3] = (230, 220, 80)
            tile[-1, :, :3] = (230, 220, 80)
            tile[:, 0, :3] = (230, 220, 80)
            tile[:, -1, :3] = (230, 220, 80)
            tile[0, :, 3] = 220
            tile[-1, :, 3] = 220
            tile[:, 0, 3] = 220
            tile[:, -1, 3] = 220
            return tile

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
        item_colors = {
            Item.WOOD: (140, 100, 60),
            Item.PLANK: (160, 120, 70),
            Item.STICK: (170, 140, 90),
            Item.STONE: (150, 150, 150),
            Item.WOOD_SWORD: (200, 180, 120),
            Item.STONE_SWORD: (130, 130, 130),
            Item.WOOD_PICKAXE: (190, 160, 100),
            Item.STONE_PICKAXE: (120, 120, 120),
            Item.APPLE: (200, 60, 60),
        }
        items = {
            item_id: color_tile(color) for item_id, color in item_colors.items()
        }
        agent = color_tile((230, 220, 80))
        mob = color_tile((120, 200, 120))
        ui_slot = slot_tile()
        ui_selected = selected_tile()
        ui_heart = None
        ui_hunger = None
        agent_labels = {idx: label_tile(str(idx))
                        for idx in range(self.config.num_agents)}

        if self.config.asset_path and os.path.isdir(self.config.asset_path):
            # Assets can be overridden by placing PNGs named by enum.
            for terrain_id in Terrain:
                custom = self._load_png(
                    f"terrain_{terrain_id.name.lower()}.png")
                if custom is not None:
                    terrain[terrain_id] = custom
            for block_id in Block:
                if block_id == Block.EMPTY:
                    continue
                custom = self._load_png(
                    f"block_{block_id.name.lower()}.png")
                if custom is not None:
                    blocks[block_id] = custom
            for item_id in Item:
                filename = f"item_{ITEM_NAMES[item_id]}.png"
                custom = self._load_png(filename)
                if custom is not None:
                    items[item_id] = custom
            custom_agent = self._load_png("entity_agent.png")
            if custom_agent is not None:
                agent = custom_agent
            custom_mob = self._load_png("entity_mob.png")
            if custom_mob is not None:
                mob = custom_mob
            custom_slot = self._load_png("ui_inventory_slot.png")
            if custom_slot is not None:
                ui_slot = custom_slot
            custom_selected = self._load_png("ui_selected.png")
            if custom_selected is not None:
                ui_selected = custom_selected
            ui_heart = self._load_png("ui_heart.png")
            ui_hunger = self._load_png("ui_hunger.png")

        return RenderAssets(
            terrain=terrain,
            blocks=blocks,
            agent=agent,
            mob=mob,
            items=items,
            ui_slot=ui_slot,
            ui_selected=ui_selected,
            ui_heart=ui_heart,
            ui_hunger=ui_hunger,
            agent_labels=agent_labels,
        )

    def _text_tile(self, text: str, color: tuple[int, int, int]) -> np.ndarray:
        key = (text, color)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        pygame = self._pygame
        assert pygame is not None
        ts = self.config.tile_size
        font_size = max(10, ts // 2)
        font = pygame.font.Font(None, font_size)
        surface = pygame.Surface((ts, ts), pygame.SRCALPHA)
        text_surface = font.render(text, True, color)
        shadow = font.render(text, True, (0, 0, 0))
        rect = text_surface.get_rect()
        rect.bottomright = (ts - 2, ts - 2)
        shadow_rect = shadow.get_rect()
        shadow_rect.bottomright = (rect.right + 1, rect.bottom + 1)
        surface.blit(shadow, shadow_rect)
        surface.blit(text_surface, rect)
        rgb = pygame.surfarray.array3d(surface)
        alpha = pygame.surfarray.array_alpha(surface)
        rgba = np.dstack((rgb, alpha))
        tile = np.transpose(rgba, (1, 0, 2))
        self._text_cache[key] = tile
        return tile

    def _load_png(self, filename: str) -> np.ndarray | None:
        path = os.path.join(self.config.asset_path or "", filename)
        if not os.path.isfile(path):
            return None
        pygame = self._pygame
        surface = pygame.image.load(path).convert_alpha()
        surface = pygame.transform.scale(
            surface, (self.config.tile_size, self.config.tile_size))
        rgb = pygame.surfarray.array3d(surface)
        alpha = pygame.surfarray.array_alpha(surface)
        rgba = np.dstack((rgb, alpha))
        return np.transpose(rgba, (1, 0, 2))

    def render(self, world: GridcraftWorld, render_mode: str) -> np.ndarray | None:
        self._init_pygame()
        pygame = self._pygame
        assert pygame is not None
        assert self.screen is not None
        assert self.assets is not None

        frame = np.zeros(
            (self.config.height * self.config.tile_size,
             self.config.width * self.config.tile_size + self._inventory_panel_width(), 3),
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

        agent_ids = sorted(world.agents.keys())
        id_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
        for agent_id, agent in world.agents.items():
            if agent.alive:
                self._blit(frame, self.assets.agent, agent.x, agent.y)
                label_idx = id_to_index.get(agent_id)
                if label_idx is not None:
                    label_tile = self.assets.agent_labels.get(label_idx)
                    if label_tile is not None:
                        self._blit(frame, label_tile, agent.x, agent.y)
                self._draw_held_item(frame, agent, agent.x, agent.y)

        self._draw_inventories(frame, world)

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
        self._blit_at(frame, tile, x * ts, y * ts)

    def _blit_at(self, frame: np.ndarray, tile: np.ndarray, x0: int, y0: int) -> None:
        frame_h, frame_w, _ = frame.shape
        tile_h, tile_w, _ = tile.shape
        if x0 >= frame_w or y0 >= frame_h:
            return
        if x0 + tile_w <= 0 or y0 + tile_h <= 0:
            return

        x_start = max(0, -x0)
        y_start = max(0, -y0)
        x_end = min(tile_w, frame_w - x0)
        y_end = min(tile_h, frame_h - y0)
        if x_end <= x_start or y_end <= y_start:
            return

        tile_view = tile[y_start:y_end, x_start:x_end, :]
        target = frame[y0 + y_start:y0 + y_end, x0 + x_start:x0 + x_end, :]
        if tile_view.shape[2] == 4:
            src_rgb = tile_view[:, :, :3].astype(np.float32)
            alpha = tile_view[:, :, 3:4].astype(np.float32) / 255.0
            dst_rgb = target.astype(np.float32)
            blended = src_rgb * alpha + dst_rgb * (1.0 - alpha)
            target[:] = blended.astype(np.uint8)
        else:
            target[:] = tile_view

    def _inventory_panel_width(self) -> int:
        cols = 4
        padding = max(2, self.config.tile_size // 4)
        return padding * 2 + cols * self.config.tile_size

    def _draw_inventories(self, frame: np.ndarray, world: GridcraftWorld) -> None:
        assert self.assets is not None
        ts = self.config.tile_size
        items = []
        cols = 4
        rows = 3
        padding = max(2, ts // 4)
        section_height = padding + ts + padding + rows * ts + padding
        panel_x = self.config.width * ts + padding
        y = padding

        agent_ids = sorted(world.agents.keys())
        id_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
        for agent_id in agent_ids:
            agent = world.agents[agent_id]
            if not agent.alive:
                continue
            self._blit_at(frame, self.assets.agent, panel_x, y)
            label_idx = id_to_index.get(agent_id)
            if label_idx is not None:
                label_tile = self.assets.agent_labels.get(label_idx)
                if label_tile is not None:
                    self._blit_at(frame, label_tile, panel_x, y)
            self._draw_vitals(frame, agent, panel_x, y)
            slots_y = y + ts + padding
            items = agent.inventory_order if agent.inventory_order else list(
                Item)
            selected_index = self._selected_index(agent, items)
            for idx in range(rows * cols):
                col = idx % cols
                row = idx // cols
                slot_x = panel_x + col * ts
                slot_y = slots_y + row * ts
                self._blit_at(frame, self.assets.ui_slot, slot_x, slot_y)
                if idx < len(items):
                    item_id = items[idx]
                    count = agent.inventory.get(item_id, 0)
                    if count > 0:
                        tile = self.assets.items.get(item_id)
                        if tile is not None:
                            self._blit_at(frame, tile, slot_x, slot_y)
                        count_tile = self._text_tile(str(count), (255, 0, 0))
                        self._blit_at(frame, count_tile, slot_x, slot_y)
                if idx == selected_index:
                    self._blit_at(frame, self.assets.ui_selected,
                                  slot_x, slot_y)
            y += section_height

    def _selected_index(self, agent, items: list[Item]) -> int:
        if agent.equipped is not None and agent.equipped in items:
            return items.index(agent.equipped)
        return 0

    def _selected_item(self, agent) -> Item | None:
        items = agent.inventory_order if agent.inventory_order else list(Item)
        if not items:
            return None
        item_id = items[self._selected_index(agent, items)]
        if agent.inventory.get(item_id, 0) > 0:
            return item_id
        return None

    def _scaled_item_tile(self, item_id: Item, size: int) -> np.ndarray | None:
        key = (int(item_id), size)
        cached = self._scaled_item_cache.get(key)
        if cached is not None:
            return cached
        base = self.assets.items.get(item_id)
        if base is None:
            return None
        h, w = base.shape[:2]
        if h == size and w == size:
            self._scaled_item_cache[key] = base
            return base
        y_idx = np.linspace(0, h - 1, size).astype(int)
        x_idx = np.linspace(0, w - 1, size).astype(int)
        scaled = base[np.ix_(y_idx, x_idx)]
        self._scaled_item_cache[key] = scaled
        return scaled

    def _draw_held_item(self, frame: np.ndarray, agent, grid_x: int, grid_y: int) -> None:
        item_id = self._selected_item(agent)
        if item_id is None:
            return
        ts = self.config.tile_size
        size = max(1, ts // 2)
        tile = self._scaled_item_tile(item_id, size)
        if tile is None:
            return
        x0 = grid_x * ts + ts - 2 * size - 2
        y0 = grid_y * ts + (ts - 1*size) // 2
        self._blit_at(frame, tile, x0, y0)

    def _draw_vitals(self, frame: np.ndarray, agent, panel_x: int, panel_y: int) -> None:
        assert self.assets is not None
        ts = self.config.tile_size
        icon_size = max(1, ts // 2)
        x0 = panel_x + ts + 4
        y0 = panel_y - 2
        heart = self.assets.ui_heart
        hunger = self.assets.ui_hunger

        if heart is not None:
            heart_icon = self._scaled_ui_tile(heart, icon_size)
            for i in range(max(0, agent.hp)):
                self._blit_at(frame, heart_icon, x0 + i * (icon_size + 2), y0)
        if hunger is not None:
            hunger_icon = self._scaled_ui_tile(hunger, icon_size)
            y1 = y0 + icon_size + 4
            for i in range(max(0, agent.hunger)):
                self._blit_at(frame, hunger_icon, x0 + i * (icon_size + 2), y1)

    def _scaled_ui_tile(self, tile: np.ndarray, size: int) -> np.ndarray:
        h, w = tile.shape[:2]
        if h == size and w == size:
            return tile
        y_idx = np.linspace(0, h - 1, size).astype(int)
        x_idx = np.linspace(0, w - 1, size).astype(int)
        return tile[np.ix_(y_idx, x_idx)]

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
