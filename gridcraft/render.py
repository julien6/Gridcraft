from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import GridcraftConfig
from .constants import Block, EntityType, Item, ITEM_NAMES, Terrain
from .entities import AgentState
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
    unknown: np.ndarray
    unknown_item: np.ndarray
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
        self._scaled_tile_cache: dict[tuple[str, int, int], np.ndarray] = {}

    def _init_pygame(self) -> None:
        if self._pygame is None:
            import pygame

            pygame.init()
            pygame.display.set_caption("Gridcraft")
            self._pygame = pygame
            width = self._frame_width()
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

        def unknown_tile() -> np.ndarray:
            pygame = self._pygame
            assert pygame is not None
            ts = self.config.tile_size
            font = pygame.font.Font(None, max(14, ts // 2 + 4))
            surface = pygame.Surface((ts, ts), pygame.SRCALPHA)
            surface.fill((245, 245, 245, 255))
            pygame.draw.rect(surface, (160, 160, 160, 255), surface.get_rect(), width=max(1, ts // 16))
            text_surface = font.render("?", True, (50, 50, 50))
            rect = text_surface.get_rect(center=(ts // 2, ts // 2))
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
        unknown = unknown_tile()
        unknown_item = unknown
        agent_labels = {idx: label_tile(str(idx))
                        for idx in range(self.config.num_agents)}
        if self.config.asset_path is None:
            self.config.asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../gridcraft/assets")
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
            custom_unknown = self._load_png("interrogation_block.png")
            if custom_unknown is not None:
                unknown = custom_unknown
            custom_unknown_item = self._load_png("interrogation_item.png")
            if custom_unknown_item is not None:
                unknown_item = custom_unknown_item

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
            unknown=unknown,
            unknown_item=unknown_item,
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

    def render(
        self,
        world: GridcraftWorld | None,
        render_mode: str,
        tabular_observations: object | None = None,
        overlay_info: object | None = None,
    ) -> np.ndarray | None:
        self._init_pygame()
        pygame = self._pygame
        assert pygame is not None
        assert self.screen is not None
        assert self.assets is not None

        if tabular_observations is not None and world is None:
            frame = self._render_tabular_frame(tabular_observations)
            self._draw_overlay_info(frame, overlay_info)
            return self._present_frame(frame, render_mode)

        if world is None:
            raise ValueError("world is required when tabular_observations is not provided")
        observations = world.observations()
        frame = self._render_world_frame(world, observations, tabular_observations)
        self._draw_overlay_info(frame, overlay_info)
        return self._present_frame(frame, render_mode)

    def _render_world_frame(
        self,
        world: GridcraftWorld,
        observations: dict[str, dict],
        extra_tabular_observations: object | None = None,
    ) -> np.ndarray:
        grid_width = self.config.width * self.config.tile_size
        base_width = grid_width + self._inventory_panel_width()
        if extra_tabular_observations is not None:
            frame_width = base_width + self._inventory_panel_width()
        else:
            frame_width = base_width
        frame = np.zeros(
            (self.config.height * self.config.tile_size,
             frame_width, 3),
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
            self._draw_ground_item(frame, item.item, item.x, item.y)

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

        self._draw_inventories(frame, world, observations)

        if extra_tabular_observations is not None:
            extra_observations = self._normalize_tabular_observations(extra_tabular_observations)
            fake_world = self._world_from_tabular_observations(extra_observations)
            extra_frame = np.zeros(
                (self.config.height * self.config.tile_size,
                 base_width, 3),
                dtype=np.uint8,
            )
            self._draw_inventories(extra_frame, fake_world, extra_observations)
            frame[:, base_width:, :] = extra_frame[:, grid_width:, :]

        return frame

    def _present_frame(self, frame: np.ndarray, render_mode: str) -> np.ndarray | None:
        pygame = self._pygame
        assert pygame is not None
        assert self.screen is not None
        assert self.clock is not None
        if render_mode == "human":
            if self.screen.get_size() != (frame.shape[1], frame.shape[0]):
                self.screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
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

    def _draw_overlay_info(self, frame: np.ndarray, overlay_info: object | None) -> None:
        if overlay_info is None:
            return
        text = self._format_overlay_info(overlay_info)
        if not text:
            return
        pygame = self._pygame
        assert pygame is not None
        font_size = max(14, self.config.tile_size // 3)
        font = pygame.font.Font(None, font_size)
        surface = font.render(text, True, (255, 255, 255))
        pad_x = max(8, self.config.tile_size // 5)
        pad_y = max(4, self.config.tile_size // 8)
        width = surface.get_width() + pad_x * 2
        height = surface.get_height() + pad_y * 2
        x0 = max(0, (frame.shape[1] - width) // 2)
        y0 = max(0, frame.shape[0] - height - pad_y)
        bg = np.zeros((height, width, 4), dtype=np.uint8)
        bg[:, :, :3] = (0, 0, 0)
        bg[:, :, 3] = 185
        self._blit_at(frame, bg, x0, y0)
        rgb = pygame.surfarray.array3d(surface)
        alpha = pygame.surfarray.array_alpha(surface)
        rgba = np.dstack((rgb, alpha))
        tile = np.transpose(rgba, (1, 0, 2))
        self._blit_at(frame, tile, x0 + pad_x, y0 + pad_y)

    @staticmethod
    def _format_overlay_info(overlay_info: object) -> str:
        if isinstance(overlay_info, str):
            return overlay_info
        if not isinstance(overlay_info, dict):
            return str(overlay_info)
        parts = []
        step = overlay_info.get("step")
        if step is not None:
            parts.append(f"step={step}")
        action = overlay_info.get("action")
        if action is not None:
            if isinstance(action, dict):
                action = ", ".join(f"{agent}={value}" for agent, value in sorted(action.items()))
            parts.append(f"action={action}")
        reward = overlay_info.get("reward")
        if reward is not None:
            if isinstance(reward, float):
                reward = f"{reward:.3f}"
            parts.append(f"reward={reward}")
        done = overlay_info.get("done")
        if done is not None:
            parts.append(f"done={bool(done)}")
        return " | ".join(parts)

    def _render_tabular_frame(self, tabular_observations: object) -> np.ndarray:
        observations = self._normalize_tabular_observations(tabular_observations)
        fake_world = self._world_from_tabular_observations(observations)
        frame = np.zeros(
            (self.config.height * self.config.tile_size,
             self.config.width * self.config.tile_size + self._inventory_panel_width(), 3),
            dtype=np.uint8,
        )
        self._draw_inventories(frame, fake_world, observations)
        return frame

    def _normalize_tabular_observations(self, tabular_observations: object) -> dict[str, dict]:
        if isinstance(tabular_observations, dict):
            normalized = {}
            for agent_id, observation in tabular_observations.items():
                if isinstance(observation, dict):
                    grid = np.asarray(observation.get("grid"), dtype=np.int8)
                    self_vec = np.asarray(
                        observation.get("self", np.zeros(2 + len(Item))),
                        dtype=np.int16,
                    )
                else:
                    grid = np.asarray(observation, dtype=np.int8)
                    self_vec = np.zeros(2 + len(Item), dtype=np.int16)
                    mask = None
                if isinstance(observation, dict):
                    mask = self._normalize_tabular_mask(observation.get("mask"), grid.shape, self_vec.shape)
                normalized[str(agent_id)] = {
                    "grid": self._clip_tabular_grid(grid),
                    "self": self_vec,
                }
                if mask is not None:
                    normalized[str(agent_id)]["mask"] = mask
            return normalized

        arr = np.asarray(tabular_observations, dtype=np.int8)
        if arr.ndim == 3:
            arr = arr.reshape(1, *arr.shape)
        if arr.ndim != 4 or arr.shape[1] != 3:
            raise ValueError("tabular_observations must be a dict or an array shaped (agents, 3, view, view)")
        return {
            f"agent_{idx}": {
                "grid": self._clip_tabular_grid(arr[idx]),
                "self": np.zeros(2 + len(Item), dtype=np.int16),
            }
            for idx in range(arr.shape[0])
        }

    @staticmethod
    def _normalize_tabular_mask(mask: object, grid_shape: tuple[int, ...], self_shape: tuple[int, ...]) -> dict[str, np.ndarray] | None:
        if not isinstance(mask, dict):
            return None
        grid_mask = mask.get("grid")
        self_mask = mask.get("self")
        normalized: dict[str, np.ndarray] = {}
        if grid_mask is not None:
            grid_arr = np.asarray(grid_mask, dtype=np.bool_)
            if grid_arr.shape != grid_shape:
                raise ValueError(f"tabular observation grid mask must be shaped {grid_shape}, got {grid_arr.shape}")
            normalized["grid"] = grid_arr
        if self_mask is not None:
            self_arr = np.asarray(self_mask, dtype=np.bool_)
            if self_arr.shape[0] != self_shape[0]:
                raise ValueError(f"tabular observation self mask must have length {self_shape[0]}, got {self_arr.shape[0]}")
            normalized["self"] = self_arr
        return normalized or None

    @staticmethod
    def _clip_tabular_grid(grid: np.ndarray) -> np.ndarray:
        clipped = np.asarray(grid, dtype=np.int16).copy()
        if clipped.shape[0] != 3:
            raise ValueError("tabular observation grid must be shaped (3, view, view)")
        clipped[0] = np.clip(clipped[0], 0, len(Terrain) - 1)
        clipped[1] = np.clip(clipped[1], 0, len(Block) - 1)
        clipped[2] = np.clip(clipped[2], 0, len(EntityType) - 1)
        return clipped.astype(np.int8)

    def _world_from_tabular_observations(self, observations: dict[str, dict]):
        agents = {}
        for index, agent_id in enumerate(sorted(observations.keys())):
            self_vec = np.asarray(observations[agent_id].get("self", []), dtype=np.int16)
            hp = int(self_vec[0]) if self_vec.size > 0 else self.config.hp_max
            hunger = int(self_vec[1]) if self_vec.size > 1 else self.config.hunger_max
            counts = self_vec[2:2 + len(Item)] if self_vec.size >= 2 + len(Item) else np.zeros(len(Item), dtype=np.int16)
            inventory = {
                Item(item_index): int(count)
                for item_index, count in enumerate(counts)
                if int(count) > 0
            }
            agents[agent_id] = AgentState(
                agent_id=agent_id,
                x=0,
                y=index,
                hp=max(0, hp),
                hunger=max(0, hunger),
                inventory=inventory,
                inventory_order=list(Item),
                equipped=None,
                alive=True,
            )

        class _TabularWorld:
            pass

        fake_world = _TabularWorld()
        fake_world.agents = agents
        return fake_world

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
        inventory_width = cols * self.config.tile_size
        observation_width = self.config.view_size * self._observation_tile_size()
        return padding * 3 + inventory_width + observation_width

    def _frame_width(self) -> int:
        return self.config.width * self.config.tile_size + self._inventory_panel_width()

    def _draw_inventories(
        self,
        frame: np.ndarray,
        world: GridcraftWorld,
        observations: dict[str, dict],
    ) -> None:
        assert self.assets is not None
        ts = self.config.tile_size
        cols = 4
        rows = 3
        padding = max(2, ts // 4)
        observation_tile_size = self._observation_tile_size()
        observation_size = self.config.view_size * observation_tile_size
        inventory_height = padding + ts + padding + rows * ts
        section_height = padding + max(inventory_height, observation_size) + padding
        panel_x = self.config.width * ts + padding
        y = padding

        agent_ids = sorted(world.agents.keys())
        id_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
        for agent_id in agent_ids:
            agent = world.agents[agent_id]
            if not agent.alive:
                continue
            observation_data = observations.get(agent_id, {})
            self_mask = observation_data.get("mask", {}).get("self")
            self._blit_at(frame, self.assets.agent, panel_x, y)
            label_idx = id_to_index.get(agent_id)
            if label_idx is not None:
                label_tile = self.assets.agent_labels.get(label_idx)
                if label_tile is not None:
                    self._blit_at(frame, label_tile, panel_x, y)
            self._draw_vitals(frame, agent, panel_x, y, self_mask)
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
                    self_idx = 2 + int(item_id)
                    item_known = self_mask is None or self_idx >= len(self_mask) or bool(self_mask[self_idx])
                    count = agent.inventory.get(item_id, 0)
                    if not item_known:
                        unknown_tile = self._scaled_cached_tile("unknown_item", 0, self.assets.unknown_item, ts)
                        self._blit_at(frame, unknown_tile, slot_x, slot_y)
                    elif count > 0:
                        tile = self.assets.items.get(item_id)
                        if tile is not None:
                            self._blit_at(frame, tile, slot_x, slot_y)
                        count_tile = self._text_tile(str(count), (255, 0, 0))
                        self._blit_at(frame, count_tile, slot_x, slot_y)
                if idx == selected_index:
                    self._blit_at(frame, self.assets.ui_selected,
                                  slot_x, slot_y)
            observation = observation_data.get("grid")
            if observation is not None:
                observation_mask = observation_data.get("mask", {}).get("grid")
                observation_x = panel_x + cols * ts + padding
                self._draw_spatial_observation(
                    frame,
                    observation,
                    observation_x,
                    slots_y,
                    observation_tile_size,
                    observation_mask,
                )
            y += section_height

    def _observation_tile_size(self) -> int:
        visible_rows = 3
        return max(2, (visible_rows * self.config.tile_size) // self.config.view_size)

    def _draw_spatial_observation(
        self,
        frame: np.ndarray,
        observation: np.ndarray,
        x0: int,
        y0: int,
        tile_size: int,
        mask: np.ndarray | None = None,
    ) -> None:
        assert self.assets is not None
        size = observation.shape[1]
        for gy in range(size):
            for gx in range(size):
                px = x0 + gx * tile_size
                py = y0 + gy * tile_size
                if mask is not None and not bool(np.any(mask[:, gy, gx])):
                    unknown_tile = self._scaled_cached_tile("unknown", 0, self.assets.unknown, tile_size)
                    self._blit_at(frame, unknown_tile, px, py)
                    continue
                terrain_tile = self._scaled_terrain_tile(
                    Terrain(int(observation[0, gy, gx])),
                    tile_size,
                )
                self._blit_at(frame, terrain_tile, px, py)
                block_id = Block(int(observation[1, gy, gx]))
                if block_id != Block.EMPTY:
                    block_tile = self._scaled_block_tile(block_id, tile_size)
                    if block_tile is not None:
                        self._blit_at(frame, block_tile, px, py)
                entity_id = EntityType(int(observation[2, gy, gx]))
                if entity_id == EntityType.ITEM:
                    self._draw_observation_ground_item(frame, px, py, tile_size)
                else:
                    entity_tile = self._scaled_entity_tile(entity_id, tile_size)
                    if entity_tile is not None:
                        self._blit_at(frame, entity_tile, px, py)

    def _scaled_terrain_tile(self, terrain_id: Terrain, size: int) -> np.ndarray:
        return self._scaled_cached_tile("terrain", int(terrain_id), self.assets.terrain[terrain_id], size)

    def _scaled_block_tile(self, block_id: Block, size: int) -> np.ndarray | None:
        tile = self.assets.blocks.get(block_id)
        if tile is None:
            return None
        return self._scaled_cached_tile("block", int(block_id), tile, size)

    def _scaled_entity_tile(self, entity_id: EntityType, size: int) -> np.ndarray | None:
        if entity_id == EntityType.AGENT:
            return self._scaled_cached_tile("entity", int(entity_id), self.assets.agent, size)
        if entity_id == EntityType.MOB:
            return self._scaled_cached_tile("entity", int(entity_id), self.assets.mob, size)
        return None

    def _scaled_cached_tile(
        self,
        group: str,
        tile_id: int,
        tile: np.ndarray,
        size: int,
    ) -> np.ndarray:
        key = (group, tile_id, size)
        cached = self._scaled_tile_cache.get(key)
        if cached is not None:
            return cached
        scaled = self._scaled_ui_tile(tile, size)
        self._scaled_tile_cache[key] = scaled
        return scaled

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

    def _draw_ground_item(self, frame: np.ndarray, item_id: Item, grid_x: int, grid_y: int) -> None:
        ts = self.config.tile_size
        size = max(1, int(ts * 0.6))
        tile = self._scaled_item_tile(item_id, size)
        if tile is None:
            return
        x0 = grid_x * ts + (ts - size) // 2
        y0 = grid_y * ts + (ts - size) // 2
        self._blit_at(frame, tile, x0, y0)

    def _draw_observation_ground_item(
        self,
        frame: np.ndarray,
        x0: int,
        y0: int,
        tile_size: int,
    ) -> None:
        size = max(1, int(tile_size * 0.6))
        tile = self._scaled_item_tile(Item.WOOD, size)
        if tile is None:
            return
        item_x = x0 + (tile_size - size) // 2
        item_y = y0 + (tile_size - size) // 2
        self._blit_at(frame, tile, item_x, item_y)

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

    def _draw_vitals(self, frame: np.ndarray, agent, panel_x: int, panel_y: int, self_mask: np.ndarray | None = None) -> None:
        assert self.assets is not None
        ts = self.config.tile_size
        icon_size = max(1, ts // 2)
        x0 = panel_x + ts + 4
        y0 = panel_y - 2
        heart = self.assets.ui_heart
        hunger = self.assets.ui_hunger

        hp_known = self_mask is None or len(self_mask) < 1 or bool(self_mask[0])
        hunger_known = self_mask is None or len(self_mask) < 2 or bool(self_mask[1])

        if hp_known and heart is not None:
            heart_icon = self._scaled_ui_tile(heart, icon_size)
            for i in range(max(0, agent.hp)):
                self._blit_at(frame, heart_icon, x0 + i * (icon_size + 2), y0)
        y1 = y0 + icon_size + 4
        if hunger_known and hunger is not None:
            hunger_icon = self._scaled_ui_tile(hunger, icon_size)
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
            self._scaled_tile_cache.clear()
            self._scaled_item_cache.clear()
            self._text_cache.clear()

    def _pump_events(self) -> None:
        pygame = self._pygame
        assert pygame is not None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
