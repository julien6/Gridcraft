from __future__ import annotations

from dataclasses import dataclass, field

from .constants import Item


@dataclass
class AgentState:
    agent_id: str
    x: int
    y: int
    hp: int
    hunger: int
    inventory: dict[Item, int] = field(default_factory=dict)
    inventory_order: list[Item] = field(default_factory=list)
    equipped: Item | None = None
    alive: bool = True


@dataclass
class MobState:
    mob_id: int
    x: int
    y: int
    hp: int
    alive: bool = True


@dataclass
class ItemDrop:
    item: Item
    count: int
    x: int
    y: int
