from __future__ import annotations

from enum import IntEnum


class Terrain(IntEnum):
    GRASS = 0
    WATER = 1
    DIRT = 2


class Block(IntEnum):
    EMPTY = 0
    TREE = 1
    STONE = 2
    CRAFTING_TABLE = 3


class EntityType(IntEnum):
    NONE = 0
    AGENT = 1
    MOB = 2
    ITEM = 3


class Item(IntEnum):
    WOOD = 0
    PLANK = 1
    STICK = 2
    STONE = 3
    WOOD_SWORD = 4
    STONE_SWORD = 5
    WOOD_PICKAXE = 6
    STONE_PICKAXE = 7
    APPLE = 8


ITEM_NAMES = {
    Item.WOOD: "wood",
    Item.PLANK: "plank",
    Item.STICK: "stick",
    Item.STONE: "stone",
    Item.WOOD_SWORD: "wood_sword",
    Item.STONE_SWORD: "stone_sword",
    Item.WOOD_PICKAXE: "wood_pickaxe",
    Item.STONE_PICKAXE: "stone_pickaxe",
    Item.APPLE: "apple",
}


CRAFTING_RECIPES = {
    "plank": {"inputs": {Item.WOOD: 1}, "outputs": {Item.PLANK: 2}},
    "stick": {"inputs": {Item.PLANK: 2}, "outputs": {Item.STICK: 4}},
    "wood_sword": {"inputs": {Item.STICK: 1, Item.PLANK: 1}, "outputs": {Item.WOOD_SWORD: 1}},
    "stone_sword": {"inputs": {Item.STICK: 1, Item.STONE: 1}, "outputs": {Item.STONE_SWORD: 1}},
    "wood_pickaxe": {"inputs": {Item.STICK: 1, Item.PLANK: 1}, "outputs": {Item.WOOD_PICKAXE: 1}},
    "stone_pickaxe": {"inputs": {Item.STICK: 1, Item.STONE: 1}, "outputs": {Item.STONE_PICKAXE: 1}},
}


ACTION_NAMES = [
    "stay",
    "move_n",
    "move_s",
    "move_w",
    "move_e",
    "harvest",
    "pickup",
    "attack",
    "eat",
    "craft_plank",
    "craft_stick",
    "craft_wood_sword",
    "craft_stone_sword",
    "craft_wood_pickaxe",
    "craft_stone_pickaxe",
]


ACTION_TO_RECIPE = {
    "craft_plank": "plank",
    "craft_stick": "stick",
    "craft_wood_sword": "wood_sword",
    "craft_stone_sword": "stone_sword",
    "craft_wood_pickaxe": "wood_pickaxe",
    "craft_stone_pickaxe": "stone_pickaxe",
}
