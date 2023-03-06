from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys

import logging

logging.basicConfig(filename="agentLog.log", level=logging.INFO, filemode='w')


from start_loc import find_resources


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
            
        self.faction_names = {
            'player_0': 'TheBuilders',
            'player_1': 'FirstMars' 
        }
        
        
        # self.bots = {}
        # self.botpos = []
        # self.bot_factory = {}

        self.bot_task = {}  ## keep check of the usage of the robot, one of <ice, ore, rubble>; as well as factory group.
        self.bot_affiliations = {}
        self.factory_inventory = {}   ## keep track of total # of bots under each factory

        # self.factory_pos = {}
        # self.bots_meta_info = {}
        # self.factory_bot_map = {}

        # self.factory_queue = {}
        # self.move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        '''
        Early Phase
        '''
        
        actions = dict()
        if step == 0:
            # Declare faction
            actions['faction'] = self.faction_names[self.player]
            actions['bid'] = -100 # Learnable
        else:
            # Factory placement period
            # optionally convert observations to python objects with utility functions
            game_state = obs_to_game_state(step, self.env_cfg, obs) 
            opp_factories = [f.pos for _,f in game_state.factories[self.opp_player].items()]
            my_factories = [f.pos for _,f in game_state.factories[self.player].items()]
            
            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal
            
            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 100 metal n water (learnable)
                potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
                
                ice_map = game_state.board.ice
                ore_map = game_state.board.ore
                rubble_map = game_state.board.rubble
                valid_spawn_mask = game_state.board.valid_spawns_mask

                start_loc = find_resources(ice_map, ore_map, rubble_map, valid_spawn_mask)

                best_loc = start_loc.find_best_resource_loc()

                spawn_loc = best_loc
                # logging.info(f"factory position: {best_loc}")
                actions['spawn']=spawn_loc
                actions['metal']=min(150, metal_left)
                actions['water']=min(150, metal_left)
            
        return actions

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        
        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]

        factory_tiles, factory_units, factory_ids = [], [], []
        for unit_id, factory in factories.items():
            if unit_id not in self.factory_inventory.keys():
                ## TODO: initiate factory's inventory, declearing total bots, HEAVY/LIGHT, task_distribution,
                self.factory_inventory[unit_id] = {
                    "total_bots": 0, 
                    "heavy_bots": [], 
                    "light_bots": [], 
                    "ice": [], 
                    "ore": [], 
                    "rubble": []
                    }
                logging.info(f"{unit_id} position: {factory.pos}")
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
                factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
                ## TODO: update the 
                # self.factory_inventory[unit_id]["total_bots"] += 1
                ## as this bot id is not-known until the next act, will append to "heavy_bots" attribute there.   
                logging.info(f"{unit_id} build a heavy robot")           
            if factory.water_cost(game_state) <= factory.cargo.water / 5 - 200:
                actions[unit_id] = factory.water()
                logging.info(f"{unit_id} waters the lichen")   
            factory_tiles += [factory.pos]
            factory_units += [factory]
            factory_ids += [unit_id]
        factory_tiles = np.array(factory_tiles)
    
        # factory_tiles, factory_units = factory_act(factories, actions, game_state, self.env_cfg, 
        #                                            self.bot_usages, self.factory_bot_total, self.factory_bots_counts)

        units = game_state.units[self.player]

        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)

        ## initiate the bot's information
        for unit_id, unit in units.items():
            

            ## if the unit.pos is within factory_tiles && and don't have assigned task, 
            # that means the bot is just created and pending task assignments. 
            if unit_id not in self.bot_task and np.any(np.all(unit.pos==factory_tiles, axis=1)):
                logging.info(f"{unit_id} position at initialization: {unit.pos}")
                # logging.info(f"factory_tiles for reference: {factory_tiles}")

                factory_id_initial = factory_ids[np.argwhere(np.all(factory_tiles == unit.pos, axis=1))[0][0]]

                if unit.unit_type == "HEAVY":
                    self.factory_inventory[factory_id_initial]["heavy_bots"].append(unit_id)
                elif unit.unit_type == "LIGHT":
                    self.factory_inventory[factory_id_initial]["light_bots"].append(unit_id)
                
                ## TODO: assign task to the robot based on the factory's inventory
                if self.factory_inventory[factory_id_initial]["total_bots"] % 4 <= 1:
                    ## do ice mining if it's 1st or 2nd robot of a cycle.
                    self.bot_task[unit_id] = "ice"
                elif self.factory_inventory[factory_id_initial]["total_bots"] % 4 == 2:
                    self.bot_task[unit_id] = "ore"
                elif self.factory_inventory[factory_id_initial]["total_bots"] % 4 == 3:
                    self.bot_task[unit_id] = "rubble"

                self.factory_inventory[factory_id_initial]["total_bots"] += 1
                
                if unit_id not in self.bot_affiliations:
                    self.bot_affiliations[unit_id] = factory_id_initial

                ## store the meta info from the robot side, so it knows its assignement and home factory. 
                logging.info(f"{unit_id} is affiliated to {self.bot_affiliations[unit_id]}, will work in {self.bot_task[unit_id]}")

            if unit_id in self.bot_task: 
                ## TODO: if unit already have task, either dig/go to mine/return to factory
                ## TODO: some helper function dictate cargo, power, nearest factory, neareast resources info. 
                ## input: robot pos, factory pos, resources pos, 
                if self.bot_task[unit_id] in ("ice", "ore"):
                    logging.info(f"{unit_id} is working with {self.bot_affiliations[unit_id]}" + "\n" + 
                                 f"it has {unit.power} power, {unit.cargo} in cargo, at position {unit.pos}")
                    
                    
                elif self.bot_task[unit_id] == "rubble":
                    pass 

        #unit_act(factory_tiles, factory_units, units, ice_tile_locations, actions, game_state)
        return actions


def factory_act(factories, actions, game_state, env_cfg, bot_usages, factory_bot_total, factory_bots_counts):
    factory_tiles, factory_units = [], []
    for unit_id, factory in factories.items():
        if factory.power >= env_cfg.ROBOTS["HEAVY"].POWER_COST and \
        factory.cargo.metal >= env_cfg.ROBOTS["HEAVY"].METAL_COST:
            
            actions[unit_id] = factory.build_heavy()

            if unit_id not in factory_bot_total.keys():
                factory_bot_total[unit_id] = 1
                factory_bots_counts[unit_id] = {"ice": 1, "ore": 0, "rubble": 0}
            factory_bot_total[unit_id] += 1
            if factory_bot_total[unit_id] % 4 <= 1:
                bot_usages[unit_id] = "ice"
                factory_bots_counts[unit_id]["ice"] += 1
            elif factory_bot_total[unit_id] % 4 == 2:
                bot_usages[unit_id] = "ore"
                factory_bots_counts[unit_id]["ore"] += 1
            elif factory_bot_total[unit_id] % 4 == 3:
                bot_usages[unit_id] = "rubble"
                factory_bots_counts[unit_id]["rubble"] += 1
            
        
        # if factory.power >= env_cfg.ROBOTS["LIGHT"].POWER_COST and \
        # factory.cargo.metal >= env_cfg.ROBOTS["LIGHT"].METAL_COST:
        #     actions[unit_id] = factory.build_light()
        if factory.water_cost(game_state) <= factory.cargo.water / 5 - 200:
            actions[unit_id] = factory.water()
        factory_tiles += [factory.pos]
        factory_units += [factory]
    factory_tiles = np.array(factory_tiles)
    return factory_tiles, factory_units


def unit_act(factory_tiles, factory_units, units, ice_tile_locations, 
             actions, game_state):
    for unit_id, unit in units.items():
        # track the closest factory
        closest_factory = None
        adjacent_to_factory = False
        if len(factory_tiles) > 0:

            factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
            closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
            closest_factory = factory_units[np.argmin(factory_distances)]
            adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

            # previous ice mining code, if cargo is not filled up, keep mining
            if unit.cargo.ice < 50:
                ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                if np.all(closest_ice_tile == unit.pos):
                    if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.dig(repeat=0, n=1)]
                else:
                    direction = direction_to(unit.pos, closest_ice_tile)
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
            # else if we have enough ice, we go back to the factory and dump it.
            elif unit.cargo.ice >= 50:
                direction = direction_to(unit.pos, closest_factory_tile)
                if adjacent_to_factory:
                    if unit.power >= unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
                else:
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
        else:
            print("No factories left")