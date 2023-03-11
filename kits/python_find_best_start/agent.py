from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory, relative_pos
import numpy as np
import sys

import logging

logging.basicConfig(filename="agentLog.log", level=logging.INFO, filemode='w')


from start_loc import find_resources

from aStarAlgo import aStarAlgorithm


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
        self.tile_cost = [] ## 2d array shows the cost of each tile. 
        self.bot_action_queue = {}

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
    
#     def check_collision(self, pos, direction, unitpos, unit_type = 'LIGHT'):
#         move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
# #         move_deltas = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])
        
#         new_pos = pos + move_deltas[direction]
        
#         # if unit_type == "LIGHT":
#         #     return str(new_pos) in unitpos or str(new_pos) in self.botposheavy.values()
#         # else:
#         return str(new_pos) in unitpos

    def find_path(self, unit, target_tile):
        """
        based on unit pos and target_tile, return the path to target_tile with lowest cost in a list of coords format. 
        Use A* algorithm along with the cost map. Retrict the A* search space by reducing the size of path. 

        heuristic is mostly determined by distance to target and rubble.

        """
        pass

    
    def get_direction(self, unit, closest_tile, sorted_tiles):
                
        closest_tile = np.array(closest_tile)
        direction = direction_to(np.array(unit.pos), closest_tile)
        k=0
        # all_unit_positions = set(self.botpos.values())
        unit_type = unit.unit_type
        while k < min(len(sorted_tiles)-1, 500):
            k += 1
            closest_tile = sorted_tiles[k]
            closest_tile = np.array(closest_tile)
            direction = direction_to(np.array(unit.pos), closest_tile)
        
        # if self.check_collision(unit.pos, direction, all_unit_positions, unit_type):
        #     for direction_x in np.arange(4,-1,-1):
        #         if not self.check_collision(np.array(unit.pos), direction_x, all_unit_positions, unit_type):
        #             direction = direction_x
        #             break

        # if self.check_collision(np.array(unit.pos), direction, all_unit_positions, unit_type):
        #     direction = np.random.choice(np.arange(5))
        
        
        # self.botpos[unit.unit_id] = str(np.array(unit.pos) + move_deltas[direction])
            
        return direction

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        
        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """
        logging.info(f"step in round: {step}")

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
            move_cost = None

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
                if self.bot_task[unit_id] == "ice" and unit.cargo.ice < 50:
                    logging.info(f"{unit_id} is working with {self.bot_affiliations[unit_id]}" + "\n" + 
                                 f"it has {unit.power} power, {unit.cargo} in cargo, at position {unit.pos}")
                    ## TODO: Set up another structure to store the status of each bot: pending task, GoToTarget, Digging, GoToHome
                    ## if pending task, find task
                    ##   TODO: write a function get the top 3 ice resources closest to the unit, 
                    ##   another function to calculate the cost of getting there, power, time
                    ## elif GoToTarget:
                    ##   check if any emergency happens, if not keep going
                    ## elif Digging:
                    ##    check when need to stops, if not keep digging,
                    ## elif GoToHome:
                    ##    Keep going. 

                    ## find nearest ice resource, 
                    ice_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                    logging.info(f"Now find the nearest ice resource, ice_distances:")

                    sorted_ice = [ice_tile_locations[k] for k in np.argsort(ice_distances)]
                    closest_ice = sorted_ice[0]

                    logging.info(f"The cloest ice locates at: {closest_ice}")

                    if unit_id not in self.bot_action_queue:
                        self.bot_action_queue[unit_id] = []

                    if len(self.bot_action_queue[unit_id]) == 0:
                        self.bot_action_queue[unit_id] = path_planning(unit.pos, closest_ice, game_state.board)


                    if np.all(closest_ice == unit.pos):
                        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=False)]
                    else:

                        direction = self.get_direction(unit, closest_ice, sorted_ice)
                        move_cost = unit.move_cost(game_state, direction)

                elif self.bot_task[unit_id] == "ore":
                    pass
                    
                elif self.bot_task[unit_id] == "rubble":
                    pass 

                if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                    actions[unit_id] = [unit.move(self.bot_action_queue[unit_id].pop(0), repeat=False)]

        #unit_act(factory_tiles, factory_units, units, ice_tile_locations, actions, game_state)
        return actions
    

def path_planning(startLoc, targetLoc, game_state_board):
    startX, startY = startLoc
    endX, endY = targetLoc

    minX, maxX, minY, maxY = min(startX, endX), max(startX, endX), min(startY, endY), max(startY, endY)

    trimmedGraph = np.zeros((maxX - minX + 1, maxY - minY + 1))  ## placeholder
    # np.abs(
    #     np.array([xCol[minY: maxY+1] for xCol in game_state_board.valid_spawns_mask[minX: maxX+1]]) - 1
    #     )

    offsetX, offsetY = minX, minY

    startX_offset, startY_offset = startX - offsetX, startY - offsetY
    endX_offset, endY_offset = endX - offsetX, endY - offsetY

    logging.info(f"startX: {startX_offset}")
    logging.info(f"startY: {startY_offset}")
    logging.info(f"endX: {endX_offset}")
    logging.info(f"endY: {endY_offset}")


    logging.info(f"trimmed graph looks like \n {trimmedGraph}")
    path = aStarAlgorithm(startX_offset, startY_offset, endX_offset, endY_offset, trimmedGraph)

    ## resume original path
    actual_path = []
    for step in path:
        actual_path.append([step[0]+offsetX, step[1]+offsetY])

    logging.info(f"path: {actual_path}")

    ## convert path to actions
    action_queues = []
    i = 1
    while i < len(actual_path):
        action_queues.append(relative_pos(actual_path[i-1], actual_path[i]))
        i += 1
        
    logging.info(f"action quue: {action_queues}")

    return action_queues


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