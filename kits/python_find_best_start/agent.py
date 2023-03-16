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
        self.factory_pos_dict = {}
        self.tile_cost = [] ## 2d array shows the cost of each tile. 
        self.bot_action_queue = {}  ## store the action to proceed for each bot
        self.bot_status = {} ## <unit_id: in_progress, going_to_target, going_home>
        self.all_bot_locations = {} ## {unit_id: [0/1, coord]}  0 if ally bot, 1 if enemy bot.

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
                actions['spawn']=spawn_loc
                actions['metal']=min(120, metal_left)
                actions['water']=min(120, water_left)
            
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
        logging.info(f"==========step in round: {step}==========")

        game_state = obs_to_game_state(step, self.env_cfg, obs)

        ## factory acts
        ally_factories = game_state.factories[self.player]
        for factory_id, ally_factory in ally_factories.items():
            if factory_id not in self.factory_pos_dict:
                self.factory_pos_dict[factory_id] = ally_factory.pos
        enemy_factories = game_state.factories[self.opp_player]
        invalid_tiles = []
        for factory_id, enemy_factory in enemy_factories.items():
            ## whole enemy factory region is not movable
            invalid_tiles += [enemy_factory.pos]
            invalid_tiles += [[enemy_factory.pos[0]+1, enemy_factory.pos[1]]]
            invalid_tiles += [[enemy_factory.pos[0]-1, enemy_factory.pos[1]]]
            invalid_tiles += [[enemy_factory.pos[0], enemy_factory.pos[1]+1]]
            invalid_tiles += [[enemy_factory.pos[0], enemy_factory.pos[1]-1]]
            invalid_tiles += [[enemy_factory.pos[0]+1, enemy_factory.pos[1]+1]]
            invalid_tiles += [[enemy_factory.pos[0]-1, enemy_factory.pos[1]+1]]
            invalid_tiles += [[enemy_factory.pos[0]+1, enemy_factory.pos[1]-1]]
            invalid_tiles += [[enemy_factory.pos[0]-1, enemy_factory.pos[1]-1]]

        ## allied factories coordinates, facttories 
        ally_factory_tiles, factory_units, ally_factory_ids = factory_act(ally_factories, self.factory_inventory, actions, game_state, self.env_cfg)
        
        LOOKAHEAD = 40
        BASELINE_HEAVY_BOT_CHARGE = 30 ## per extra recharge made to each heavy robot. 

        TARGET_QUANTITY_HEAVY_BOT = 5
        ## evaluate the resources needed in the next 50 rounds, 
        for factory_id, ally_factory in ally_factories.items():
            waterNeeded = LOOKAHEAD * (ally_factory.water_cost(game_state) + 1)
            powerNeeded = len(self.factory_inventory[factory_id]["heavy_bots"]) * BASELINE_HEAVY_BOT_CHARGE
            metalNeeded = (TARGET_QUANTITY_HEAVY_BOT - len(self.factory_inventory[factory_id]["heavy_bots"])) * 100
            

            waterExpectToGain = min(ally_factory.cargo.ice, LOOKAHEAD * 100) * 0.25
            metalExpectToGan = min(ally_factory.cargo.ore, LOOKAHEAD * 50) * 0.2

            waterShortage = waterNeeded - (ally_factory.cargo.water + waterExpectToGain)
            metalShortage = metalNeeded - (ally_factory.cargo.metal + metalExpectToGan)
            lichenShortage = getLichenShortage()

            logging.info(f"{factory_id} has waterShortage {waterShortage}, metalShortage {metalShortage}")

            if waterShortage > metalShortage:
                self.factory_inventory[factory_id]["priority"] = "water"
            else:
                self.factory_inventory[factory_id]["priority"] = "metal"
        

        ## Robots acts
        ally_units = game_state.units[self.player]
        enemy_units = game_state.units[self.opp_player]

        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)

        ore_map = game_state.board.ore
        ore_tile_locations = np.argwhere(ore_map == 1)


        for unit_id, unit in ally_units.items():
            self.all_bot_locations[unit_id] = [0, unit.pos]
            invalid_tiles += [unit.pos]

        for unit_id, unit in enemy_units.items():
            self.all_bot_locations[unit_id] = [1, unit.pos]
            invalid_tiles += [unit.pos]

        invalid_tiles = [list(t) for t in set(tuple(element) for element in invalid_tiles)]
        logging.info(f"map of blockers {invalid_tiles}")
        logging.info(f"size of invalid tiles {len(invalid_tiles)}")
        map_with_blocks = np.zeros((48, 48))
        for x, y in invalid_tiles:
            map_with_blocks[x][y] = 1


        unit_act(ally_units, self.bot_task, self.bot_affiliations, self.bot_status, self.bot_action_queue, 
                 ally_factory_tiles, self.factory_pos_dict, ally_factory_ids, 
                 self.factory_inventory, self.all_bot_locations,
                 ice_tile_locations, ore_tile_locations, map_with_blocks, game_state, actions)

        
        return actions
    

def path_planning(startLoc, targetLoc, map_with_blocks, game_state_board):
    startX, startY = startLoc
    endX, endY = targetLoc

    minX, maxX, minY, maxY = min(startX, endX), max(startX, endX), min(startY, endY), max(startY, endY)

    trimmedGraph = [xCol[minY: maxY+1] for xCol in map_with_blocks[minX: maxX+1]]
    # np.zeros((maxX - minX + 1, maxY - minY + 1))  ## placeholder
    # np.abs(
    #     np.array([xCol[minY: maxY+1] for xCol in game_state_board.valid_spawns_mask[minX: maxX+1]]) - 1
    #     )

    offsetX, offsetY = minX, minY

    startX_offset, startY_offset = startX - offsetX, startY - offsetY
    endX_offset, endY_offset = endX - offsetX, endY - offsetY

    # logging.info(f"startX: {startX_offset}")
    # logging.info(f"startY: {startY_offset}")
    # logging.info(f"endX: {endX_offset}")
    # logging.info(f"endY: {endY_offset}")


    logging.info(f"trimmed graph looks like \n {trimmedGraph}")
    path = aStarAlgorithm(startX_offset, startY_offset, endX_offset, endY_offset, trimmedGraph)

    ## resume original path
    actual_path = []
    for step in path:
        actual_path.append([step[0]+offsetX, step[1]+offsetY])

    # logging.info(f"path: {actual_path}")

    ## convert path to actions
    action_queues = []
    i = 1
    while i < len(actual_path):
        action_queues.append(relative_pos(actual_path[i-1], actual_path[i]))
        i += 1
        
    logging.info(f"action quue: {action_queues}")

    return action_queues


def factory_act(factories, factory_inventory, actions, game_state, env_cfg):

    factory_tiles, factory_units, factory_ids = [], [], []
    
    for unit_id, factory in factories.items():
        if unit_id not in factory_inventory.keys():
            ## TODO: initiate factory's inventory, declearing total bots, HEAVY/LIGHT, task_distribution,
            factory_inventory[unit_id] = {
                "total_bots": 0, 
                "heavy_bots": [], 
                "light_bots": [], 
                "ice": [], 
                "ore": [], 
                "rubble": [],
                "factory_pos": factory.pos
                }
            logging.info(f"{unit_id} position: {factory.pos}")
        if factory.power >= env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= env_cfg.ROBOTS["HEAVY"].METAL_COST:

            actions[unit_id] = factory.build_heavy()
            ## TODO: update the 
            # self.factory_inventory[unit_id]["total_bots"] += 1
            ## as this bot id is not-known until the next act, will append to "heavy_bots" attribute there.   
            logging.info(f"{unit_id} build a heavy robot")           
        elif factory.water_cost(game_state) <= factory.cargo.water - 50:
            actions[unit_id] = factory.water()
            logging.info(f"{unit_id} waters the lichen")   
        factory_tiles += [factory.pos]
        factory_units += [factory]
        factory_ids += [unit_id]
    factory_tiles = np.array(factory_tiles)

    return factory_tiles, factory_units, factory_ids


def unit_act(units, bot_task, bot_affiliations, bot_status, bot_action_queue, factory_tiles, 
             factories_pos, factory_ids, factory_inventory, all_bot_locations,
             ice_tile_locations, ore_tile_locations, map_with_blocks, game_state, actions):
    ## initiate the bot's information
    for unit_id, unit in units.items():
        ## ===== some basic info for the unit ==============
        move_cost = None

        ## find nearest ice resource for this unit
        ice_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
        sorted_ice = [ice_tile_locations[k] for k in np.argsort(ice_distances)]
        closest_ice = sorted_ice[0]

        ## ========= ACTIVATION ==========
        ## if bot not have any affiliations, it means it just got created.
        if unit_id not in bot_affiliations:
            ## trace the home factory, then assign affiliation for the bot
            factory_id_initial = factory_ids[np.argwhere(np.all(factory_tiles == unit.pos, axis=1))[0][0]]
            bot_affiliations[unit_id] = factory_id_initial

            ## update home factory inventory, when this robt is activated
            if unit.unit_type == "HEAVY":
                factory_inventory[factory_id_initial]["heavy_bots"].append(unit_id)
            elif unit.unit_type == "LIGHT":
                factory_inventory[factory_id_initial]["light_bots"].append(unit_id)

            factory_inventory[factory_id_initial]["total_bots"] += 1

            bot_status[unit_id] = "pending_mission"

            logging.info(f"{unit_id} is affiliated to {bot_affiliations[unit_id]}, it's activated now.")

        ## one more basic info to add
        home_factory_coord = factory_inventory[bot_affiliations[unit_id]]["factory_pos"]

        ## ======= STATUS UPDATES ========== after last action, see if status needs change.
        if bot_status[unit_id] != "pending_mission":
            
            if bot_status[unit_id] == "going_to_target":
                ## if bot already arrives, change to status to in progress, otherwise keep going to target.
                if np.all(closest_ice == unit.pos):
                    bot_status[unit_id] = "in_progress"
            elif bot_status[unit_id] == "in_progress":
                if unit.cargo.ice > 80: ## if collect enougth resources, change status to go home.
                    bot_status[unit_id] = "going_home"
            elif bot_status[unit_id] == "going_home":
                if np.all(home_factory_coord == unit.pos):  ## if arrives home, change status to offloading.
                    bot_status[unit_id] = "offloading"
            elif bot_status[unit_id] == "offloading":
                if unit.cargo.ice == 0: ## if cargo empty, means offload finishes, mission complete
                    if unit.power < 500:
                        bot_status[unit_id] = "pick_up_power"
                    else:
                        bot_status[unit_id] = "pending_mission"
            elif bot_status[unit_id] == "pick_up_power":
                if unit.power < 500:
                    pass
                else:
                    bot_status[unit_id] = "pending_mission"

        ## ======== TASK ASSIGNMENT ==========
        ## if unit_id hasn't been involved in tasks(just activated), or finish task. 
        if bot_status[unit_id] == "pending_mission":

            priority_task_this_factory = "water" ## factory_inventory[bot_affiliations[unit_id]]["priority"]
            logging.info(f"{unit_id} belongs to {bot_affiliations[unit_id]}, and the top priority is {priority_task_this_factory}")

            ## assign the task to the robot.
            if priority_task_this_factory == "water":
                bot_task[unit_id] = "ice"
                bot_status[unit_id] = "going_to_target"
            elif priority_task_this_factory == "metal":
                bot_task[unit_id] = "ore"
                bot_status[unit_id] = "going_to_target"
        
        ## ======== CONTROL TASK QUEUEING ==========
        if bot_status[unit_id] == "going_to_target":

            if unit_id not in bot_action_queue:
                bot_action_queue[unit_id] = []
            ## plan path if it doesn't have path in plan
            if len(bot_action_queue[unit_id]) == 0:
                bot_action_queue[unit_id] = path_planning(unit.pos, closest_ice, map_with_blocks, game_state.board)
                actions[unit_id] = [unit.move(act, repeat=False) for act in bot_action_queue[unit_id]]
                bot_action_queue[unit_id] = []
            
            # if len(bot_action_queue[unit_id]) > 0:
            #     CAS_status = check_collision(unit.pos, bot_action_queue[unit_id][0], all_bot_locations, "going_to_target", bot_action_queue)

            #     if CAS_status == "clear":
            #         move_cost = unit.move_cost(game_state, bot_action_queue[unit_id][0])  ## cost to move on the tile
            #         if unit.power >= move_cost + unit.action_queue_cost(game_state):
            #             actions[unit_id] = [unit.move(bot_action_queue[unit_id].pop(0), repeat=False)]
            #     elif CAS_status == "pause":
            #         # stand by
            #         pass

        elif bot_status[unit_id] == "in_progress":
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                actions[unit_id] = [unit.dig(repeat=False)]

            # CAS_status = check_collision(unit.pos, 0, all_bot_locations, "in_progress", bot_action_queue)

            # if CAS_status == "clear":
            #     if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
            #         actions[unit_id] = [unit.dig(repeat=False)]
            # elif CAS_status == "escape":
            #     pass
            
            # do nothing if not enough power
        elif bot_status[unit_id] == "going_home":
            ## plan path if it doesn't have path in plan
            if len(bot_action_queue[unit_id]) == 0:
                bot_action_queue[unit_id] = path_planning(unit.pos, home_factory_coord, map_with_blocks, game_state.board)
                actions[unit_id] = [unit.move(act, repeat=False) for act in bot_action_queue[unit_id]]
                bot_action_queue[unit_id] = []

            # if len(bot_action_queue[unit_id]) > 0:
            #     CAS_status = check_collision(unit.pos, bot_action_queue[unit_id][0], all_bot_locations, "going_home", bot_action_queue)

            #     if CAS_status == "clear":
            #         move_cost = unit.move_cost(game_state, bot_action_queue[unit_id][0])  ## cost to move on the tile
            #         if unit.power >= move_cost + unit.action_queue_cost(game_state):
            #             actions[unit_id] = [unit.move(bot_action_queue[unit_id].pop(0), repeat=False)]
            #     elif CAS_status == "pause":
            #         pass

        elif bot_status[unit_id] == "offloading":
            actions[unit_id] = [unit.transfer(0, 0, unit.cargo.ice, repeat=False)]
        elif bot_status[unit_id] == "pick_up_power":
            actions[unit_id] = [unit.pickup(4, 500)]
        else: 
            pass
            # elif bot_task[unit_id] == "ore":
            #     pass
                
            # elif bot_task[unit_id] == "rubble":
            #     pass 

def check_collision(unit_pos, potential_next_move, all_bot_locations, bot_status, bot_action_queue):
    ## if next pos to move on has an ally bot, stop.
    if potential_next_move == 0:
        next_pos = unit_pos
    elif potential_next_move == 1:
        next_pos = [unit_pos[0], unit_pos[1] - 1]
    elif potential_next_move == 2:
        next_pos = [unit_pos[0] + 1, unit_pos[1]]
    elif potential_next_move == 3:
        next_pos = [unit_pos[0], unit_pos[1] + 1]
    elif potential_next_move == 4:
        next_pos = [unit_pos[0] - 1, unit_pos[1]]

    #action_temp = "clear"
    for unit_id, [is_enemy, loc] in all_bot_locations.items():
        # if manhattan_distance(unit_pos, loc) == 1:
        #     pass

        if not is_enemy: 
            if np.all(loc == next_pos):  ## if next move will move onto a friend bot's position
                action_temp = "pause"
                return action_temp
            else:
                return "clear"
        else:
            if manhattan_distance(loc, unit_pos) == 1:
                if bot_status == "in_progress": 
                    action_temp = "escape"
                    return action_temp
                elif (bot_status == "going_home" or bot_status == "going_to_target"):
                    if next_pos != loc: ## if not moving towards the enemy, do nothing, clear, keep doing
                        return "clear"
                    else:
                        action_temp = "correct_trajectory"  ## correct trajectory if moving towards enemey
                        return action_temp    


def manhattan_distance(loc1, loc2):
    """
    return the manhattan distance of two locations
    """
    x1, y1 = loc1
    x2, y2 = loc2
    
    return abs(x2-x1) + abs(y2-y1)



def getLichenShortage():
    return 0











# def unit_act(factory_tiles, factory_units, units, ice_tile_locations, 
    #          actions, game_state):
    # for unit_id, unit in units.items():
    #     # track the closest factory
    #     closest_factory = None
    #     adjacent_to_factory = False
    #     if len(factory_tiles) > 0:

    #         factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
    #         closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
    #         closest_factory = factory_units[np.argmin(factory_distances)]
    #         adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

    #         # previous ice mining code, if cargo is not filled up, keep mining
    #         if unit.cargo.ice < 50:
    #             ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
    #             closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
    #             if np.all(closest_ice_tile == unit.pos):
    #                 if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
    #                     actions[unit_id] = [unit.dig(repeat=0, n=1)]
    #             else:
    #                 direction = direction_to(unit.pos, closest_ice_tile)
    #                 move_cost = unit.move_cost(game_state, direction)
    #                 if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
    #                     actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
    #         # else if we have enough ice, we go back to the factory and dump it.
    #         elif unit.cargo.ice >= 50:
    #             direction = direction_to(unit.pos, closest_factory_tile)
    #             if adjacent_to_factory:
    #                 if unit.power >= unit.action_queue_cost(game_state):
    #                     actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
    #             else:
    #                 move_cost = unit.move_cost(game_state, direction)
    #                 if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
    #                     actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
    #     else:
    #         print("No factories left")