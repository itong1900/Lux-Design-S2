from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory, relative_pos
import numpy as np
import sys
import math

import random

import logging

logging.basicConfig(filename="agentLog.log", level=logging.INFO, filemode='w')


from start_loc import find_resources

from aStarAlgo import aStarAlgorithm

## action_type = {0: "stay_center", 1:"go_up", 2:"go_right", 3:"go_down", 4: "go_left", 
# 5:"dig", 6:"transfer", 7:"pick_up"} 

## 5962256, good example for smart moving 7047892, deadlock example 842862, scenario debug 5130822 
## magic number 3606906, another deadlock 3046035+1676694
## good example for stand alone factories 7773979
## 1:1 pk 9104075: good example of water shortage is not well captured.
## magic number two 2335761
## magic number three self defence 5629335
## interesting case: 3191402
## 3/26 saved id 4162342
## 3/28 debug example 7717980 good example of cannot moveinto enemy factory, 9135040 earlier
## 3/28 lose due to infinite fight 5365301, defence 完回头撞7925541, bug here 9964357, perfect example of 3 robots. 


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
        
        ## === factory related objects =====
        self.factory_inventory = {}    ## keep track of total # of bots under each factory

        self.ally_factory_center_tiles = []   ## <center coords> These following 3 objects keep track of the up-to-date available ally factories in'column based' manner, refresh completely each round.
        self.factory_units = []    ## <factory unit obj>, 
        self.ally_factory_ids = []  ## <factory unit id> in the same order.
        
        ## === bot related objects ==========
        self.bot_affiliations = {}
        self.bot_action_queue = {}  ## store the action to proceed for each bot, 0-4 move, 5 dig, 6 pickup. 
        self.bot_status = {} ## <unit_id: in_progress, going_to_target, going_home>
        self.all_bot_locations = {} ## {unit_id: [0/1, coord], LIGHT/HEAVY}  0 if ally bot, 1 if enemy bot.
        self.bot_mission = {}  ## keep track of bot's mission <ice, ore, rubble>
        self.bot_target_tile = {}  ## keep track of the bot's target tile 
        self.bot_power_need_cache = {}

        self.rubble_conditions = {}  ## keep track of the rubble to clean up for each factory id {factory_id_1: (x,y): rub_num, ...}
        self.rubble_bot_target_tiles = {}

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
                actions['metal']=min(200, metal_left)
                actions['water']=min(200, water_left)
            
        return actions
    

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        
        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """
        logging.info(f"==========step in round: {step}==========")

        logging.info(f"start general debug: status at start of round {self.bot_status}")
        logging.info(f"start general debug: mission at start of round {self.bot_mission}")
        logging.info(f"start general debug: action_queue at start of round {self.bot_action_queue}")
        logging.info(f"start general debug: bot_target_tile at start of round {self.bot_target_tile}")
        logging.info(f"-----------------------------")

        game_state = obs_to_game_state(step, self.env_cfg, obs)

        ## factory acts
        ally_factories = game_state.factories[self.player]

        ## add enemy factories area to unavailable tiles. 
        invalid_tiles = []
        for factory_id, enemy_factory in game_state.factories[self.opp_player].items():
            ## whole enemy factory region is not valid to move onto
            invalid_tiles += [enemy_factory.pos]
            invalid_tiles += [[enemy_factory.pos[0]+1, enemy_factory.pos[1]]]
            invalid_tiles += [[enemy_factory.pos[0]-1, enemy_factory.pos[1]]]
            invalid_tiles += [[enemy_factory.pos[0], enemy_factory.pos[1]+1]]
            invalid_tiles += [[enemy_factory.pos[0], enemy_factory.pos[1]-1]]
            invalid_tiles += [[enemy_factory.pos[0]+1, enemy_factory.pos[1]+1]]
            invalid_tiles += [[enemy_factory.pos[0]-1, enemy_factory.pos[1]+1]]
            invalid_tiles += [[enemy_factory.pos[0]+1, enemy_factory.pos[1]-1]]
            invalid_tiles += [[enemy_factory.pos[0]-1, enemy_factory.pos[1]-1]]


        ## General Information
        ally_units = game_state.units[self.player]
        enemy_units = game_state.units[self.opp_player]

        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)

        ore_map = game_state.board.ore
        ore_tile_locations = np.argwhere(ore_map == 1)

        ## refresh every rounds
        self.all_bot_locations = {}

        ## TODO: maybe troublesome when using unit.pos in tiles 
        ally_botpos_validate_build_bot_pos = []
        for unit_id, unit in ally_units.items():
            self.all_bot_locations[unit_id] = [0, unit.pos, unit.unit_type]
            ally_botpos_validate_build_bot_pos.append([unit.pos[0], unit.pos[1]])
            ## if bot steps on resources tile, still consider as valid tile
            if unit.pos in ice_tile_locations or unit.pos in ore_tile_locations:
                continue
            invalid_tiles += [unit.pos]
            
        for unit_id, unit in enemy_units.items():
            self.all_bot_locations[unit_id] = [1, unit.pos, unit.unit_type]
            ## if bot steps on resources tile, still consider as valid tile
            if unit.pos in ice_tile_locations or unit.pos in ore_tile_locations:
                continue
            invalid_tiles += [unit.pos]

        invalid_tiles = [list(t) for t in set(tuple(element) for element in invalid_tiles)]
        ## ==============factory_act call out ================= 
        self.ally_factory_center_tiles, self.factory_units, self.ally_factory_ids  = factory_act(ally_factories, self.factory_inventory, ally_botpos_validate_build_bot_pos, actions, game_state, self.env_cfg, step)
        # ally_factory_tiles, factory_units, ally_factory_ids


        # LOOKAHEAD = 40
        # BASELINE_HEAVY_BOT_CHARGE = 30 ## per extra recharge made to each heavy robot. 

        # TARGET_QUANTITY_HEAVY_BOT = 5
        ## evaluate the resources needed in the next 50 rounds, 
        for factory_id, ally_factory in ally_factories.items():
            # waterNeeded = LOOKAHEAD * (ally_factory.water_cost(game_state) + 1)
            # powerNeeded = len(self.factory_inventory[factory_id]["heavy_bots"]) * BASELINE_HEAVY_BOT_CHARGE
            # metalNeeded = (TARGET_QUANTITY_HEAVY_BOT - len(self.factory_inventory[factory_id]["heavy_bots"])) * 100
            

            # waterExpectToGain = min(ally_factory.cargo.ice, LOOKAHEAD * 100) * 0.25
            # metalExpectToGan = min(ally_factory.cargo.ore, LOOKAHEAD * 50) * 0.2

            targetRubbleTiles, layer_reaching_target_tile = searchRubbleTiles(game_state.board.rubble, ally_factory.pos)
            self.rubble_conditions[factory_id] = targetRubbleTiles

            # waterShortage = waterNeeded - (ally_factory.cargo.water + waterExpectToGain)
            # metalShortage = metalNeeded - (ally_factory.cargo.metal + metalExpectToGan)
            # lichenShortage = getLichenShortage()

            # logging.info(f"{factory_id} has waterShortage {waterShortage}, metalShortage {metalShortage}")

            # if waterShortage > metalShortage:
            #     self.factory_inventory[factory_id]["priority"] = "water"
            # else:
            #     self.factory_inventory[factory_id]["priority"] = "metal"
        
        # logging.info(f"map of blockers {invalid_tiles}")
        # logging.info(f"size of invalid tiles {len(invalid_tiles)}")
        ##  ============Unit Act =================
        map_with_blocks = np.zeros((48, 48))
        for x, y in invalid_tiles:
            map_with_blocks[x][y] = 1


        unit_act(ally_units, self.bot_mission, self.bot_target_tile, self.bot_affiliations, 
                 self.bot_status, self.bot_action_queue, self.bot_power_need_cache,
                 self.ally_factory_center_tiles, self.ally_factory_ids, 
                 self.factory_inventory, self.factory_units, self.all_bot_locations,
                 ice_tile_locations, ore_tile_locations, map_with_blocks, 
                 self.rubble_conditions, self.rubble_bot_target_tiles, invalid_tiles,
                 game_state, actions, step)

        logging.info(f"round ends general debug: status at end of round {self.bot_status}")
        logging.info(f"round ends general debug: mission at end of round {self.bot_mission}")
        logging.info(f"round ends general debug: action_queue at end of round {self.bot_action_queue}")
        logging.info(f"round ends general debug: bot_target_tile at end of round {self.bot_target_tile}")

        return actions


def path_planning_test(startLoc, targetLoc, map_with_blocks, game_state_board):
    startX, startY = startLoc
    endX, endY = targetLoc

    ## expand one unit for broader path search, but shouldn't exceed boundaries
    expansion = 1
    minX, maxX, minY, maxY = max(0, min(startX, endX)-expansion), \
        min(47, max(startX, endX)+expansion), max(0, min(startY, endY)-expansion), \
            min(47, max(startY, endY)+expansion)

    trimmedGraph = [xCol[minY: maxY+1] for xCol in map_with_blocks[minX: maxX+1]]
    logging.info(f"super_debug, walkable graph {trimmedGraph}")

    dimGraphX, dimGraphY = len(trimmedGraph), len(trimmedGraph[0])
    power_cost = np.zeros([dimGraphX, dimGraphY])
    for x in range(minX, maxX+1):
        for y in range(minY, maxY+1):
            ## need offset here
            power_cost[x-minX][y-minY] = math.floor(20 + game_state_board.rubble[x][y])  
    
    # np.zeros((maxX - minX + 1, maxY - minY + 1))  ## placeholder
    # np.abs(
    #     np.array([xCol[minY: maxY+1] for xCol in game_state_board.valid_spawns_mask[minX: maxX+1]]) - 1
    #     )

    offsetX, offsetY = minX, minY

    startX_offset, startY_offset = startX - offsetX, startY - offsetY
    endX_offset, endY_offset = endX - offsetX, endY - offsetY

    logging.info(f"path planning beta debug start {startLoc}")
    logging.info(f"path planning beta debug end {targetLoc}")  

    path, total_cost = aStarAlgorithm(startX_offset, startY_offset, endX_offset, endY_offset, trimmedGraph, power_cost)
    
    if len(path) > 20:
        path = path[ :20]

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

    return action_queues, total_cost


def path_planning(startLoc, targetLoc, map_with_blocks, game_state_board):
    startX, startY = startLoc
    endX, endY = targetLoc

    ## expand one unit for broader path search, but shouldn't exceed boundaries
    expansion = 1
    minX, maxX, minY, maxY = max(0, min(startX, endX)-expansion), \
        min(47, max(startX, endX)+expansion), max(0, min(startY, endY)-expansion), \
            min(47, max(startY, endY)+expansion)

    trimmedGraph = [xCol[minY: maxY+1] for xCol in map_with_blocks[minX: maxX+1]]

    dimGraphX, dimGraphY = len(trimmedGraph), len(trimmedGraph[0])
    power_cost = np.zeros([dimGraphX, dimGraphY])
    for x in range(minX, maxX+1):
        for y in range(minY, maxY+1):
            ## need offset here
            power_cost[x-minX][y-minY] = math.floor(20 + game_state_board.rubble[x][y])  
    
    # np.zeros((maxX - minX + 1, maxY - minY + 1))  ## placeholder
    # np.abs(
    #     np.array([xCol[minY: maxY+1] for xCol in game_state_board.valid_spawns_mask[minX: maxX+1]]) - 1
    #     )

    offsetX, offsetY = minX, minY

    startX_offset, startY_offset = startX - offsetX, startY - offsetY
    endX_offset, endY_offset = endX - offsetX, endY - offsetY

    logging.info(f"path planning non test debug startloc {startLoc}")
    logging.info(f"path planning non test debug endloc {targetLoc}")  

    path, total_cost = aStarAlgorithm(startX_offset, startY_offset, endX_offset, endY_offset, trimmedGraph, power_cost)
    
    if len(path) > 20:
        path = path[ :20]

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
        
    return action_queues


def factory_act(ally_factories, factory_inventory, ally_botpos_validate_build_bot_pos, actions, game_state, env_cfg, step):

    factory_tiles, factory_units, factory_ids = [], [], []
    
    for unit_id, factory in ally_factories.items():
        if unit_id not in factory_inventory.keys():
            ## TODO: initiate factory's inventory, declaring total bots, HEAVY/LIGHT, task_distribution,
            factory_inventory[unit_id] = {
                "total_bots": 0, 
                "heavy_bots": [], 
                "light_bots": [], 
                "ice": [], 
                "ore": [], 
                "rubble": [],
                "factory_pos": factory.pos
                }
        if factory.power >= env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= env_cfg.ROBOTS["HEAVY"].METAL_COST and \
                list([factory.pos[0], factory.pos[1]]) not in ally_botpos_validate_build_bot_pos:
            ## TODO: ensure the center is not occupied
            actions[unit_id] = factory.build_heavy()
            ## TODO: update the 
            # self.factory_inventory[unit_id]["total_bots"] += 1
            ## as this bot id is not-known until the next act, will append to "heavy_bots" attribute there.   
            
        elif factory.can_water(game_state) and step > 900 and factory.cargo.water > (1000-step)+100:
            actions[unit_id] = factory.water()
            
        factory_tiles += [factory.pos]
        factory_units += [factory]
        factory_ids += [unit_id]
    factory_tiles = np.array(factory_tiles)

    ## deep copy to avoid factory_inventory length change dynamically.
    factory_ids_list_old = list(factory_inventory.keys())

    for factory_id_in_record in factory_ids_list_old:
        ## not in the up-to-date factory id list, that means the factory no longer exisits
        if factory_id_in_record not in factory_ids:
            # affected_bots = factory_inventory[factory_id_in_record]["heavy_bots"]
            del factory_inventory[factory_id_in_record]

    return factory_tiles, factory_units, factory_ids


def unit_act(units, bot_mission, bot_target_tile, bot_affiliations, bot_status, bot_action_queue, bot_power_need_cache,
             factory_tiles, factory_ids, factory_inventory, factory_units, all_bot_locations, 
             ice_tile_locations, ore_tile_locations, map_with_blocks, rubble_conditions, rubble_bot_target_tiles, 
             invalid_tiles ,game_state, actions, round):
    
    bot_loc_next_round = {} ## store the locations of each bot as unit_id: <>,  for CAS use

    ## initiate the bot's information
    for unit_id, unit in units.items():

        ## ===== some basic info for the unit ==============
        move_cost = None
        new_queue_added = False ## flip to true if the new actions added to actions[unit_id]
        unexpected_return = False ## flip to true, if the bot has to abort mission and return home. 

        ## check if there's enemy bot at the surroundings (up, dowm, left, right)
        defend_coords = get_adjacent_tile(unit.pos)
        bot_under_attack = None
        for enemy_unit_id, [is_enemy, enemy_pos, _] in all_bot_locations.items():
            if is_enemy:
                if [enemy_pos[0], enemy_pos[1]] in defend_coords:
                    bot_under_attack = [enemy_unit_id, direction_to(unit.pos, enemy_pos)] ## store defend info.
                    break ## potentially encounting a few attacks, but will defend the first one. 
            else:
                continue


        ## find nearest ice resource for this unit
        ice_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
        sorted_ice = [ice_tile_locations[k] for k in np.argsort(ice_distances)]
        closest_ice = sorted_ice[0]

        ## find nearest ore resource for this unit
        ore_distances = np.mean((ore_tile_locations - unit.pos) ** 2, 1)
        sorted_ore = [ore_tile_locations[k] for k in np.argsort(ore_distances)]
        closest_ore = sorted_ore[0]


        ## ========= ACTIVATION / REBASE ==========
        ## if bot not have any affiliations, it means it just got created.
        ## rebase a bot in case its affilicated factory destroyed, rebase to the nearest factory
        rebased = False
        if unit_id not in bot_affiliations or bot_affiliations[unit_id] not in factory_ids:
            ## trace the home factory, then assign affiliation for the bot. TODO: make it easier, the closest one will be affiliated, in case
            ## some factory is destroyed halfway, need to reassign affiliation. 
            ## label case as this part handles two conditions. 
            case = None
            if unit_id not in bot_affiliations:
                case = 0
            elif bot_affiliations[unit_id] not in factory_ids:
                case = 1
                rebased = True

            factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
            nearest_factory_indexes = np.argsort(factory_distances)
            #sorted_factory = [factory_tiles[k] for k in np.argsort(factory_distances)]
            closest_factory_pos = factory_tiles[nearest_factory_indexes[0]]
            closest_factory_id = factory_ids[nearest_factory_indexes[0]]
            
            #factory_id_initial = factory_ids[np.argwhere(np.all(factory_tiles == unit.pos, axis=1))[0][0]]

            bot_affiliations[unit_id] = closest_factory_id

            ## update home factory inventory, when this robt is activated
            if unit.unit_type == "HEAVY":
                factory_inventory[closest_factory_id]["heavy_bots"].append(unit_id)
            elif unit.unit_type == "LIGHT":
                factory_inventory[closest_factory_id]["light_bots"].append(unit_id)

            factory_inventory[closest_factory_id]["total_bots"] += 1

            if case == 0:
                bot_status[unit_id] = "pending_mission"
                logging.info(f"{unit_id} is affiliated to {bot_affiliations[unit_id]}, it's activated now.")
            elif case == 1:
                bot_status[unit_id] = "going_home"  ## go to the new home factory first immediately.
                logging.info(f"{unit_id} rebased to {bot_affiliations[unit_id]}.")


        ## one more basic info to add
        home_factory_coord = factory_inventory[bot_affiliations[unit_id]]["factory_pos"]

        ## ======= STATUS UPDATES ========== after last action, see if status needs change.
        if bot_status[unit_id] != "pending_mission":
            
            if bot_status[unit_id] == "task_prep":
                ## NOTE: now if factory don't have enough power, will go anyways, 
                # once this is done, status automatically update to going to target.
                if unit.power >= bot_power_need_cache[unit_id]-50: ## avoid the queue cost
                    bot_status[unit_id] = "going_to_target"

            elif bot_status[unit_id] == "going_to_target":
                ## if bot already arrives, change to status to in progress, otherwise keep going to target.
                if  list([bot_target_tile[unit_id][0], bot_target_tile[unit_id][1]]) == list([unit.pos[0], unit.pos[1]]): #np.all(bot_target_tile[unit_id] == unit.pos):
                    bot_status[unit_id] = "in_progress"
                ## in case short of power due to fight or whatever reason, go home, 
                if map_with_blocks[home_factory_coord[0]][home_factory_coord[1]] == 1:
                    map_with_block_copy = map_with_blocks.copy()
                    map_with_block_copy[home_factory_coord[0]][home_factory_coord[1]] = 0
                    _, cost_to_back_home = path_planning_test(unit.pos, home_factory_coord, map_with_block_copy, game_state.board)
                else:
                    _, cost_to_back_home = path_planning_test(unit.pos, home_factory_coord, map_with_blocks, game_state.board)
                if unit.power < cost_to_back_home + 50:
                    bot_status[unit_id] = "going_home"
                    unexpected_return = True

            elif bot_status[unit_id] == "in_progress":
            
                if bot_mission[unit_id] != "rubble":
                    if unit.cargo.ice > 80 or unit.cargo.ore > 60: ## if collect enougth resources, change status to go home. ## TODO need to be dynamic here
                        bot_status[unit_id] = "going_home"
                else:
                    in_progress_tile = rubble_bot_target_tiles[unit_id][0]
                    
                    ## if this tile's rubble has cleaned up, pop the item, and see what's next, otherwise continue.
                    if game_state.board.rubble[in_progress_tile[0]][in_progress_tile[1]] == 0:
                        
                        rubble_bot_target_tiles[unit_id].pop(0)
                        ## if there's more to dig after popping the first,
                        if len(rubble_bot_target_tiles[unit_id]) > 0:
                            bot_status[unit_id] = "going_to_target"
                            bot_target_tile[unit_id] = rubble_bot_target_tiles[unit_id][0]
                        else:
                            bot_status[unit_id] = "going_home"
                
                ## in case short of power due to fight or whatever reason, go home, 
                if map_with_blocks[home_factory_coord[0]][home_factory_coord[1]] == 1:
                    map_with_block_copy = map_with_blocks.copy()
                    map_with_block_copy[home_factory_coord[0]][home_factory_coord[1]] = 0
                    _, cost_to_back_home = path_planning_test(unit.pos, home_factory_coord, map_with_block_copy, game_state.board)
                else:
                    _, cost_to_back_home = path_planning_test(unit.pos, home_factory_coord, map_with_blocks, game_state.board)
                if unit.power < cost_to_back_home + 50:
                    bot_status[unit_id] = "going_home"
                    unexpected_return = True


            elif bot_status[unit_id] == "going_home":
                if list([home_factory_coord[0], home_factory_coord[1]]) == list([unit.pos[0], unit.pos[1]]):  # np.all(home_factory_coord == unit.pos):<-troublesome command  ## if arrives home, change status to offloading.
                    bot_status[unit_id] = "offloading"
            elif bot_status[unit_id] == "offloading":
                if unit.cargo.ice == 0 and unit.cargo.ore == 0: ## if cargo empty, means offload finishes, mission complete
                    bot_status[unit_id] = "pending_mission"
        
        ## ======== TASK ASSIGNMENT ==========
        ## if unit_id hasn't been involved in tasks(just activated), or finish task. 
        if bot_status[unit_id] == "pending_mission":
            ## TODO: replace with when factory_inventory[bot_affiliations[unit_id]]["priority"] is ready
            if round < 300:
                water_or_rubble = "water"
            else:
                # radn = random.random()
                # if radn < 0.6: 
                #     water_or_rubble = "water"
                # elif radn < 0.8:
                #     water_or_rubble = "rubble"
                # else:
                #     water_or_rubble = "metal"
                water_or_rubble = "water" if random.random() < 0.6 else "rubble"
            priority_task_this_factory = water_or_rubble
            logging.info(f"{unit_id} belongs to {bot_affiliations[unit_id]}, and the top priority is {priority_task_this_factory}")

            ## assign the task to the robot.
            if priority_task_this_factory == "water":
                bot_mission[unit_id] = "ice"
                bot_status[unit_id] = "task_prep"
                bot_target_tile[unit_id] = closest_ice  ## TODO: convert to some function ice tile to dig.
            elif priority_task_this_factory == "metal":
                bot_mission[unit_id] = "ore"
                bot_status[unit_id] = "task_prep"
                bot_target_tile[unit_id] = closest_ore
            elif priority_task_this_factory == "rubble":
                bot_mission[unit_id] = "rubble"
                bot_status[unit_id] = "task_prep"


        if bot_status[unit_id] == "task_prep":
            ## always have new queue in this state
            new_queue_added = True

            ## initialize if it's a new bot
            pending_queue_to_add = []
            if unit_id not in bot_action_queue:
                bot_action_queue[unit_id] = []
            ## the bot should have empty queue as it's either created just now or it finished up a mission earlier and get assigned a new mission.
            if len(bot_action_queue[unit_id]) != 0:
                bot_action_queue[unit_id] = []
            
            if bot_mission[unit_id] != "rubble":
                # get the move_actions to queue and the expected cost to the target

                ## in case target tile is blocked.
                if map_with_blocks[bot_target_tile[unit_id][0]][bot_target_tile[unit_id][1]] == 1:
                    map_with_block_copy = map_with_blocks.copy()
                    map_with_block_copy[bot_target_tile[unit_id][0]][bot_target_tile[unit_id][1]] = 0
                    move_acts, cost_to_target = path_planning_test(unit.pos, bot_target_tile[unit_id], map_with_block_copy, game_state.board)
                else:
                    move_acts, cost_to_target = path_planning_test(unit.pos, bot_target_tile[unit_id], map_with_blocks, game_state.board)
                power_needed = cost_to_target * 2 + 60 * 4 + 40 + 100   ## cost for move + dig + queue cost + buffer
                bot_power_need_cache[unit_id] = power_needed
                ## append a power pick up act if power is not enough
                if unit.power < power_needed:
                    bot_action_queue[unit_id].append(6)
                    factory_power = factory_units[factory_ids.index(bot_affiliations[unit_id])].power
                    logging.info(f"factory_power {factory_power} should be an integer")
                    logging.info(f"power_needed {power_needed} ")
                    logging.info(f"{unit_id}'s power {unit.power} ")
                    logging.info(f"amount to pick up {min(factory_power, power_needed-unit.power)}")
                    pending_queue_to_add = [unit.pickup(4, int(min(factory_power, power_needed-unit.power)))]

                ## append the move acts to the 
                bot_action_queue[unit_id] += move_acts
                logging.info(f"debug task_prep3: {unit_id} bot_action_queue {bot_action_queue[unit_id]}")
                pending_queue_to_add += [unit.move(act, repeat=False) for act in move_acts]
                ## consolidate all prep work into the actual queue, trim to first 20 if longer than 20
                if len(pending_queue_to_add) > 20:
                    pending_queue_to_add = pending_queue_to_add[:20]
                actions[unit_id] = pending_queue_to_add
                ## finally, update the next round's position, for none move actions, move act will always be 0
                next_move = 0 if bot_action_queue[unit_id][0] > 4 else bot_action_queue[unit_id][0]
                bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, next_move)
            else:  ## when mission is rubble
                rubble_tile_to_dig = []
                total_rubbles = 0
                round_digging_needed = 0 
                total_distance_within_rubble_tiles = 0
                last_tile = None
                for idx, (tile_loc, rubble_num) in enumerate(rubble_conditions[bot_affiliations[unit_id]].items()):
                    rubble_tile_to_dig.append(list(tile_loc))
                    round_digging_needed += math.ceil(rubble_num/20)
                    total_rubbles += rubble_num
                    if idx > 0:
                        total_distance_within_rubble_tiles += manhattan_distance(last_tile, tile_loc)
                    last_tile = tile_loc
                    if len(rubble_tile_to_dig) >= 3: 
                        break
                ## target_tile will be a list of coord instead of a single coord
                rubble_bot_target_tiles[unit_id] = rubble_tile_to_dig ## store to tiles to dig this mission
                
                bot_target_tile[unit_id] = rubble_tile_to_dig[0]
                # cost
                if map_with_blocks[bot_target_tile[unit_id][0]][bot_target_tile[unit_id][1]] == 1:
                    map_with_block_copy = map_with_blocks.copy()
                    map_with_block_copy[bot_target_tile[unit_id][0]][bot_target_tile[unit_id][1]] = 0
                    move_acts, cost_to_target = path_planning_test(unit.pos, bot_target_tile[unit_id], map_with_block_copy, game_state.board)
                else:
                    move_acts, cost_to_target = path_planning_test(unit.pos, bot_target_tile[unit_id], map_with_blocks, game_state.board)
                estimated_power_needed = round_digging_needed * 60 + total_distance_within_rubble_tiles * 30 + cost_to_target * 2 + 10 * 5 + 100
                bot_power_need_cache[unit_id] = estimated_power_needed
                ## append a power pick up act if power is not enough
                if unit.power < bot_power_need_cache[unit_id]:
                    bot_action_queue[unit_id].append(6)
                    factory_power = factory_units[factory_ids.index(bot_affiliations[unit_id])].power
                    pending_queue_to_add = [unit.pickup(4, int(min(factory_power, estimated_power_needed-unit.power)))]
                ## append the move acts to the 
                bot_action_queue[unit_id] += move_acts
                pending_queue_to_add += [unit.move(act, repeat=False) for act in move_acts]
                ## consolidate all prep work into the actual queue, trim to first 20 if longer than 20
                if len(pending_queue_to_add) > 20:
                    pending_queue_to_add = pending_queue_to_add[:20]
                actions[unit_id] = pending_queue_to_add
                ## finally, update the next round's position, for none move actions, move act will always be 0
                next_move = 0 if bot_action_queue[unit_id][0] > 4 else bot_action_queue[unit_id][0]
                bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, next_move)

            
        ## ======== CONTROL TASK QUEUEING ==========
        elif bot_status[unit_id] == "going_to_target":

            # if unit_id not in bot_action_queue:
            #     bot_action_queue[unit_id] = []

            ## plan path again when doing long distance travel, it happens when queue run out but still not at target. 
            if len(bot_action_queue[unit_id]) == 0:
                new_queue_added = True
                ## in case target tile is blocked.
                if map_with_blocks[bot_target_tile[unit_id][0]][bot_target_tile[unit_id][1]] == 1:
                    map_with_block_copy = map_with_blocks.copy()
                    map_with_block_copy[bot_target_tile[unit_id][0]][bot_target_tile[unit_id][1]] = 0
                    bot_action_queue[unit_id] = path_planning(unit.pos, bot_target_tile[unit_id], map_with_block_copy, game_state.board)
                else:  ## running out of queue or other reasons needs new path
                    bot_action_queue[unit_id] = path_planning(unit.pos, bot_target_tile[unit_id], map_with_blocks, game_state.board)
                actions[unit_id] = [unit.move(act, repeat=False) for act in bot_action_queue[unit_id]]
            move_cost = move_cost_complete(unit, game_state, bot_action_queue[unit_id]) #unit.move_cost(game_state, bot_action_queue[unit_id][0])
            total_power_needed = (move_cost + 10) if new_queue_added else move_cost 
            
            try:
                # bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, bot_action_queue[unit_id][0])
                if unit.power >= total_power_needed:
                    bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, bot_action_queue[unit_id][0])
                else:
                    bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, 0)
            except:
                bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, 0)
                

        elif bot_status[unit_id] == "in_progress":
            if len(bot_action_queue[unit_id]) == 0:
                if bot_mission[unit_id] == "ice":
                    rounds_to_dig = math.ceil((81 - unit.cargo.ice) / 20)
                elif bot_mission[unit_id] == "ore":
                    rounds_to_dig = math.ceil((61 - unit.cargo.ore) / 20)
                elif bot_mission[unit_id] == "rubble":
                    rounds_to_dig = math.ceil(game_state.board.rubble[unit.pos[0]][unit.pos[1]]/20)
                bot_action_queue[unit_id] = [5 for _ in range(rounds_to_dig)]
                actions[unit_id] = [unit.dig(repeat=False) for _ in range(rounds_to_dig)]

            ## when doing self defense: maybe buggy here. TODO: FIX 3/27
            bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, 0)


        elif bot_status[unit_id] == "going_home":
            ## plan path if it doesn't have path in plan, or it just got rebased, update actions to new home.
            if len(bot_action_queue[unit_id]) == 0 or rebased or unexpected_return:
                new_queue_added = True
                if map_with_blocks[home_factory_coord[0]][home_factory_coord[1]] == 1:
                    map_with_block_copy = map_with_blocks.copy()
                    map_with_block_copy[home_factory_coord[0]][home_factory_coord[1]] = 0
                    bot_action_queue[unit_id] = path_planning(unit.pos, home_factory_coord, map_with_block_copy, game_state.board)
                else:
                    bot_action_queue[unit_id] = path_planning(unit.pos, home_factory_coord, map_with_blocks, game_state.board)
                actions[unit_id] = [unit.move(act, repeat=False) for act in bot_action_queue[unit_id]]
            
            move_cost = move_cost_complete(unit, game_state, bot_action_queue[unit_id])
            total_power_needed = (move_cost + 10) if new_queue_added else move_cost
            
            try:
                if unit.power >= total_power_needed:  ## so normal action can be executed.
                    bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, bot_action_queue[unit_id][0])
                else:
                    bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, 0)
            except:
                bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, 0)


        elif bot_status[unit_id] == "offloading":
            if bot_mission[unit_id] == "ice":
                actions[unit_id] = [unit.transfer(0, 0, unit.cargo.ice, repeat=False)]
            elif bot_mission[unit_id] == "ore":
                actions[unit_id] = [unit.transfer(0, 1, unit.cargo.ore, repeat=False)]
            bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, 0)


        ## =============== Self Defence System ======================
        ## self defence system prevent robots from being destroyed by enemey bots,
        ## it works as whenever a bot detects an enemy bot in one of the surrounding tile, 
        ## it first evaluate its weight comparing to the enemy's weight, then decide to counter-attack or escape.
        if bot_under_attack is not None and \
        (len(bot_action_queue[unit_id]) == 0 or bot_action_queue[unit_id][0] == 0 or bot_action_queue[unit_id][0] > 4):
            logging.info(f"self defence system debug: mybot {unit_id} under attack {bot_under_attack}")
            enemy_id, enemy_dir = bot_under_attack
            enemy_pos = [all_bot_locations[enemy_id][1][0], all_bot_locations[enemy_id][1][1]]

            if enemy_pos not in invalid_tiles:  ## if enemy not at enemy factory, 

                ## inserting the defense move
                if all_bot_locations[enemy_id][2] == "HEAVY":
                    new_queue_added = True
                    _, counter_dir = attack_and_resume(enemy_dir)
                    bot_action_queue[unit_id].insert(0, counter_dir) 
                    bot_action_queue[unit_id].insert(0, enemy_dir) 
                    ## TODO: FIX 3/27: bot_loc_next_round update
                    actions[unit_id] = action_translation(bot_action_queue[unit_id], unit)
                    move_cost = move_cost_complete(unit, game_state, bot_action_queue[unit_id])
                    total_power_needed = (move_cost + 10) if new_queue_added else move_cost
                    try:
                        if unit.power >= total_power_needed: 
                            bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, bot_action_queue[unit_id][0])
                        else:
                            bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, 0)
                    except:
                        bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, 0)
                else:
                    pass ## ignore if the attack comes from LIGHT or the bot is already moving.

            else: ## if enemy bot is adjacent to its home factory, escape.
                if all_bot_locations[enemy_id][2] == "HEAVY":
                    new_queue_added = True
                    _, counter_dir = attack_and_resume(enemy_dir)
                    ## nudge away and come back.
                    bot_action_queue[unit_id].insert(0, enemy_dir) 
                    bot_action_queue[unit_id].insert(0, counter_dir) 
                    actions[unit_id] = action_translation(bot_action_queue[unit_id], unit)
                    move_cost = move_cost_complete(unit, game_state, bot_action_queue[unit_id])
                    total_power_needed = (move_cost + 10) if new_queue_added else move_cost
                    try:
                        if unit.power >= total_power_needed: 
                            bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, bot_action_queue[unit_id][0])
                        else:
                            bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, 0)
                    except:
                        bot_loc_next_round[unit_id] = get_next_round_loc(unit.pos, 0)

    ## ============= Global CAS System ==============
    ## validate all the move globally to avoid collisions between ally bots
    ## Collisions only occur when bots moving onto a same tile at the same round. 
    ## To avoid this from happening, when more than 1 bots will move on the same tile next round, 
    ## these bots go through the CAS system, so they can go onto the conflict tile in order without collisions. 
    potential_collisions = {}
    for unit_id, loc in bot_loc_next_round.items():
        ## TODO: exist Nonetype error here. due to [x,y]: None in bot_loc_next_round
        if tuple(loc) not in potential_collisions:
            potential_collisions[tuple(loc)] = []
        potential_collisions[tuple(loc)].append(unit_id)
    ## if any loc got 2 or more bots, a potential collision is detected
    standby_units = set()
    for loc, unit_ids in potential_collisions.items():
        if len(unit_ids) > 1:
            resolve_conflicts(list(loc), unit_ids, bot_status, bot_action_queue, all_bot_locations, standby_units)
    # logging.info(f"debug1: potential_collisions {potential_collisions}")
    # logging.info(f"debug2: replanned_units {replanned_units}")

    logging.info(f"before exe general debug: status at end of round {bot_status}")
    logging.info(f"before exe general debug: mission at end of round {bot_mission}")
    logging.info(f"before exe general debug: action_queue at end of round {bot_action_queue}")
    logging.info(f"before exe general debug: bot_target_tile at end of round {bot_target_tile}")
    logging.info(f"before exe general debug: bot_loc_next_round at end of round {bot_loc_next_round}")
    logging.info(f"-----------------------------")

    ## finally execute the validated moves
    have_requeue_cost = set()
    for unit_id, unit in units.items():
        if unit_id in standby_units:   ## actions[unit_id] is updated
            actions[unit_id] = [unit.move(act, repeat=False) for act in bot_action_queue[unit_id]]
            have_requeue_cost.add(unit_id)

        if bot_status[unit_id] == "in_progress": 
            total_cost = unit.dig_cost(game_state) + (unit.action_queue_cost(game_state) if unit_id in have_requeue_cost else 0)
            if unit.power >= total_cost and len(bot_action_queue[unit_id]) > 0:
                bot_action_queue[unit_id].pop(0)  # the first action will be executed 
        elif bot_status[unit_id] == "going_to_target" or bot_status[unit_id] == "going_home":
            move_cost = move_cost_complete(unit, game_state, bot_action_queue[unit_id]) # 0 if bot_action_queue[unit_id][0] == 0 else unit.move_cost(game_state, bot_action_queue[unit_id][0])
            total_cost = move_cost + (unit.action_queue_cost(game_state) if unit_id in have_requeue_cost else 0)
            
            if unit.power >= total_cost and len(bot_action_queue[unit_id]) > 0:
                bot_action_queue[unit_id].pop(0)  # the first action will be executed 
        else:
            if len(bot_action_queue[unit_id]) > 0:
                bot_action_queue[unit_id].pop(0)
        

def resolve_conflicts(conflict_loc, unit_id_list, bot_status, bot_action_queue, all_bots_locations, standby_units):
    ## when multiple agents going onto the same tile, follow the principal,
    ## 1. if there's already a bot on that tile and will keep in that tile on the next round,
    ## most likely in these "offload" "task_prep" "in_progress" status, all others standby.
    ## 2. if none such bot, let the first 

    exist_bot_already_on_spot = None
    for unit_id in unit_id_list:
        if list([all_bots_locations[unit_id][1][0], all_bots_locations[unit_id][1][1]]) == conflict_loc:  ##np.all(all_bots_locations[unit_id][1] == conflict_loc):
            exist_bot_already_on_spot = unit_id
            break

    ## case 1, except for the one on spot, all others stand by
    if exist_bot_already_on_spot is not None:
        for unit_id in unit_id_list:
            if unit_id == exist_bot_already_on_spot:
                continue
            else:
                bot_action_queue[unit_id].insert(0, 0)  ## insert a pause action 
                standby_units.add(unit_id)
    else: ## case 2, none bot is on the spot, the first bot entered, others standby.
        spot_filled = False
        for unit_id in unit_id_list:
            if not spot_filled:
                spot_filled = True
                continue
            else:
                bot_action_queue[unit_id].insert(0, 0) 
                standby_units.add(unit_id)


    # for unit_id in unit_id_list:
    #     if bot_status[unit_id] == "offloading" or bot_status[unit_id] == "task_prep" or bot_status[unit_id] == "in_progress":
    #         pass ## for static agent, stay their way
    #     else:
    #         bot_action_queue[unit_id].insert(0, 0)  ## insert a pause action 
    #         replanned_units.add(unit_id)


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
    for unit_id, [is_enemy, loc, _] in all_bot_locations.items():
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

def searchRubbleTiles(rubble_map, factory_pos, min_target_num_tiles = 5):
    targetHit = False
    rubble_tile_to_clean = {}
    layer = 1

    center_x, center_y = factory_pos[0], factory_pos[1]
    while not targetHit:
        for x in range(max(0, center_x - 1 - layer), min(47, center_x + 2 + layer)):
            for y in range(max(0, center_y - 1 - layer), min(47, center_y + 2 + layer)):
                ## skip if inside the factory:
                if x <= center_x + 1 and x >= center_x - 1 and y <= center_y + 1 and y >= center_y - 1:
                    continue
                if rubble_map[x][y] > 0:
                    rubble_tile_to_clean[tuple([x, y])] = rubble_map[x][y]
        if len(rubble_tile_to_clean) >= min_target_num_tiles:
            targetHit = True
        layer += 1
    return rubble_tile_to_clean, layer


def getLichenShortage():
    return 0


def get_next_round_loc(currentPos, dir):
    xCur, yCur = currentPos
    if dir == 0:
        return [xCur, yCur]
    elif dir == 1:  # up
        return [xCur, yCur-1]
    elif dir == 2: # right
        return [xCur+1, yCur]
    elif dir == 3: ## down
        return [xCur, yCur+1]
    elif dir == 4: ## left
        return [xCur-1, yCur]
    else:
        return [xCur, yCur]



def manhattan_distance(start, end):
    startX, startY = start
    endX, endY = end
    return abs(startX - endX) + abs(startY- endY)


def get_adjacent_tile(loc):
    """
    Return the adjacent coordinates
    """
    x, y = loc[0], loc[1]
    result = []
    if x > 0:  ## not at left most
        result.append([x-1, y]) ## get left 
    if x < 47: ## not at right most
        result.append([x+1, y])  ## get right
    if y > 0: ## not at top
        result.append([x, y-1])  ## get one unit up
    if y < 47: ## not at bottom
        result.append([x, y+1])  ## get one unit down

    return result

def attack_and_resume(dir):
    """
    return a pair of move actions attacking the "dir" and 
    resume position by going counter direction of "dir"
    """
    if dir == 1: 
        next = 3
    elif dir == 2:
        next = 4
    elif dir == 3:
        next = 1
    elif dir == 4: 
        next = 2
    
    return [dir, next]


def action_translation(action_queue, unit):
    """
    translate the action to actual command, 
    0-4: move action exactly same
    5: dig:
    6: pick up, 
    """
    result = []
    for act in action_queue:
        if act >= 0 and act <= 4:
            result.append(unit.move(act, repeat=False))
        elif act == 5:
            result.append(unit.dig(repeat=False))
        else:
            pass ## cannot handle yet
    return result

def tell_day_night(round):
    """
    return 1 if the round is day, 0 if night
    """
    modulus = round % 50
    if modulus >= 31  or modulus <= 49:
        return 0
    elif modulus <= 30:
        return 1
    
def move_cost_complete(unit, game_state, unit_action_queue):
    """
    return the move_cost when more context is given, 
    i.e. when first_next_act is none or 0 or > 4
    """
    if len(unit_action_queue) == 0:
        return 0
    
    first_next_act = unit_action_queue[0]
    if first_next_act >= 1 and first_next_act <= 4:
        value_to_return = unit.move_cost(game_state, first_next_act)
        return value_to_return if value_to_return is not None else 0
    else:
        return 0