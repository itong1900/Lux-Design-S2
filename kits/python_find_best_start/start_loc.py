import numpy as np

from scipy.ndimage import distance_transform_cdt
from scipy.spatial import KDTree


def manhattan_dist_to_nth_closest(binary_mask, n=1):
    if n == 1:
        # Get the distance map from every pixel to the nearest negative pixel
        distance_map = distance_transform_cdt(1-binary_mask, metric = "taxicab")
        return distance_map
    else:
        true_coords = np.transpose(np.nonzero(binary_mask))
        tree = KDTree(true_coords)  
        # query the nearest to nth closest distances using p=1 for Manhattan distance
        dist, _ = tree.query(np.transpose(np.nonzero(~binary_mask)), k=n, p=1) 
        return np.reshape(dist[:, n-1], binary_mask.shape) # reshape the result to match the input shape and add an extra dimension for the different closest distances
        
def count_region_cells(is_low_rubble_arr, start, min_dist=2, max_dist=np.inf, exponent=1):
    
    def dfs(is_low_rubble_arr, loc):
        distance_from_start = abs(loc[0]-start[0]) + abs(loc[1]-start[1])
        if not (0 <= loc[0] < is_low_rubble_arr.shape[0] and 
                0<=loc[1]<is_low_rubble_arr.shape[1]):   # check to see if we're still inside the map
            return 0
        if (not is_low_rubble_arr[loc]) or visited[loc]:     # we're only interested in low rubble, not visited yet cells
            return 0
        if not (min_dist <= distance_from_start <= max_dist):      
            return 0
        
        visited[loc] = True

        count = 1.0 * exponent ** distance_from_start
        count += dfs(is_low_rubble_arr, (loc[0]-1, loc[1]))
        count += dfs(is_low_rubble_arr, (loc[0]+1, loc[1]))
        count += dfs(is_low_rubble_arr, (loc[0], loc[1]-1))
        count += dfs(is_low_rubble_arr, (loc[0], loc[1]+1))

        return count
    
    visited = np.zeros_like(is_low_rubble_arr, dtype=bool)
    return dfs(is_low_rubble_arr, start)



class find_resources():
    def __init__(self, ice_map, ore_map, rubble_map, valid_spawn_mask) -> None:
        self.ice_map = ice_map
        self.ore_map = ore_map
        self.rubble_map = rubble_map
        self.valid_spawn_mask = valid_spawn_mask
        self.RESOURCE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25])
        self.ICE_PREFERENCE = 3
        self.LOW_RUBBLE = self.rubble_map < 25

    
    def find_best_resource_loc(self):
        ## higher means better
        # this is the distance to the n-th closest ice, for each coordinate
        ice_distances = [manhattan_dist_to_nth_closest(self.ice_map, i) for i in range(1,5)]

        # this is the distance to the n-th closest ore, for each coordinate
        ore_distances = [manhattan_dist_to_nth_closest(self.ore_map, i) for i in range(1,5)]

        weigted_ice_dist_scores = np.sum(
            np.array(ice_distances) * self.RESOURCE_WEIGHTS[:, np.newaxis, np.newaxis], 
            axis=0)
        weighted_ore_dist_scores = np.sum(
            np.array(ore_distances) * self.RESOURCE_WEIGHTS[:, np.newaxis, np.newaxis], 
            axis=0)
        

        ## get low rubble scores
        low_rubble_scores = np.zeros_like(self.LOW_RUBBLE, dtype=float)
        for i in range(self.LOW_RUBBLE.shape[0]):
            for j in range(self.LOW_RUBBLE.shape[1]):
                low_rubble_scores[i,j] = count_region_cells(self.LOW_RUBBLE, (i,j), min_dist=0, max_dist=8, exponent=0.9)


        resource_scores = (weigted_ice_dist_scores * self.ICE_PREFERENCE + weighted_ore_dist_scores)
        resource_scores = np.max(resource_scores)-resource_scores
        
        # resource_scores = resource_scores * self.valid_spawn_mask

        combined_scores = (low_rubble_scores * 2 + resource_scores) * self.valid_spawn_mask

        best_loc_score = np.argmax(combined_scores)
        x, y = np.unravel_index(best_loc_score, (48, 48))
        return [x, y]