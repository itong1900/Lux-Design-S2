import numpy as np

def my_turn_to_place_factory(place_first: bool, step: int):
    if place_first:
        if step % 2 == 1:
            return True
    else:
        if step % 2 == 0:
            return True
    return False

# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1
        
def relative_pos(src, target):
    x_diff = target[0] - src[0]
    y_diff = target[1] - src[1]
    if x_diff != 0 and y_diff != 0:
        print("warning: not adjacent")

    if x_diff != 0:
        if x_diff < 0:
            return 4
        elif x_diff > 0:
            return 2
    elif y_diff != 0:
        if y_diff < 0:
            return 1
        elif y_diff > 0:
            return 3
    elif x_diff == 0 and y_diff == 0:
        print("warning: same tile")
        return 0


def at_same_spot(loc1, loc2):
    """
    return true if loc1 coord the same as loc2 coords
    """
    return np.all(loc1 == loc2, axis=1)

