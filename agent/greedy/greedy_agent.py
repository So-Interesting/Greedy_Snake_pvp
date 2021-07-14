import copy
import math
import numpy as np

def get_id(x, y, width):
    return x * width + y
    

def floyd(height, width, snakes):
    mat = np.zeros((height * width,height * width))
    for x1 in range(height):
        for y1 in range(width):
            for x2 in range(height):
                for y2 in range(width):
                    id1 = get_id(x1, y1, width)
                    id2 = get_id(x2, y2, width)
                    if (id1 != id2):
                        mat[id1][id2] = math.inf
                        mat[id2][id1] = math.inf
                    else : mat[id1][id2] = 0
    for x in range(height):
        for y in range(width):
            id1 = get_id(x, y, width)
            if (x == 0) : id2 = id1 + (height - 1) * width
            else : id2 = id1 - width
            mat[id1][id2] = 1
            if (y == 0) : id2 = id1 + width - 1
            else : id2 = id1 - 1
            mat[id1][id2] = 1
            if (x == height - 1) : id2 = id1 - (height - 1) * width
            else : id2 = id1 + width
            mat[id1][id2] = 1
            if (y == width - 1) : id2 = id1 - width + 1
            else : id2 = id1 + 1
            mat[id1][id2] = 1
    for i, (snake_x, snake_y) in enumerate(snakes[0]):
        id1 = get_id(snake_x,snake_y,width)
        for x in range(height):
            for y in range(width):
                id2 = get_id(x, y ,width)
                if (id1 != id2):
                    mat[id1][id2] = math.inf
                    mat[id2][id1] = math.inf
    for i, (snake_x, snake_y) in enumerate(snakes[1]):
        id1 = get_id(snake_x,snake_y,width)
        for x in range(height):
            for y in range(width):
                id2 = get_id(x, y ,width)
                if (id1 != id2):
                    mat[id1][id2] = math.inf
                    mat[id2][id1] = math.inf
    sum_id = height * width
    for k in range(sum_id):
        for i in range(sum_id):
            for j in range(sum_id):
                if (mat[i][k] + mat[k][j] < mat[i][j]):
                    mat[i][j] = mat[i][k] + mat[k][j]
    return mat

def get_min_bean(x, y, beans_position, width, height, snakes):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    mat = floyd(height, width, snakes)
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        snake_id = get_id(y, x, width)
        beans_id = get_id(bean_y, bean_x, width)
        distance = mat[snake_id][beans_id]
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_x, min_y, index


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position, width, height, snakes)
        beans_position.pop(index)

        next_distances = []
        mat = floyd(height, width, snakes)
        bean_id = get_id(bean_y, bean_x, width)
        head_y_tmp = (head_y - 1) % height
        head_id_tmp = get_id(head_y_tmp, head_x, width)
        up_distance = math.inf if head_surrounding[0] > 1 else \
            mat[head_id_tmp][bean_id]
            # math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
        next_distances.append(up_distance)
        head_y_tmp = (head_y + 1) % height
        head_id_tmp = get_id(head_y_tmp, head_x, width)
        down_distance = math.inf if head_surrounding[1] > 1 else \
            mat[head_id_tmp][bean_id]
            # math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
        next_distances.append(down_distance)
        head_x_tmp = (head_x - 1) % width
        head_id_tmp = get_id(head_y, head_x_tmp, width)
        left_distance = math.inf if head_surrounding[2] > 1 else \
            mat[head_id_tmp][bean_id]
            # math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(left_distance)
        head_x_tmp = (head_x + 1) % width
        head_id_tmp = get_id(head_y, head_x_tmp, width)
        right_distance = math.inf if head_surrounding[3] > 1 else \
            mat[head_id_tmp][bean_id]
            # math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(right_distance)
        actions.append(next_distances.index(min(next_distances)))
    return actions


def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions[i]
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        one_hot_action = [one_hot_action]
        joint_action.append(one_hot_action)
    return joint_action
